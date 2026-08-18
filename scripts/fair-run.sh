#!/bin/bash
# fair-run.sh — run the same benchmark against several engines, comparably.
#
# The tool measures; this script is what makes the measurement mean something. Its rules were
# not chosen for elegance — each one exists because a run without it produced a number that
# was wrong in a way nobody noticed.
#
# ONE ENGINE AT A TIME, PROCESSES STOPPED. Not just models unloaded: a peer engine's process
# still holds memory and perturbs the next one's placement. Measured once, a model that should
# have gone to the fast card went to the slow one because another engine's process was up —
# and the resulting "win" was a hardware difference wearing a software label.
#
# THE SAME CARDS FOR EVERYONE. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced for all three so an
# index means the same physical card in each, and which cards each engine may use comes from
# scripts/gpu-policy.sh — one declaration, applied to every launch. Restricting one engine to
# a card while the others keep the machine is the easiest way to publish a fiction.
#
# DERIVED, NOT LISTED. Whether ollama is pinned to one card is decided by the weights against
# one card's usable capacity, read at run time. A pin that is right for a model that fits and
# wrong for one that does not cannot be a constant.
#
# COLD, GREEDY, STREAMED. Every engine is restarted per engine, ollama runs at its own
# defaults, warm-up is discarded, temperature is 0 so no engine is measured through a
# different sampler, and `--stream` makes the decode rate a client-side quantity that all
# three can be held to.
#
# vLLM fixes its model at launch, so it gets one isolated run per model, and it cannot read
# mixed-quant GGUF — point it at the equivalent HF checkpoint with
# VLLM_SERVE="hf:<repo> --name <tag-the-others-use>".
#
# Usage:
#   BENCH_MODE=cpu  scripts/fair-run.sh --models <tag> --prompts short,long
#   BENCH_MODE=gpu  GPUS=0 VLLM_SERVE="hf:<repo> --name <tag>" \
#       OUT=results/three-way.json scripts/fair-run.sh --models <tag> --prompts short,medium,long
#   BENCH_MODE=gpu  scripts/fair-run.sh --models <big-tag> ...     # every card, no GPUS pin
#
# Do not pass -o; use OUT=. CPU mode forces --num-gpu 0 on ollama, which is only fair because
# the other engines are launched CPU-only alongside it.
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-$HOME/.ollama/models}"   # where ollama keeps its blobs
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo ollama)}"
LOKEN_BIN="${LOKEN_BIN:-../loken/target/release/server}"   # one CUDA binary, both modes
ASSAY="${ASSAY:-target/release/assay}"
VLLM_PORT=8000
BENCH_MODE="${BENCH_MODE:-cpu}"
GPU_PIN="${GPU_PIN:-}"                       # older spelling of GPUS=<n>: expose only that card
OUT="${OUT:-results/fairbench.json}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"          # each engine's port, so two runs can coexist
LOKEN_PORT="${LOKEN_PORT:-11435}"
[ "$BENCH_MODE" = cpu ] || [ "$BENCH_MODE" = gpu ] || { echo "BENCH_MODE must be cpu|gpu" >&2; exit 1; }

# WHICH CARDS EACH ENGINE MAY USE — read from gpu-policy.sh, which is the only place that
# decision is written. It resolves a card list to device UUIDs, because CUDA_VISIBLE_DEVICES=0
# means a different physical card depending on the ordering, and translates one policy into
# each engine's own lever: OLLAMA_SCHED_SPREAD when ollama can see more than one card (seeing
# them is not using them — its scheduler otherwise settles on one), and --tensor-parallel-size
# for vLLM.
#
# GPU_PIN is kept as the older spelling of "expose only this card"; fold it in before the
# policy is read, so there is still exactly one place the decision is made.
[ -n "$GPU_PIN" ] && GPUS="${GPUS:-$GPU_PIN}"
. "$(dirname "$(readlink -f "$0")")/gpu-policy.sh"
NUMGPU=(); LLMSERVE=()
if [ "$BENCH_MODE" = cpu ]; then NUMGPU=(--num-gpu 0); LLMSERVE=(--cpu); fi

# First model of the cell, parsed from --models: the placement decision below needs it,
# and so does the pre-bench probe further down.
_probe_model=""
_seen_models=0
for _a in "$@"; do
  if [ "$_seen_models" = 1 ]; then _probe_model="${_a%%,*}"; break; fi
  [ "$_a" = "--models" ] && _seen_models=1
done

# Weight bytes behind an ollama tag, read from its manifest.
blob_bytes_of() {
  local tag mf blob
  case "$1" in *:*) tag="${1/:/\/}";; *) tag="$1/latest";; esac
  mf="$MODELS_DIR/manifests/registry.ollama.ai/library/$tag"
  [ -f "$mf" ] || { echo 0; return; }
  blob=$(python3 -c "import json;d=json.load(open('$mf'));print([l['digest'] for l in d['layers'] if 'model' in l['mediaType']][0].replace('sha256:','sha256-'))" 2>/dev/null) || { echo 0; return; }
  stat -c%s "$MODELS_DIR/blobs/$blob" 2>/dev/null || echo 0
}

# HOW MANY CARDS — derived from the weights, never from a list of model names.
#
# Restricting ollama to one card is the right handicap while the weights fit that card: every
# engine is then measured on the same silicon. Above it the same restriction inverts — ollama
# gets one card plus host spill while the others get the machine, and the resulting "win" is a
# missing card rather than a placement. Measured once on a model larger than one card: ollama
# offloaded most layers to a single card and mapped the rest to the host, while the second card
# sat idle. So restrict only when it fits; otherwise let every engine have everything.
_one_card=0
if [ "$BENCH_MODE" = gpu ]; then
  while read -r _mb; do
    # A runtime keeps a slice of each card for context and activations; ollama's own log
    # reports ~0.91 of the nameplate as available. Judge the fit against that, not the total.
    _u=$(( _mb * 1024 * 1024 * 91 / 100 ))
    [ "$_u" -gt "$_one_card" ] && _one_card=$_u
  done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
fi
_blob=$(blob_bytes_of "${_probe_model:-}")
if [ "$BENCH_MODE" = gpu ] && [ -z "$GPU_PIN" ] && [ "$_one_card" -gt 0 ] && [ "$_blob" -gt "$_one_card" ]; then
  OLLAMA_GPUS=all             # one card cannot hold it: give every engine the machine
  LOKEN_GPUS=all
  VLLM_GPUS=all
  echo "  ⓘ $_probe_model weighs $((_blob/1000000000)) GB > one card ($((_one_card/1000000000)) GB) — every engine unpinned" >&2
fi
gpu_policy_banner
# --stream is the protocol default: it measures a decode rate client-side and identically
# for all three engines. STREAM=0 lifts it so the non-streaming path can be covered too -
# most integrations call it, and it had never been measured.
COMMON=(--warmup 0 --carbon-intensity 50)
[ "${STREAM:-1}" = 1 ] && COMMON+=(--stream)

# CPU thermal fairness: the two engines run back-to-back, so without a gate the
# SECOND engine starts on a hotter package and clocks lower — a measurable
# ordering bias on long CPU runs (the engine measured first runs on a colder machine, which
# recommends gating on the real package temp, not blind sleeps). Wait for the CPU
# package to fall below COOL_C (default 45°C) before each engine's CPU run so both
# measure cool-vs-cool. GPU mode: no-op. Cap the wait so a hot box still finishes.
# The package sensor is found by TYPE, because the zone NUMBER differs per machine and a
# hardcoded one silently reads someone else's thermometer — or nothing at all.
_discover_pkg_temp() {
  local z
  for z in /sys/class/thermal/thermal_zone*; do
    case "$(cat "$z/type" 2>/dev/null)" in
      x86_pkg_temp|cpu-thermal|soc_thermal|Package*) echo "$z/temp"; return 0 ;;
    esac
  done
  return 1
}
PKG_TEMP_SENSOR="${PKG_TEMP_SENSOR:-$(_discover_pkg_temp || true)}"   # milli-°C, may be empty
cool_wait() {
  [ "$BENCH_MODE" = cpu ] || return 0
  [ -r "$PKG_TEMP_SENSOR" ] || return 0
  local thr_mc=$(( ${COOL_C:-45} * 1000 )) t
  for _ in $(seq 1 120); do            # ≤120 s cap
    t=$(cat "$PKG_TEMP_SENSOR" 2>/dev/null || echo 0)
    [ "$t" -le "$thr_mc" ] && { echo "  🌡 pkg $((t/1000))°C ≤ ${COOL_C:-45}°C — proceeding" >&2; return 0; }
    sleep 2
  done
  echo "  🌡 pkg still $((t/1000))°C after cap — proceeding anyway" >&2
}

kill_all() {
  # Kill engines AND their GPU-holding children: ollama spawns `llama-server`;
  # vLLM spawns `VLLM::EngineCore` + multiprocessing `resource_tracker` python
  # workers (note the UPPERCASE name → match case-insensitively). Missing these
  # leaks ~13GB/GPU of orphaned VRAM that silently wrecks the next engine's run.
  pkill -9 -x ollama 2>/dev/null || true
  # -x (exact comm), NOT -f: free-text -f patterns match any OTHER shell whose
  # command line merely MENTIONS these words (e.g. a wrapper doing its own
  # cleanup) and silently kill it mid-run — bit us repeatedly on 2026-07-02.
  pkill -9 -x llama-server 2>/dev/null || true
  pkill -9 -x server 2>/dev/null || true
  pkill -9 -f 'bin/vllm serve' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  sleep 3
  # Wait for VRAM to drain to NEAR-ZERO (killed CUDA procs release asynchronously).
  # Drain to driver overhead only, not to "nearly free". A memory estimator that finds the
  # card not quite pristine can decide a near-limit model does not fit and place it on the
  # CPU — which then gets recorded as that engine's GPU number, several times too slow. The
  # same engine started on a clean card places the same model on the GPU. So drain hard and
  # let it settle before starting the next engine.
  for _ in $(seq 1 40); do
    local mx; mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
    [ -z "$mx" ] && mx=0
    [ "$mx" -lt 600 ] && break
    sleep 2
  done
  sleep 3   # extra settle: VRAM "used" can read low before the allocator fully releases
}
# wait_url <url> [seconds] [must-contain] — an engine answering is not an engine ready:
# vLLM serves /v1/models while the weights are still loading, so the third argument lets a
# probe demand a substring rather than a status code.
wait_url() {
  local url="$1" secs="${2:-60}" want="${3:-}"
  for _ in $(seq 1 "$secs"); do
    if [ -n "$want" ]; then
      curl -s -m2 "$url" 2>/dev/null | grep -q "$want" && return 0
    else
      curl -s -m2 "$url" >/dev/null 2>&1 && return 0
    fi
    sleep 1
  done
  return 1
}

# ── THE ENGINES ──────────────────────────────────────────────────────────────────────────
# One declaration each. Everything below starts, probes and stops them identically; what
# differs between them is DATA in this table, not three shapes of code further down.
#
#   BIN    what to execute                ARGS   its own arguments
#   ENV    what it needs beyond the GPU policy   CWD  where to run it from
#   PROBE  the URL that answers when it is up    WANT a substring the probe must return
#   WAIT   seconds to allow                      LOG  where its output goes
declare -A E_BIN E_ARGS E_ENV E_CWD E_PROBE E_WANT E_WAIT

engine_start() {
  local n="$1"
  # Two statements, not one: bash expands every word of a `local` before assigning any of
  # them, so "$n" in a second assignment on the same line reads the OUTER n — unset here.
  local log="$LOGDIR/$n.log"
  kill_all
  mkdir -p "$LOGDIR"
  ( cd "${E_CWD[$n]:-$PWD}" \
    && env $(gpu_env_for "$n") ${E_ENV[$n]:-} nohup "${E_BIN[$n]}" ${E_ARGS[$n]:-} >"$log" 2>&1 & )
  wait_url "${E_PROBE[$n]}" "${E_WAIT[$n]:-60}" "${E_WANT[$n]:-}" \
    || { echo "  ⚠ $n did not come up within ${E_WAIT[$n]:-60}s — see $log" >&2; return 1; }
  return 0
}

TMP=$(mktemp -d); PARTS=()
TMPDIR_LOGS="$TMP/logs"; LOGDIR="${LOGDIR:-$TMPDIR_LOGS}"; mkdir -p "$LOGDIR"

# The three engines, declared once.
LOKEN_ABS="$(cd "$(dirname "$LOKEN_BIN")" && pwd)/$(basename "$LOKEN_BIN")"
E_BIN[ollama]="$OLLAMA_BIN"; E_ARGS[ollama]="serve"
E_ENV[ollama]="OLLAMA_VULKAN=0 OLLAMA_MODELS=$MODELS_DIR/"
E_PROBE[ollama]="http://127.0.0.1:$OLLAMA_PORT/api/version"; E_WAIT[ollama]=60

E_BIN[loken]="$LOKEN_ABS"; E_ARGS[loken]="serve --models-dir $MODELS_DIR --keep-alive 30m ${LLMSERVE[*]:-}"
# Started from the directory that HOLDS config.toml. There is no --config flag: the server
# looks for ./config.toml first, so the working directory silently decides which configuration
# is measured — and a different memory fraction is a different per-card budget, hence a
# different placement for any model near the boundary.
E_CWD[loken]="$(cd "$(dirname "$0")/.." && pwd)"
E_PROBE[loken]="http://127.0.0.1:$LOKEN_PORT/api/version"; E_WAIT[loken]=60

# vLLM is launched through a wrapper because its own flags differ per source (an HF repo or a
# converted ollama tag). The wrapper is data here like the others, not a special case below.
E_BIN[vllm]="$(dirname "$(readlink -f "$0")")/vllm-serve-hf.sh"
E_PROBE[vllm]="http://127.0.0.1:$VLLM_PORT/v1/models"; E_WANT[vllm]='"id"'; E_WAIT[vllm]=600
echo "▶ BENCH_MODE=$BENCH_MODE  GPU_PIN='${GPU_PIN:-none}'  → $OUT"

# ── 1. ollama, isolated ──────────────────────────────────────────────────────
# RETRY on CPU-fallback: a VRAM-tight model can spill to the CPU at load non-deterministically
# — a transient memory reading at placement time, not a real fit limit. The same model on the
# same card usually places entirely on the GPU and occasionally does not. Recording the
# fallback as that engine's GPU result inflates the comparison severalfold. So: detect a large
# "CPU model buffer" (>2000 MiB = real fallback, vs the ~270 MiB embed buffer that's
# normal even on full-GPU) in the ollama log and RETRY (up to 3×) on a fresh restart.
# Only applies in GPU mode (CPU mode legitimately runs everything on CPU).
# _probe_model was parsed with the placement decision above.
cpubuf_of_log() { grep -oE "CPU(_Mapped)? model buffer size = +[0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+$" | sort -rn | head -1; }
# GUARD: GPU_PIN restricts CUDA_VISIBLE_DEVICES to ONE card. A model whose weights exceed
# that card's usable VRAM then spills to the host and decodes at CPU speed — which reads as a
# catastrophic loss and is really a pin the model never fitted behind. The tell in the results
# is a GPU decode rate that matches the CPU one. Warn only: GPU_PIN is
# an explicit operator choice, and unpinning it silently would measure a machine nobody
# asked for. Without GPU_PIN the same comparison is made above and acted on.
if [ "$BENCH_MODE" = gpu ] && [ -n "$GPU_PIN" ]; then
  _mcsv=""; _sm=0; for _a in "$@"; do [ "$_sm" = 1 ] && { _mcsv="$_a"; break; }; [ "$_a" = "--models" ] && _sm=1; done
  IFS=',' read -ra _MS <<< "$_mcsv"
  for _m in "${_MS[@]}"; do
    _sz=$(blob_bytes_of "$_m")
    if [ "$_one_card" -gt 0 ] && [ "$_sz" -gt "$_one_card" ]; then
      echo "⚠️⚠️  GPU_PIN=$GPU_PIN + '$_m' weighs $((_sz/1000000000)) GB > one card ($((_one_card/1000000000)) GB) → it will SPILL to CPU (fake loss, GPU tok/s ≈ CPU tok/s)." >&2
      echo "⚠️⚠️  Re-run without GPU_PIN so every engine gets the same cards." >&2
    fi
  done
fi
for _att in 1 2 3; do
  kill_all
  echo "▶ ollama (default params)${_att:+ [try $_att]} …"
  engine_start ollama || true
  # PRE-BENCH placement probe (GPU mode, NON-spread only): force a one-token load and check
  # whether ollama CPU-fell BEFORE wasting a full multi-context bench; retry on a fresh
  # restart if it did. SKIPPED when SCHED_SPREAD is on: spread already gives correct
  # multi-GPU placement on a fresh load, and a probe load at a DIFFERENT context size than
  # the bench forces a reload that itself CPU-falls (the probe would sabotage the fix). The
  # probe must load at the SAME num_ctx as the bench (4096) so it does not trigger a reload.
  if [ "$BENCH_MODE" = gpu ] && [ -n "$_probe_model" ] && [ "$(_gpu_count "$OLLAMA_GPUS")" = 1 ]; then
    curl -s -m 180 http://127.0.0.1:$OLLAMA_PORT/api/generate -H "Content-Type: application/json" \
      -d "{\"model\":\"$_probe_model\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":4096}}" >/dev/null 2>&1 || true
    # `|| true`: cpubuf_of_log is a grep pipeline, and under `set -o pipefail`
    # a log with no CPU-buffer line makes it exit non-zero, which `set -e` turns
    # into a silent death of the whole cell - the empty-string guard below never
    # gets to run. That is how qwen3.5:35b produced no row while both engines
    # served it perfectly well by hand.
    cpubuf=$(cpubuf_of_log "$LOGDIR/ollama.log" || true); [ -z "$cpubuf" ] && cpubuf=0
    if [ "$cpubuf" -gt 2000 ] && [ "$_att" -lt 3 ]; then
      echo "  ⚠ ollama CPU-fell at load (CPU model buffer ${cpubuf} MiB) — retrying on fresh restart" >&2
      continue
    fi
    [ "$cpubuf" -gt 2000 ] && echo "  ⚠ ollama STILL CPU-fell after 3 tries (CPU buffer ${cpubuf} MiB) — model is VRAM-marginal on this box; recording ollama's CPU-fallback as its real default behaviour" >&2
  fi
  cool_wait
  "$ASSAY" --ollama http://127.0.0.1:$OLLAMA_PORT "${NUMGPU[@]}" "${COMMON[@]}" "$@" -o "$TMP/ollama.json" || true
  break
done
if [ -f "$TMP/ollama.json" ]; then PARTS+=("$TMP/ollama.json"); fi

# ── 2. LOKEN, isolated ───────────────────────────────────────────────────────
kill_all
echo "▶ LOKEN${LLMSERVE:+ ${LLMSERVE[*]}} …"
engine_start loken || true
cool_wait
"$ASSAY" --loken http://127.0.0.1:$LOKEN_PORT "${NUMGPU[@]}" "${COMMON[@]}" "$@" -o "$TMP/loken.json" || true
if [ -f "$TMP/loken.json" ]; then PARTS+=("$TMP/loken.json"); fi

# ── 3. vLLM, isolated (GPU only, when requested) ─────────────────────────────
if [ "$BENCH_MODE" = gpu ] && [ -n "${VLLM_SERVE:-}" ]; then
  kill_all
  echo "▶ vLLM: $VLLM_SERVE …"
  # Which wrapper depends on where the weights come from; its own index flag stays because
  # vLLM parses CUDA_VISIBLE_DEVICES as integers and a UUID breaks it. The card SET is the
  # policy's; this only says which of those it starts on.
  if [[ "$VLLM_SERVE" == hf:* ]]; then
    E_BIN[vllm]="$(dirname "$(readlink -f "$0")")/vllm-serve-hf.sh"
    E_ARGS[vllm]="${VLLM_SERVE#hf:} --gpu ${GPU_PIN:-0} --port $VLLM_PORT"
  else
    E_BIN[vllm]="$(dirname "$(readlink -f "$0")")/vllm-serve-ollama.sh"
    E_ARGS[vllm]="$VLLM_SERVE --gpu ${GPU_PIN:-0} --port $VLLM_PORT"
  fi
  echo "  waiting for vLLM to finish loading weights (minutes)…"
  engine_start vllm || true
  "$ASSAY" --vllm "$VLLM_PORT" "${COMMON[@]}" "$@" -o "$TMP/vllm.json" || true
  if [ -f "$TMP/vllm.json" ]; then PARTS+=("$TMP/vllm.json"); fi
fi

kill_all
# ── merge the per-engine results into one file ───────────────────────────────
mkdir -p "$(dirname "$OUT")"
jq -s '{timestamp: .[0].timestamp, config: .[0].config, results: (map(.results) | add)}' "${PARTS[@]}" > "$OUT"
echo "▶ merged ${#PARTS[@]} engine(s) → $OUT"
