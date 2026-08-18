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
OLLAMA_BIN=/usr/local/bin/ollama
LOKEN_BIN="${LOKEN_BIN:-../loken/target/release/server}"   # one CUDA binary, both modes
ASSAY="${ASSAY:-target/release/assay}"
VLLM_PORT=8000
BENCH_MODE="${BENCH_MODE:-cpu}"
GPU_PIN="${GPU_PIN:-}"                       # e.g. 0 → expose ONLY that card's UUID
MAIN_GPU="${MAIN_GPU:-0}"   # ollama-only index; empty = do not restrict it to one card
OUT="${OUT:-results/fairbench.json}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"          # each engine's port, so two runs can coexist
LOKEN_PORT="${LOKEN_PORT:-11435}"
[ "$BENCH_MODE" = cpu ] || [ "$BENCH_MODE" = gpu ] || { echo "BENCH_MODE must be cpu|gpu" >&2; exit 1; }

# GPU order: CUDA_VISIBLE_DEVICES (the var ollama, LOKEN AND vLLM honour — NOT
# CUDA_DEVICE_ORDER) lists the device UUIDs in nvidia-smi order, so index 0 means the
# same physical card in every engine's enumeration, all cards visible. BUT ollama's
# scheduler still self-picks the slower card for a single-GPU-fit model even
# with the right CVD (proven) — so ollama ALSO gets `--main-gpu 0` to force it onto
# the fast card. LOKEN and vLLM (tp=1) use device 0. Result: all engines on the SAME
# card → the comparison measures the engine, not the GPU. That reasoning holds only
# while the weights FIT one card; above it the same flag becomes a handicap, which is
# what the size-derived decision below undoes.
ALL_UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | paste -sd, || echo "")
FAST_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)
if [ -n "$GPU_PIN" ]; then
  CVD=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | sed -n "$((GPU_PIN+1))p")
  FAST_UUID="$CVD"
else
  CVD="$ALL_UUIDS"
fi
PINENV=(env "CUDA_VISIBLE_DEVICES=$CVD")
export CUDA_VISIBLE_DEVICES="$CVD"   # inherited by the vllm-serve-*.sh helpers
NUMGPU=(); LLMSERVE=(); OLLAMA_MG=(); OLLAMA_SPREAD=()
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

# HOW MANY CARDS OLLAMA MAY USE — derived from the weights, never from a list of names.
# `--main-gpu N` does not merely PREFER a card, it RESTRICTS ollama to one: its own log
# says `gpu_count=1 available_gpu_count=2`. That is the right handicap while the weights
# fit that card — every engine then measured on the same silicon — and the wrong one above
# it, where ollama gets one card plus host spill while the others get the machine.
# Measured 2026-08-14 on a 42.5 GB model: ollama offloaded 26 layers to a single card and
# mapped 26.2 GB to the host while the second card sat at 18 MiB. Every cell for a model
# larger than one card had been scoring our placement against ollama's missing card.
# So pin to the fast card only when it fits there; otherwise let ollama spread over all.
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
  MAIN_GPU=""                 # one card cannot hold it: stop restricting ollama to one
  : "${FAIR_SPREAD:=1}"       # and let its scheduler reach every visible card
  echo "  ⓘ $_probe_model weighs $((_blob/1000000000)) GB > one card ($((_one_card/1000000000)) GB) — ollama unpinned, SCHED_SPREAD on" >&2
fi
if [ "$BENCH_MODE" = gpu ] && [ -n "$MAIN_GPU" ]; then OLLAMA_MG=(--main-gpu "$MAIN_GPU"); fi
# FAIR_SPREAD=1 → ollama uses ALL exposed GPUs (OLLAMA_SCHED_SPREAD). Derived above for
# anything larger than one card; still settable by hand to override the derivation.
if [ "$BENCH_MODE" = gpu ] && [ -z "$GPU_PIN" ] && [ "${FAIR_SPREAD:-0}" = 1 ]; then OLLAMA_SPREAD=(OLLAMA_SCHED_SPREAD=1); fi
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
wait_url() { for i in $(seq 1 "${2:-60}"); do curl -s -m2 "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

TMP=$(mktemp -d); PARTS=()
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
  echo "▶ ollama (default params${GPU_PIN:+, pinned GPU$GPU_PIN})${_att:+ [try $_att]} …"
  "${PINENV[@]}" env OLLAMA_VULKAN=0 "${OLLAMA_SPREAD[@]}" OLLAMA_MODELS="$MODELS_DIR/" nohup "$OLLAMA_BIN" serve >/tmp/ollama_fairbench.log 2>&1 &
  wait_url http://127.0.0.1:$OLLAMA_PORT/api/version 60 || { echo "ollama did not start" >&2; }
  # PRE-BENCH placement probe (GPU mode, NON-spread only): force a one-token load and check
  # whether ollama CPU-fell BEFORE wasting a full multi-context bench; retry on a fresh
  # restart if it did. SKIPPED when SCHED_SPREAD is on: spread already gives correct
  # multi-GPU placement on a fresh load, and a probe load at a DIFFERENT context size than
  # the bench forces a reload that itself CPU-falls (the probe would sabotage the fix). The
  # probe must load at the SAME num_ctx as the bench (4096) so it does not trigger a reload.
  if [ "$BENCH_MODE" = gpu ] && [ -n "$_probe_model" ] && [ ${#OLLAMA_SPREAD[@]} -eq 0 ]; then
    curl -s -m 180 http://127.0.0.1:$OLLAMA_PORT/api/generate -H "Content-Type: application/json" \
      -d "{\"model\":\"$_probe_model\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":4096}}" >/dev/null 2>&1 || true
    # `|| true`: cpubuf_of_log is a grep pipeline, and under `set -o pipefail`
    # a log with no CPU-buffer line makes it exit non-zero, which `set -e` turns
    # into a silent death of the whole cell - the empty-string guard below never
    # gets to run. That is how qwen3.5:35b produced no row while both engines
    # served it perfectly well by hand.
    cpubuf=$(cpubuf_of_log /tmp/ollama_fairbench.log || true); [ -z "$cpubuf" ] && cpubuf=0
    if [ "$cpubuf" -gt 2000 ] && [ "$_att" -lt 3 ]; then
      echo "  ⚠ ollama CPU-fell at load (CPU model buffer ${cpubuf} MiB) — retrying on fresh restart" >&2
      continue
    fi
    [ "$cpubuf" -gt 2000 ] && echo "  ⚠ ollama STILL CPU-fell after 3 tries (CPU buffer ${cpubuf} MiB) — model is VRAM-marginal on this box; recording ollama's CPU-fallback as its real default behaviour" >&2
  fi
  cool_wait
  "$ASSAY" --ollama http://127.0.0.1:$OLLAMA_PORT "${NUMGPU[@]}" "${OLLAMA_MG[@]}" "${COMMON[@]}" "$@" -o "$TMP/ollama.json" || true
  break
done
if [ -f "$TMP/ollama.json" ]; then PARTS+=("$TMP/ollama.json"); fi

# ── 2. LOKEN, isolated ───────────────────────────────────────────────────────
kill_all
echo "▶ LOKEN${GPU_PIN:+ (pinned GPU$GPU_PIN)}${LLMSERVE:+ ${LLMSERVE[*]}} …"
# Start the server from the directory that HOLDS config.toml. There is no --config
# flag: the server looks for `./config.toml` first (config.rs), so the working
# directory silently decides which configuration is measured. Launched from the
# repository root the file is not found and the built-in default applies - a
# different max_gpu_memory_fraction, hence a different per-card budget, hence a
# different PLACEMENT for any model sitting near the boundary. deepseek-r1:70b-q3ks
# is 30.9 GB against two 16.4 GB cards: at the configured fraction it loads
# tensor-parallel, at the default it spills ten layers to the host and decodes at a
# fifth of the rate. The bench must measure the server the config describes.
LOKEN_ABS="$(cd "$(dirname "$LOKEN_BIN")" && pwd)/$(basename "$LOKEN_BIN")"
CONFIG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
( cd "$CONFIG_DIR" && "${PINENV[@]}" nohup "$LOKEN_ABS" serve --models-dir "$MODELS_DIR" --keep-alive 30m "${LLMSERVE[@]}" >/tmp/loken_fairbench.log 2>&1 & )
wait_url http://127.0.0.1:$LOKEN_PORT/api/version 60 || { echo "LOKEN did not start" >&2; }
cool_wait
"$ASSAY" --loken http://127.0.0.1:$LOKEN_PORT "${NUMGPU[@]}" "${COMMON[@]}" "$@" -o "$TMP/loken.json" || true
if [ -f "$TMP/loken.json" ]; then PARTS+=("$TMP/loken.json"); fi

# ── 3. vLLM, isolated (GPU only, when requested) ─────────────────────────────
if [ "$BENCH_MODE" = gpu ] && [ -n "${VLLM_SERVE:-}" ]; then
  kill_all
  echo "▶ vLLM: $VLLM_SERVE${GPU_PIN:+ (GPU$GPU_PIN)} …"
  # Pin vLLM to the chosen card. vLLM only accepts INTEGER device indices in
  # CUDA_VISIBLE_DEVICES (a UUID breaks its int() parse), and the helper forces
  # CUDA_DEVICE_ORDER=PCI_BUS_ID, where the index is a stable bus position.
  VG=(--gpu "${GPU_PIN:-0}")
  if [[ "$VLLM_SERVE" == hf:* ]]; then
    nohup scripts/vllm-serve-hf.sh ${VLLM_SERVE#hf:} "${VG[@]}" --port "$VLLM_PORT" >/tmp/vllm_fairbench.log 2>&1 &
  else
    nohup scripts/vllm-serve-ollama.sh $VLLM_SERVE "${VG[@]}" --port "$VLLM_PORT" >/tmp/vllm_fairbench.log 2>&1 &
  fi
  echo "  waiting for vLLM /v1/models (weight load, minutes)…"
  for i in $(seq 1 300); do
    curl -s -m2 "http://127.0.0.1:$VLLM_PORT/v1/models" 2>/dev/null | grep -q '"id"' && { echo "  vLLM ready"; break; }
    sleep 2
  done
  "$ASSAY" --vllm "$VLLM_PORT" "${COMMON[@]}" "$@" -o "$TMP/vllm.json" || true
  if [ -f "$TMP/vllm.json" ]; then PARTS+=("$TMP/vllm.json"); fi
fi

kill_all
# ── merge the per-engine results into one file ───────────────────────────────
mkdir -p "$(dirname "$OUT")"
jq -s '{timestamp: .[0].timestamp, config: .[0].config, results: (map(.results) | add)}' "${PARTS[@]}" > "$OUT"
echo "▶ merged ${#PARTS[@]} engine(s) → $OUT"
scripts/bench_warm.sh "$OUT" 2>/dev/null | sort || true
