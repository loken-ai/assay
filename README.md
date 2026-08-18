# assay

Benchmark a local inference server against another one, on the same prompts and the same
clock. It drives [Ollama](https://ollama.com), [LOKEN](https://github.com/loken-ai/loken)
and vLLM through their own APIs, sweeps models, context lengths and prompt sizes, and reports
what each of them actually did rather than what it claims.

```sh
assay --ollama http://localhost:11434 --loken http://localhost:11435 \
         --models qwen3:latest --num-ctx 4096 --prompts short,medium,long
```

## What it measures, and why it is fussy about it

A benchmark that is easy to run is easy to run wrong, so this one is opinionated where the
difference between engines hides:

- **prefill and decode are separated.** A single tokens-per-second figure conflates the two,
  and they do not move together - one engine can win the decode and lose the request.
- **energy, not only speed.** GPU and host samplers run alongside the request, so a result
  carries joules per token beside tokens per second. Two engines at the same rate are not
  equivalent if one of them draws half again as much.
- **the engines are made comparable before they are compared.** `--num-gpu 0` forces Ollama
  onto the CPU for a CPU-to-CPU run; `--main-gpu` pins it to a chosen card on a box with
  asymmetric GPUs, where its scheduler may otherwise pick the slower one. Without those, the
  two sides are not running the same experiment.
- **vision suites** exist beside the text ones, because an image prompt exercises a different
  path and a text-only sweep says nothing about it.

## Reproducing a fair run

The tool measures; making the measurement mean something is the harness's job, and both ship
here so a comparison can be replayed elsewhere:

```sh
BENCH_MODE=gpu GPUS=0,1 OUT=results/three-way.json \
  scripts/fair-run.sh --models qwen3:latest --prompts short,medium,long
```

- **`scripts/gpu-policy.sh`** decides which cards each engine may use, and is the only place
  that decision is written. It resolves an index list to device **UUIDs** — `CUDA_VISIBLE_DEVICES=0`
  is a different physical card depending on the ordering — and translates one policy into each
  engine's own lever: `OLLAMA_SCHED_SPREAD` when ollama can see more than one card, because
  exposing cards is not the same as using them, and `--tensor-parallel-size` for vLLM. Per-engine
  overrides (`OLLAMA_GPUS`, `LOKEN_GPUS`, `VLLM_GPUS`) exist for the deliberately asymmetric run;
  the banner says when they differ, because then the cells are not like-for-like.
- **`scripts/fair-run.sh`** runs one engine at a time with the others' processes stopped, restarts
  each one cold, discards warm-up, forces greedy decoding, and waits on a thermal gate so the
  engine measured second is not measured on a hotter machine. Whether ollama is pinned to one
  card is derived from the weights against one card's usable capacity — a pin that is right for
  a model that fits is wrong for one that does not, so it cannot be a constant.

Neither is optional if you intend to publish the numbers. Most of what these scripts do exists
because a run without it produced a figure that was wrong in a way nobody noticed.

## Build

```sh
cargo build --release      # target/release/assay
```

No GPU or model is needed to build it. The GPU sampler reads NVML at run time when a card is
present and reports nothing when it is not.

## Licence

MIT OR Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.
