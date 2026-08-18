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

## Build

```sh
cargo build --release      # target/release/assay
```

No GPU or model is needed to build it. The GPU sampler reads NVML at run time when a card is
present and reports nothing when it is not.

## Licence

MIT OR Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.
