# Changelog

All notable changes to `assay`. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-18

### Added

- Sweeps of models x context lengths x prompt sizes against Ollama, LOKEN and vLLM, driven
  through each engine's own API.
- Prefill and decode reported apart. A single tokens-per-second figure conflates them, and
  they do not move together: an engine can win the decode and lose the request.
- Energy alongside speed. GPU and host samplers run with the request, so a result carries
  joules per token. Two engines at the same rate are not equivalent if one draws half again
  as much.
- One definition per metric, applied to every engine. Rates, time to first token and
  end-to-end latency come from the client's clock rather than from whichever server happens
  to report a field, so no engine is measured through its own bookkeeping.
- Energy labelled with the domains it covers. CPU and DRAM come from Linux powercap, so a run
  elsewhere counts the GPU alone and its joules-per-token is a different quantity — the label
  says which, and the comparison table will not subtract one from the other.
- `scripts/gpu-policy.sh` and `scripts/fair-run.sh`: which cards each engine may use, written
  once and applied to all three at launch, and the protocol around it — one engine at a time
  with the others stopped, cold restarts, a thermal gate, and the ollama pin derived from the
  weights against one card's capacity rather than fixed.
- A coherence gate that runs on every cell without an expected answer, so a completion that
  is fast and degenerate is not reported as a rate.
- Vision prompt suites beside the text ones, because an image prompt takes a different path
  and a text sweep says nothing about it.
- `--concurrency` to measure what the engine serves, not only what one request gets.
- `--num-gpu 0` for a CPU-to-CPU run, `--session-id` for prefix reuse where an engine
  implements it, `--unique-prompt` to defeat prefix caching, `--host-proc-names` for the
  memory-footprint columns, `--idle-energy-secs` to record the idle floor.
