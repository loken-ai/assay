# Changelog

All notable changes to `assay`. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

### Added

- Sweeps of models x context lengths x prompt sizes against Ollama, LOKEN and vLLM, driven
  through each engine's own API.
- Prefill and decode reported apart. A single tokens-per-second figure conflates them, and
  they do not move together: an engine can win the decode and lose the request.
- Energy alongside speed. GPU and host samplers run with the request, so a result carries
  joules per token. Two engines at the same rate are not equivalent if one draws half again
  as much.
- Levers that make the engines comparable before comparing them: `--num-gpu 0` for a
  CPU-to-CPU run, `--main-gpu` to pin Ollama to a chosen card on a box whose cards are not
  equivalent, `--session-id` so a repeated prefix is measured as a steady state rather than
  as a cold first token.
- Vision prompt suites beside the text ones, because an image prompt takes a different path
  and a text sweep says nothing about it.
