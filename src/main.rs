//! LLM Benchmark — compare Ollama vs LOKEN performance
//!
//! Sweeps over models × num_ctx × prompts and reports per-token streaming
//! metrics + per-GPU sampling for each combination.
//!
//! Usage:
//!   # Single model, single context, default prompts
//!   llmbench --ollama 11434 --loken 11435 --models qwen3:latest --stream
//!
//!   # Full sweep with JSON + CSV output
//!   llmbench --ollama 11434 --loken 11435 \
//!       --models qwen3:latest,devstral-small-2:latest \
//!       --num-ctx 2048,4096 --stream -n 3 \
//!       -o sweep.json --output-csv sweep.csv

mod api;
mod stats;

use api::{BenchClient, IterationMetrics, Protocol};
use clap::Parser;
use energy::{EnergySampler, EnergyWindow};
use gpu_sampler::{GpuSample, GpuSampler};
use host_sampler::{HostSample, HostSampler};
use stats::Stats;
use std::collections::BTreeMap;
use std::time::Duration;

// Built-in prompt suite -------------------------------------------------------

// Continuation-style prompt: works for both chat-tuned and base models
// when sent through /api/generate (raw, no chat template). Avoids the
// "model immediately emits <end_of_turn>" failure mode that some chat
// models (gemma4, qwen3) hit with question-style prompts via raw API.
const PROMPT_SHORT: &str = "Once upon a time, in a kingdom far far away, there lived a";

const PROMPT_MEDIUM: &str = "The history of computing began in the early 19th century with the \
invention of the mechanical calculator. Charles Babbage proposed the Analytical Engine, considered \
the first general-purpose computer design. The next major milestone was";

// Vision prompts (used with --image). Short captioning at three lengths so
// per-cell prefill/decode comparisons mirror the text suite shape. The
// "Once upon a time" continuation style doesn't apply — vision models
// always condition on the image, so direct instructions work better.
const PROMPT_VISION_SHORT: &str = "Describe this image.";
const PROMPT_VISION_MEDIUM: &str = "Describe this image in detail. Mention the subjects, what they \
appear to be doing, the setting, and any notable objects or background elements. Aim for about three sentences.";
const PROMPT_VISION_LONG: &str = "Provide a thorough, paragraph-length description of this image. \
Cover (1) the people or main subjects and what they are doing, (2) the environment and setting, \
(3) clothing, colors, and visible objects, (4) the mood or atmosphere, and (5) any interactions \
between subjects or with the environment. Be specific and concrete rather than abstract — describe \
what you actually see rather than what it might mean. Use complete sentences and write at least \
six sentences.";

const PROMPT_LONG: &str = "You are a senior code reviewer. Review the following Rust function \
for correctness, performance, idiomatic style, and potential edge cases. Suggest \
concrete improvements with examples. Be thorough but concise.\n\n\
```rust\n\
pub fn parse_csv_line(line: &str) -> Vec<String> {\n\
    let mut result = Vec::new();\n\
    let mut current = String::new();\n\
    let mut in_quotes = false;\n\
    let mut chars = line.chars().peekable();\n\
    while let Some(c) = chars.next() {\n\
        if c == '\"' {\n\
            if in_quotes && chars.peek() == Some(&'\"') {\n\
                current.push('\"');\n\
                chars.next();\n\
            } else {\n\
                in_quotes = !in_quotes;\n\
            }\n\
        } else if c == ',' && !in_quotes {\n\
            result.push(std::mem::take(&mut current));\n\
        } else {\n\
            current.push(c);\n\
        }\n\
    }\n\
    result.push(current);\n\
    result\n\
}\n\
```\n\n\
Specifically address: 1) handling of trailing commas, 2) RFC 4180 compliance for \
embedded newlines, 3) memory allocation patterns, 4) UTF-8 correctness, 5) error \
handling for malformed input. For each issue, show a corrected snippet.";

// The `2.5K` perimeter prompt: a single ~4 K-token request that fills the KV
// cache (the completion length matches the other cells, so the column isolates
// decode-rate falloff as KV grows). This is the ACTUAL prompt the campaigns used
// (extracted verbatim from rebench030_25k.json's custom_prompt) — a Q4_K /
// Blackwell-GPU kernel review — kept in-tree so the bench is reproducible.
// Previously passed via an unpreserved external --prompt-file.
const PROMPT_2_5K: &str = include_str!("../prompts/p2_5k.txt");

/// Resolve a built-in prompt name to its content; returns None if unknown.
fn builtin_prompt(name: &str) -> Option<&'static str> {
    match name {
        "short" => Some(PROMPT_SHORT),
        "medium" => Some(PROMPT_MEDIUM),
        "long" => Some(PROMPT_LONG),
        "2.5k" | "2.5K" => Some(PROMPT_2_5K),
        "vision_short" => Some(PROMPT_VISION_SHORT),
        "vision_medium" => Some(PROMPT_VISION_MEDIUM),
        "vision_long" => Some(PROMPT_VISION_LONG),
        _ => None,
    }
}

// CLI -------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "llmbench")]
#[command(about = "Inference server benchmark — compare Ollama vs LOKEN across models, contexts, and prompts")]
#[command(version)]
struct Args {
    /// Ollama server URL (e.g., http://host:11434)
    #[arg(long)]
    ollama: Option<String>,

    /// LOKEN URL (e.g., http://host:11435)
    #[arg(long)]
    loken: Option<String>,

    /// vLLM URL or port (OpenAI-compatible /v1 API, e.g. 8000 or
    /// http://host:8000). The model is fixed at `vllm serve` launch — the
    /// bench can't pull/swap it, and `num_ctx` is set via the server's
    /// `--max-model-len` (per-request num_ctx is ignored). Use `--stream` for
    /// a fair decode rate (the OpenAI API reports no prefill/decode split).
    #[arg(long)]
    vllm: Option<String>,

    /// Comma-separated list of models to benchmark (e.g. qwen3:latest,devstral-small-2:latest)
    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,

    /// Single model alias for --models (kept for backward compat)
    #[arg(short = 'm', long, hide = true)]
    model: Option<String>,

    /// Comma-separated list of context lengths to sweep (default: 4096)
    #[arg(long, value_delimiter = ',', default_values_t = vec![4096usize])]
    num_ctx: Vec<usize>,

    /// Force Ollama's `num_gpu` (layers offloaded to GPU). Pass `--num-gpu 0`
    /// for a fair CPU-vs-CPU benchmark against the CPU-only LOKEN build —
    /// otherwise Ollama auto-offloads to GPU. LOKEN ignores it.
    #[arg(long)]
    num_gpu: Option<usize>,

    /// Force Ollama's `main_gpu` (which GPU index ollama uses). On a box with
    /// asymmetric GPUs ollama's scheduler may pick the slower card for a model
    /// that fits either; `--main-gpu 0` pins it to GPU 0 (the fast card) while
    /// both stay visible, so ollama and LOKEN compare on the same hardware.
    #[arg(long)]
    main_gpu: Option<usize>,

    /// Comma-separated list of prompt names. Text suite: short, medium, long
    /// (the default). Vision suite: vision_short, vision_medium, vision_long —
    /// require --image to be passed too.
    #[arg(long, value_delimiter = ',', default_values_t = vec!["short".to_string(), "medium".to_string(), "long".to_string()])]
    prompts: Vec<String>,

    /// Custom prompt text (overrides --prompts; runs once per sweep cell)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum completion tokens
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Number of benchmark iterations per (target, model, num_ctx, prompt) cell
    #[arg(short = 'n', long, default_value = "3")]
    iterations: usize,

    /// Number of warmup iterations (not counted in stats)
    #[arg(short, long, default_value = "1")]
    warmup: usize,

    /// Save aggregated results to JSON file
    #[arg(short, long)]
    output: Option<String>,

    /// Compare current run against a previous bench JSON. Per-cell
    /// completion_tok_s deltas printed at the end. Warns if model
    /// fingerprints differ between runs (model was re-pulled — the
    /// "regression" is then not a code regression, see
    /// project_deepcoder_bisect_2026_05_25.md).
    #[arg(long)]
    baseline: Option<String>,

    /// Save per-iteration data to CSV file
    #[arg(long)]
    output_csv: Option<String>,

    /// Use streaming mode (measures real TTFT + per-token ITL)
    #[arg(long)]
    stream: bool,

    /// Accepted and ignored.
    ///
    /// Iterations do not unload between themselves: isolation is handled once at
    /// the perimeter, and unloading per iteration forces a cold cache on every
    /// one of them, which inflates the variance it was meant to control.
    #[arg(long)]
    no_unload: bool,

    /// Disable GPU sampling (use when NVML is unavailable)
    #[arg(long)]
    no_gpu_sample: bool,

    /// GPU sample interval in milliseconds
    #[arg(long, default_value = "100")]
    gpu_sample_interval_ms: u64,

    /// Verbose output — show HTTP requests/responses
    #[arg(short, long)]
    verbose: bool,

    /// Path to an image file to attach to every iteration. When set, the
    /// image is base64-encoded once at startup and shipped in the
    /// `images[]` field of every /api/generate call — required for vision
    /// models (moondream, llava, etc). Without this flag, vision prompts
    /// run without an image and the model responds based on text alone.
    #[arg(long)]
    image: Option<String>,

    /// Coherence-gate substring. After every cell prints, if --image and
    /// --require-substr are both set, fail any iteration whose response
    /// preview doesn't contain (case-insensitive) at least one of the
    /// comma-separated substrings. Catches the Z-Image black-PNG /
    /// gemma4 "three three three" failure modes where tok/s looks
    /// healthy but output is garbage.
    #[arg(long, value_delimiter = ',')]
    require_substr: Vec<String>,

    /// Disable energy/carbon measurement (skip NVML energy + RAPL sampling).
    #[arg(long)]
    no_energy: bool,

    /// Grid carbon intensity (gCO2eq per kWh) for the absolute gCO2/token
    /// figure. Default is a documented world-average placeholder; it does NOT
    /// affect the J/token ranking (CI is a common factor across engines). Set
    /// to your region's value (e.g. France ~50, world ~480, coal grid ~800).
    #[arg(long, default_value_t = energy::DEFAULT_CARBON_INTENSITY)]
    carbon_intensity: f64,

    /// Sample idle (no-request) power for this many seconds at startup to
    /// capture an idle-energy baseline (active vs idle reporting later). 0 = off.
    #[arg(long, default_value = "0")]
    idle_energy_secs: u64,

    /// Optional session id. When set, every /api/generate request gets
    /// `options.session_id = <value>` so LOKEN can reuse cached prefix
    /// KV (vision: image-embed prefix; text: prompt prefix). Ollama
    /// ignores the field. Use to measure warm-state throughput rather
    /// than cold per-iteration.
    #[arg(long)]
    session_id: Option<String>,
}

/// A target server to benchmark
struct ServerTarget {
    label: String,
    url: String,
    protocol: Protocol,
}

/// Aggregated results for one (target, model, num_ctx, prompt) sweep cell.
struct CellResult<'a> {
    target: &'a ServerTarget,
    model: String,
    /// Captured at cell start from /api/show: (modified_at, parameter_size,
    /// quantization_level). Used so cross-session perf comparisons can
    /// verify model identity — Ollama auto-pulls update the underlying
    /// GGUF mid-bench-history, causing phantom regressions if not pinned.
    /// See project_deepcoder_bisect_2026_05_25.md
    model_fingerprint: Option<(String, String, String)>,
    num_ctx: usize,
    prompt_name: String,
    prompt_chars: usize,
    iterations: Vec<IterationMetrics>,
    gpu_samples_per_iter: Vec<Vec<GpuSample>>,
    host_samples_per_iter: Vec<Option<HostSample>>,
    /// Per-iteration energy window (None when measurement is disabled or no
    /// counters were available). Aligned 1:1 with `iterations`.
    energy_per_iter: Vec<Option<EnergyWindow>>,
    /// Carbon intensity (gCO2/kWh) used for the gCO2/token figures.
    carbon_intensity: f64,
    load_time_ms: Option<f64>,
    stats: Vec<Stats>,
    /// First-iteration response preview shown after the cell completes. Used
    /// for the coherence eyeball check — distinguishes a real WIN from a
    /// fast-but-garbage WIN (Z-Image black-PNG, gemma4 "three three three").
    first_response_preview: Option<String>,
    /// Coherence pass/fail flag. None if --require-substr was empty (no gate
    /// requested), Some(true) if every non-empty preview contained at least
    /// one required substring, Some(false) if any iteration lacked them all.
    coherence_pass: Option<bool>,
}

#[tokio::main]
async fn main() {
    let mut args = Args::parse();

    if args.ollama.is_none() && args.loken.is_none() && args.vllm.is_none() {
        eprintln!("Error: specify at least one of --ollama PORT, --loken PORT, or --vllm PORT");
        std::process::exit(1);
    }

    // Backward compat: -m/--model populates --models if --models is empty
    if args.models.is_empty() {
        if let Some(m) = args.model.take() {
            args.models.push(m);
        }
    }
    if args.models.is_empty() {
        eprintln!("Error: specify at least one model via --models <a,b,c>");
        std::process::exit(1);
    }

    // Resolve prompts: --prompt overrides everything
    let prompt_specs: Vec<(String, &str)> = if let Some(custom) = args.prompt.as_deref() {
        vec![("custom".to_string(), custom)]
    } else {
        let mut out = Vec::new();
        for name in &args.prompts {
            match builtin_prompt(name) {
                Some(p) => out.push((name.clone(), p)),
                None => {
                    eprintln!("Error: unknown prompt name '{}' (valid: short, medium, long, vision_short, vision_medium, vision_long)", name);
                    std::process::exit(1);
                }
            }
        }
        out
    };

    // Load + base64-encode the image once at startup (if --image was given).
    // We hold base64 in memory across the entire sweep so per-iteration
    // latency only includes the HTTP send — JPEG decode would otherwise add
    // ~1-3ms of variance that has nothing to do with the model.
    let image_b64: Option<Vec<String>> = match args.image.as_deref() {
        Some(path) => {
            use base64::Engine;
            match std::fs::read(path) {
                Ok(bytes) => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    println!("  Image:      {} ({} bytes raw, {} bytes b64)", path, bytes.len(), encoded.len());
                    // A vision run with no session id pays a cold prefill on every
                    // iteration: the image prefix is never reused across requests, so
                    // what gets measured is the first-token outlier rather than the
                    // steady state. Measured on a small vision model, the gap between
                    // the two engines more than halved once the id was passed - which
                    // means the number without it was reporting the cache, not the
                    // engines. So a stable id is defaulted when the caller gives none.
                    if args.session_id.is_none() {
                        args.session_id = Some("bench-vision-auto".to_string());
                        eprintln!("\n  ℹ️  Vision bench: auto-set --session-id=bench-vision-auto\n  \
                                       \x20  to enable image-prefix KV reuse across iters\n  \
                                       \x20  so the run reports the engines rather than the\n  \
                                       \x20  cache. Individual iterations may still stop\n  \
                                       \x20  early. Pass a different --session-id\n  \
                                       \x20  to override or pass --session-id '' to disable.\n");
                    }
                    Some(vec![encoded])
                }
                Err(e) => {
                    eprintln!("Error: failed to read --image {}: {}", path, e);
                    std::process::exit(1);
                }
            }
        }
        None => None,
    };

    // Build server targets
    let mut targets: Vec<ServerTarget> = Vec::new();
    if let Some(ref url) = args.ollama {
        targets.push(ServerTarget { label: "Ollama".into(), url: normalize_url(url), protocol: Protocol::Ollama });
    }
    if let Some(ref url) = args.loken {
        targets.push(ServerTarget { label: "LOKEN".into(), url: normalize_url(url), protocol: Protocol::Ollama });
    }
    if let Some(ref url) = args.vllm {
        targets.push(ServerTarget { label: "vLLM".into(), url: normalize_url(url), protocol: Protocol::OpenAI });
    }

    // vLLM reports no prefill/decode timing split — without --stream the decode
    // rate is wall-clock-inflated by prefill. Warn so a non-stream vLLM run
    // isn't misread as a loss.
    if !args.stream && targets.iter().any(|t| t.protocol == Protocol::OpenAI) {
        eprintln!("\n  ⚠️  vLLM target without --stream: decode tok/s will include prefill time\n  \
                       \x20  (the OpenAI completions API gives no prefill/decode split).\n  \
                       \x20  Pass --stream for a fair, length-invariant decode comparison.\n");
    }

    let clients: Vec<BenchClient> = targets.iter()
        .map(|t| {
            let mut c = BenchClient::with_protocol(t.url.clone(), args.verbose, t.protocol);
            c.set_num_gpu(args.num_gpu);
            c.set_main_gpu(args.main_gpu);
            c
        })
        .collect();

    // Header
    println!();
    println!("  LLM Benchmark v{}", env!("CARGO_PKG_VERSION"));
    println!("  Models:     {}", args.models.join(", "));
    println!("  num_ctx:    {}", args.num_ctx.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(", "));
    println!("  Prompts:    {}", prompt_specs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));
    println!("  Max tokens: {}", args.max_tokens);
    println!("  Iterations: {} (+ {} warmup)", args.iterations, args.warmup);
    println!("  Streaming:  {}", if args.stream { "yes" } else { "no" });
    println!("  GPU sample: {}", if args.no_gpu_sample { "off" } else { "on" });
    println!("  Energy:     {}", if args.no_energy { "off".to_string() } else { format!("on (CI={} gCO2/kWh)", args.carbon_intensity) });
    for t in &targets {
        println!("  {:<11} {}", format!("{}:", t.label), t.url);
    }
    println!();

    // Optional idle-energy baseline: sample energy with no request in flight so
    // the later report can separate active from idle draw. Done once, up front.
    let idle_energy: Option<EnergyWindow> = if !args.no_energy && args.idle_energy_secs > 0 {
        println!("  Sampling idle energy baseline for {}s (no request)...", args.idle_energy_secs);
        let s = EnergySampler::start(args.gpu_sample_interval_ms);
        tokio::time::sleep(Duration::from_secs(args.idle_energy_secs)).await;
        let w = s.stop(None).await;
        let idle_w = if w.duration_s > 0.0 { w.energy_j / w.duration_s } else { 0.0 };
        println!(
            "  Idle baseline: {:.2} J over {:.1}s = {:.1} W ({}){}",
            w.energy_j, w.duration_s, idle_w,
            if w.domains_counted.is_empty() { "no counters".to_string() } else { w.domains_counted.join("+") },
            w.note.as_deref().map(|n| format!("  [{}]", n)).unwrap_or_default(),
        );
        Some(w)
    } else {
        None
    };

    // Pre-flight: ensure every model exists on every target. Models that fail
    // are recorded in `failed_models` and skipped during the sweep — we don't
    // abort the entire run because of one missing model.
    use std::collections::HashSet;
    let mut failed_models: HashSet<(usize, String)> = HashSet::new();
    for (i, target) in targets.iter().enumerate() {
        for model in &args.models {
            print!("  [{}] Ensuring model '{}'... ", target.label, model);
            match clients[i].ensure_model(model).await {
                Ok(()) => println!("ok"),
                Err(e) => {
                    println!("FAILED: {} (skipping)", e);
                    failed_models.insert((i, model.clone()));
                }
            }
        }
    }

    // Run the sweep: target → model → num_ctx → prompt
    let mut all_cells: Vec<CellResult> = Vec::new();

    for (idx, target) in targets.iter().enumerate() {
        for model in &args.models {
            if failed_models.contains(&(idx, model.clone())) {
                continue;
            }

            // Cross-unload EVERY currently-loaded model on every server. We
            // query /api/ps to get the actually-loaded list (rather than just
            // iterating over args.models) because the previous bench step or
            // an external client may have left a different model resident —
            // and a leaked model competes for VRAM with the current subject.
            for (j, other) in targets.iter().enumerate() {
                let mut ok = 0usize;
                let mut skipped = 0usize;
                let mut loaded = clients[j].loaded_models().await;
                // Also unload anything from the bench arg list, defensively
                // (covers the rare case where /api/ps doesn't yet reflect a
                // model that's still mid-load).
                for m in &args.models {
                    if !loaded.iter().any(|n| n == m) {
                        loaded.push(m.clone());
                    }
                }
                print!("  [{}] Unloading {} loaded model(s)... ", other.label, loaded.len());
                for m in &loaded {
                    match clients[j].unload_model(m).await {
                        Ok(()) => ok += 1,
                        Err(_) => skipped += 1,
                    }
                }
                println!("ok={} skip={}", ok, skipped);
            }
            // Wait for VRAM to actually drop before loading the next
            // model. Ollama's keep_alive=0 returns 200 OK immediately but
            // the actual VRAM release is asynchronous (Ollama's runner
            // process holds it for ~10-30s). Polling nvidia-smi is more
            // reliable than a fixed sleep — without this, the next
            // model loads into a contended VRAM and falls back to CPU.
            // Skipped for vLLM: its model is resident for the server's
            // lifetime, so there is nothing to wait on (and it would never
            // free, forcing the full 30s timeout every cell).
            if target.protocol == Protocol::Ollama {
                for poll in 0..30u32 {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    let used_mb = std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                        .output().ok()
                        .and_then(|o| String::from_utf8(o.stdout).ok())
                        .and_then(|s| s.lines().next()
                            .and_then(|l| l.trim().parse::<u32>().ok()));
                    if let Some(mb) = used_mb {
                        // < 2 GB: GPU0 is essentially idle (~hundreds of MB
                        // baseline used by drivers + the bench's own server).
                        if mb < 2048 {
                            println!("  [vram-wait] GPU0 freed at {}MB after {}s", mb, poll + 1);
                            break;
                        }
                        if poll == 29 {
                            println!("  [vram-wait] GPU0 still at {}MB after 30s — proceeding anyway", mb);
                        }
                    } else {
                        break; // nvidia-smi not available — fall through
                    }
                }
            }

            // Load model once for this (target, model) — we keep it loaded
            // across all (num_ctx, prompt) cells since num_ctx is per-request.
            // vLLM has no load step (model resident at server launch); we only
            // confirm the server is reachable and serving this model.
            let load_time = if target.protocol == Protocol::OpenAI {
                print!("  [{}] Checking vLLM server (model resident at launch)... ", target.label);
                match clients[idx].load_model(model).await {
                    Ok(_) => { println!("ready"); None }
                    Err(e) => { println!("FAILED: {} (skipping)", e); continue; }
                }
            } else {
                print!("  [{}] Loading model {}... ", target.label, model);
                match clients[idx].load_model(model).await {
                    Ok((wall, server)) => {
                        let display = server.unwrap_or(wall);
                        println!("{:.0}ms", display);
                        Some(display)
                    }
                    Err(e) => {
                        println!("FAILED: {} (skipping)", e);
                        continue;
                    }
                }
            };

            for &num_ctx in &args.num_ctx {
                for (prompt_name, prompt_text) in &prompt_specs {
                    println!();
                    println!("  ===== {} | model={} | num_ctx={} | prompt={} =====",
                        target.label, model, num_ctx, prompt_name);

                    let cell = run_cell(
                        &clients[idx],
                        target,
                        model,
                        num_ctx,
                        prompt_name,
                        prompt_text,
                        args.max_tokens,
                        args.iterations,
                        args.warmup,
                        args.no_unload,
                        args.stream,
                        !args.no_gpu_sample,
                        args.gpu_sample_interval_ms,
                        !args.no_energy,
                        args.carbon_intensity,
                        load_time,
                        image_b64.as_deref(),
                        &args.require_substr,
                        args.session_id.as_deref(),
                    ).await;
                    all_cells.push(cell);
                }
            }
        }
    }

    // Final cleanup
    for (j, target) in targets.iter().enumerate() {
        for model in &args.models {
            if failed_models.contains(&(j, model.clone())) { continue; }
            let _ = clients[j].unload_model(model).await;
        }
        let _ = target;
    }

    // Per-cell tables
    for cell in &all_cells {
        stats::print_table(
            &format!("{} ({})", cell.target.label, cell.target.url),
            &format!("{} | num_ctx={} | prompt={}", cell.model, cell.num_ctx, cell.prompt_name),
            cell.prompt_chars,
            args.max_tokens,
            cell.iterations.len(),
            &cell.stats,
        );
        if let Some(preview) = cell.first_response_preview.as_deref() {
            if !preview.is_empty() {
                println!("  Coherence preview: > {}", preview);
            }
        }
        if let Some(pass) = cell.coherence_pass {
            println!("  Coherence gate:    {}", if pass { "PASS" } else { "FAIL" });
        }
        // GPU summary (if any sampler ran)
        print_gpu_summary(&cell.gpu_samples_per_iter);
        print_host_summary(&cell.host_samples_per_iter);
    }

    // Per (model, num_ctx, prompt) comparison across targets
    if targets.len() == 2 {
        print_sweep_comparison(&all_cells, &targets);
    }

    // JSON / CSV
    if let Some(ref path) = args.output {
        save_json(path, &args, &all_cells, idle_energy.as_ref());
    }
    if let Some(ref path) = args.output_csv {
        save_csv(path, &all_cells);
    }

    // Baseline comparison — verify model identity, then compute per-cell deltas.
    if let Some(ref path) = args.baseline {
        compare_to_baseline(path, &all_cells);
    }
}

/// Compare current bench results against a previously-saved JSON.
/// Matches cells by (target.label, model, num_ctx, prompt). Per-cell
/// completion_tok_s delta printed. Warns loudly if model fingerprint
/// differs — that's drift, not a code regression.
fn compare_to_baseline(path: &str, cells: &[CellResult]) {
    println!();
    println!("=== Baseline comparison vs {} ===", path);
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => { eprintln!("  Could not read baseline: {}", e); return; }
    };
    let baseline: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => { eprintln!("  Could not parse baseline JSON: {}", e); return; }
    };
    let results = match baseline.get("results").and_then(|v| v.as_array()) {
        Some(r) => r,
        None => { eprintln!("  Baseline missing 'results' array"); return; }
    };
    let mut drift_warned = false;
    for c in cells {
        // Find matching cell in baseline
        let cell_key = format!("{}|{}|{}|{}", c.target.label, c.model, c.num_ctx, c.prompt_name);
        let matched = results.iter().find(|b| {
            let target = b.get("target").and_then(|v| v.as_str()).unwrap_or("");
            let model = b.get("model").and_then(|v| v.as_str()).unwrap_or("");
            let num_ctx = b.get("num_ctx").and_then(|v| v.as_u64()).unwrap_or(0);
            let prompt = b.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
            format!("{}|{}|{}|{}", target, model, num_ctx, prompt) == cell_key
        });
        let m = match matched {
            Some(m) => m,
            None => { println!("  [{}] NO BASELINE — skipping", cell_key); continue; }
        };
        // Fingerprint check
        let cur_fp = c.model_fingerprint.as_ref().map(|(m, _, _)| m.as_str()).unwrap_or("?");
        let base_fp = m.pointer("/model_fingerprint/modified_at").and_then(|v| v.as_str()).unwrap_or("?");
        let drift = cur_fp != base_fp;
        if drift {
            drift_warned = true;
            println!("  ⚠️  [{}] DRIFT: model modified_at differs (baseline={}, current={})", cell_key, base_fp, cur_fp);
        }
        // Completion tok/s delta (find stat named "Completion tok/s")
        let cur_tok_s = c.stats.iter().find(|s| s.label == "Completion tok/s").map(|s| s.mean);
        // JSON pointer per RFC 6901: '/' inside key is escaped as ~1
        let base_tok_s = m.pointer("/stats/Completion tok~1s/mean").and_then(|v| v.as_f64());
        match (cur_tok_s, base_tok_s) {
            (Some(cur), Some(base)) => {
                let delta_pct = ((cur - base) / base) * 100.0;
                let arrow = if delta_pct > 1.0 { "↑" } else if delta_pct < -1.0 { "↓" } else { "=" };
                println!(
                    "  {} [{}] {} {:.1} → {:.1} tok/s  ({:+.1}%)",
                    arrow, cell_key, if drift { "(DRIFT)" } else { "" }, base, cur, delta_pct,
                );
            }
            _ => println!("  ? [{}] stats unavailable for delta", cell_key),
        }
    }
    if drift_warned {
        println!();
        println!("  NOTE: Some cells had model fingerprint drift — those deltas are NOT");
        println!("  code-vs-code comparisons. See project_deepcoder_bisect_2026_05_25.md.");
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_cell<'a>(
    client: &BenchClient,
    target: &'a ServerTarget,
    model: &str,
    num_ctx: usize,
    prompt_name: &str,
    prompt: &str,
    max_tokens: usize,
    iterations: usize,
    warmup: usize,
    // Kept on the signature for now to match the CLI flag's shape; the
    // per-iteration unload it gated has been removed (see comment at the
    // top of the iteration loop). Callers can keep passing it.
    _no_unload: bool,
    stream: bool,
    gpu_sample: bool,
    gpu_interval_ms: u64,
    energy_measure: bool,
    carbon_intensity: f64,
    load_time_ms: Option<f64>,
    images: Option<&[String]>,
    require_substr: &[String],
    session_id: Option<&str>,
) -> CellResult<'a> {
    let mode = if stream { "streaming" } else { "non-streaming" };

    // Warmup (no GPU sampling, results discarded). If warmup hits TWO
    // consecutive failures the model is likely in a half-failed state
    // from the prior cell (typical after VRAM contention) — unload +
    // reload it once before continuing so subsequent cells see fresh
    // state rather than cascading errors.
    if warmup > 0 {
        print!("  [{}] Warmup {} ({})... ", target.label, mode, warmup);
        let mut consec_failures = 0usize;
        let mut recovered = false;
        for i in 0..warmup {
            let result = if stream {
                client.generate_stream(model, prompt, max_tokens, Some(num_ctx), images, session_id).await
            } else {
                client.generate(model, prompt, max_tokens, Some(num_ctx), images, session_id).await
            };
            match result {
                Ok(m) => {
                    let tok_s = m.completion_tok_s.map(|t| format!("{:.1}", t)).unwrap_or_default();
                    print!("#{} {} tok/s ", i + 1, tok_s);
                    consec_failures = 0;
                }
                Err(e) => {
                    print!("#{} ERR({}) ", i + 1, e);
                    consec_failures += 1;
                    if consec_failures >= 2 && !recovered {
                        recovered = true;
                        let _ = client.unload_model(model).await;
                        match client.load_model(model).await {
                            Ok((_, _)) => print!("[recover: reloaded] "),
                            Err(e) => print!("[recover failed: {}] ", e),
                        }
                        consec_failures = 0;
                    }
                }
            }
        }
        println!();
    }

    // Benchmark iterations
    println!("  [{}] Running {} {} iterations...", target.label, iterations, mode);
    let mut all_metrics: Vec<IterationMetrics> = Vec::new();
    let mut all_gpu_samples: Vec<Vec<GpuSample>> = Vec::new();
    let mut all_host_samples: Vec<Option<HostSample>> = Vec::new();
    let mut all_energy: Vec<Option<EnergyWindow>> = Vec::new();

    for i in 0..iterations {
        // Previously: unload+reload before each iteration past i=0. That
        // forces cold-cache state for every measured iter and causes
        // wide variance (one slow iter dominates the mean). We isolate
        // models from each other at the perimeter level (cross-unload
        // before/after each model), so per-iteration isolation isn't
        // needed. Skipping the unload makes iter timings far more
        // reliable for chat-tuned models that have model-load + warmup
        // overhead on the first forward call.
        let _ = i;

        // Start GPU sampler before request, stop after
        let sampler = if gpu_sample { Some(GpuSampler::start(gpu_interval_ms)) } else { None };
        // Host footprint of the engine processes (RSS/swap/CPU via /proc).
        let host_sampler = HostSampler::start(gpu_interval_ms.max(200), &["server", "ollama"]);
        // Energy/carbon window (NVML energy counter + RAPL). Spans the whole
        // request including prefill; per-token energy uses tokens_generated.
        let energy_sampler = if energy_measure { Some(EnergySampler::start(gpu_interval_ms)) } else { None };

        let result = if stream {
            client.generate_stream(model, prompt, max_tokens, Some(num_ctx), images, session_id).await
        } else {
            client.generate(model, prompt, max_tokens, Some(num_ctx), images, session_id).await
        };

        // Prefill→decode boundary (first-token wall time) so the energy sampler
        // isolates decode-phase energy — the cache-confound-free per-token metric.
        let decode_start_s = result
            .as_ref()
            .ok()
            .and_then(|m| m.first_token_wall_ms)
            .map(|ms| ms / 1000.0);
        let energy_window = match energy_sampler {
            Some(s) => Some(s.stop(decode_start_s).await),
            None => None,
        };
        let gpu_summary = if let Some(s) = sampler { s.stop().await } else { Vec::new() };
        let host_summary = host_sampler.stop().await;

        match result {
            Ok(metrics) => {
                let tok_s = metrics.completion_tok_s.map(|t| format!("{:.1}", t)).unwrap_or("?".into());
                let ttft = metrics.ttft_ms.map(|t| format!("{:.0}ms", t)).unwrap_or("?".into());
                let tokens = metrics.tokens_generated.unwrap_or(0);
                // Show content tokens alongside total when they differ (template-leak warning).
                let tokens_str = match metrics.content_tokens {
                    Some(c) if c != tokens => format!("{}={}+{}tpl", tokens, c, tokens.saturating_sub(c)),
                    _ => format!("{}", tokens),
                };
                let itl_str = format_itl_summary(&metrics.inter_token_latencies_ms);
                println!(
                    "    #{}: {} tok/s, TTFT {}, {} tokens, ITL[{}], {:.0}ms total",
                    i + 1, tok_s, ttft, tokens_str, itl_str, metrics.e2e_latency_ms
                );
                if let Some(preview) = metrics.response_preview.as_deref() {
                    if !preview.is_empty() {
                        println!("        > {}", preview);
                    }
                }
                // Energy line: J/tok over the window using tokens_generated.
                if let Some(ew) = energy_window.as_ref() {
                    let toks = metrics.tokens_generated.unwrap_or(0);
                    let jpt = ew.j_per_tok(toks);
                    let dom = if ew.domains_counted.is_empty() { "none".to_string() } else { ew.domains_counted.join("+") };
                    match jpt {
                        Some(j) => {
                            let gpt = if toks > 0 { ew.gpu_energy_j / toks as f64 } else { 0.0 };
                            let cpt = if toks > 0 { ew.cpu_pkg_energy_j / toks as f64 } else { 0.0 };
                            // Decode-phase (prefill-excluded) J/tok — the honest
                            // cross-engine metric (immune to the prompt-cache confound).
                            let dec = ew.j_per_decode_tok(toks);
                            let dec_str = match dec {
                                Some(d) => format!(", decode {:.4} J/tok", d),
                                None => String::new(),
                            };
                            println!(
                                "        energy: {:.2} J ({}) → {:.4} J/tok [gpu {:.4} + cpu {:.4}]{}, {:.3e} gCO2/tok",
                                ew.energy_j, dom, j, gpt, cpt, dec_str,
                                ew.gco2_per_tok(toks, carbon_intensity).unwrap_or(0.0)
                            );
                        }
                        None => println!("        energy: counters unavailable ({})",
                            ew.note.as_deref().unwrap_or("none")),
                    }
                }
                all_metrics.push(metrics);
                all_gpu_samples.push(gpu_summary);
                all_host_samples.push(host_summary);
                all_energy.push(energy_window);
            }
            Err(e) => {
                eprintln!("    #{}: ERROR -- {}", i + 1, e);
                all_gpu_samples.push(gpu_summary); // keep slot alignment
                all_host_samples.push(host_summary);
                all_energy.push(energy_window);
            }
        }
    }

    // Coherence gate. If --require-substr was set, each iteration's preview
    // must contain at least one required substring (case-insensitive). A
    // single failure marks the whole cell incoherent so a fast-but-garbage
    // run can't be claimed as a WIN.
    let coherence_pass = if require_substr.is_empty() {
        None
    } else {
        let needles: Vec<String> = require_substr.iter().map(|s| s.to_lowercase()).collect();
        let mut all_ok = !all_metrics.is_empty();
        for m in &all_metrics {
            let hay = m.response_preview.as_deref().unwrap_or("").to_lowercase();
            if !needles.iter().any(|n| hay.contains(n)) {
                all_ok = false;
                break;
            }
        }
        Some(all_ok)
    };
    if let Some(pass) = coherence_pass {
        if pass {
            println!("    [coherence] PASS ({} substring(s) matched in all {} iter)",
                require_substr.len(), all_metrics.len());
        } else {
            println!("    [coherence] FAIL — at least one iteration lacked every required substring");
        }
    }

    // Compute stats
    let mut stats = Vec::new();
    if let Some(load) = load_time_ms {
        if let Some(s) = Stats::compute("Model load", "ms", &[load]) { stats.push(s); }
    }
    let ttft: Vec<f64> = all_metrics.iter().filter_map(|m| m.ttft_ms).collect();
    if let Some(s) = Stats::compute("TTFT", "ms", &ttft) { stats.push(s); }
    let prompt_tps: Vec<f64> = all_metrics.iter().filter_map(|m| m.prompt_tok_s).collect();
    if let Some(s) = Stats::compute("Prompt tok/s", "tok/s", &prompt_tps) { stats.push(s); }
    let comp_tps: Vec<f64> = all_metrics.iter().filter_map(|m| m.completion_tok_s).collect();
    if let Some(s) = Stats::compute("Completion tok/s", "tok/s", &comp_tps) { stats.push(s); }
    // Length-invariant decode rate — fair comparison when servers have
    // different EOS handling (chat-tuned models). See
    // project_bench_apples_oranges_2026_05_26.md.
    let decode_ms: Vec<f64> = all_metrics.iter().filter_map(|m| m.decode_ms_per_token).collect();
    if let Some(s) = Stats::compute("Decode ms/tok", "ms", &decode_ms) { stats.push(s); }
    let e2e: Vec<f64> = all_metrics.iter().map(|m| m.e2e_latency_ms).collect();
    if let Some(s) = Stats::compute("E2E latency", "ms", &e2e) { stats.push(s); }
    let tokens: Vec<f64> = all_metrics.iter().filter_map(|m| m.tokens_generated.map(|t| t as f64)).collect();
    if let Some(s) = Stats::compute("Tokens generated", "count", &tokens) { stats.push(s); }
    // ITL stats: aggregate every inter-token gap from every iteration into one population
    let all_itl: Vec<f64> = all_metrics.iter().flat_map(|m| m.inter_token_latencies_ms.iter().copied()).collect();
    if let Some(s) = Stats::compute("ITL (per token)", "ms", &all_itl) { stats.push(s); }
    // Energy metrics. Pair each iteration's energy window with its token count
    // (windows align 1:1 with all_metrics — both pushed in the same Ok arm).
    let j_per_tok: Vec<f64> = all_metrics.iter().zip(all_energy.iter())
        .filter_map(|(m, e)| e.as_ref().and_then(|w| w.j_per_tok(m.tokens_generated.unwrap_or(0))))
        .collect();
    if let Some(s) = Stats::compute("Energy J/tok", "J", &j_per_tok) { stats.push(s); }
    // Decode-phase J/tok (prefill excluded) — the honest cross-engine energy metric.
    let j_per_decode_tok: Vec<f64> = all_metrics.iter().zip(all_energy.iter())
        .filter_map(|(m, e)| e.as_ref().and_then(|w| w.j_per_decode_tok(m.tokens_generated.unwrap_or(0))))
        .collect();
    if let Some(s) = Stats::compute("Energy decode J/tok", "J", &j_per_decode_tok) { stats.push(s); }
    let wh_per_tok: Vec<f64> = all_metrics.iter().zip(all_energy.iter())
        .filter_map(|(m, e)| e.as_ref().and_then(|w| w.wh_per_tok(m.tokens_generated.unwrap_or(0))))
        .collect();
    if let Some(s) = Stats::compute("Energy Wh/tok", "Wh", &wh_per_tok) { stats.push(s); }
    let gco2_per_tok: Vec<f64> = all_metrics.iter().zip(all_energy.iter())
        .filter_map(|(m, e)| e.as_ref().and_then(|w| w.gco2_per_tok(m.tokens_generated.unwrap_or(0), carbon_intensity)))
        .collect();
    if let Some(s) = Stats::compute("Carbon gCO2/tok", "g", &gco2_per_tok) { stats.push(s); }

    let first_response_preview = all_metrics
        .iter()
        .find_map(|m| m.response_preview.clone().filter(|s| !s.is_empty()));

    // Capture model fingerprint so JSON output is cross-session attributable.
    let model_fingerprint = client.model_fingerprint(model).await;
    if let Some((mod_at, params, quant)) = model_fingerprint.as_ref() {
        println!("  [{}] Model fingerprint: modified_at={} params={} quant={}",
                 target.label, mod_at, params, quant);
    }

    CellResult {
        target,
        model: model.to_string(),
        model_fingerprint,
        num_ctx,
        prompt_name: prompt_name.to_string(),
        prompt_chars: prompt.len(),
        iterations: all_metrics,
        gpu_samples_per_iter: all_gpu_samples,
        host_samples_per_iter: all_host_samples,
        energy_per_iter: all_energy,
        carbon_intensity,
        load_time_ms,
        stats,
        first_response_preview,
        coherence_pass,
    }
}

/// Compact ITL summary for inline reporting (P50, P95).
fn format_itl_summary(itl: &[f64]) -> String {
    if itl.is_empty() {
        return "-".into();
    }
    if let Some(s) = Stats::compute("itl", "ms", itl) {
        format!("p50 {:.0}ms p95 {:.0}ms", s.p50, s.p95)
    } else {
        "-".into()
    }
}

/// Print a per-GPU summary across all iterations of one cell.
/// One-line host footprint summary (peak across iterations).
fn print_host_summary(samples_per_iter: &[Option<HostSample>]) {
    let hs: Vec<&HostSample> = samples_per_iter.iter().flatten().collect();
    if hs.is_empty() {
        return;
    }
    let rss_pk = hs.iter().map(|h| h.rss_peak_mb).max().unwrap_or(0);
    let rss_avg = hs.iter().map(|h| h.rss_avg_mb).sum::<u64>() / hs.len() as u64;
    let swap_pk = hs.iter().map(|h| h.swap_peak_mb).max().unwrap_or(0);
    let cpu_pk = hs.iter().map(|h| h.cpu_peak_pct).max().unwrap_or(0);
    let cpu_avg = hs.iter().map(|h| h.cpu_avg_pct).sum::<u32>() / hs.len() as u32;
    println!(
        "  Host   engine procs                 RSS pk {:>6} MB  avg {:>6} MB  swap pk {:>5} MB  CPU avg {:>4}%  pk {:>4}%",
        rss_pk, rss_avg, swap_pk, cpu_avg, cpu_pk
    );
}

fn print_gpu_summary(samples_per_iter: &[Vec<GpuSample>]) {
    // Aggregate per gpu_index across iterations.
    let mut by_idx: BTreeMap<u32, (String, Vec<&GpuSample>)> = BTreeMap::new();
    for iter_samples in samples_per_iter {
        for s in iter_samples {
            let entry = by_idx.entry(s.gpu_index).or_insert_with(|| (s.gpu_name.clone(), Vec::new()));
            entry.1.push(s);
        }
    }
    if by_idx.is_empty() {
        return;
    }
    println!("  GPU usage during this cell:");
    println!("  {:<6} {:<28} {:>12} {:>12} {:>10} {:>10}", "GPU", "Name", "VRAM peak", "VRAM avg", "Util pk", "Power pk");
    for (idx, (name, samples)) in &by_idx {
        let vram_peak = samples.iter().map(|s| s.vram_peak_mb).max().unwrap_or(0);
        let vram_avg = if !samples.is_empty() {
            samples.iter().map(|s| s.vram_avg_mb).sum::<u64>() / samples.len() as u64
        } else { 0 };
        let util_peak = samples.iter().map(|s| s.util_peak_pct).max().unwrap_or(0);
        let power_peak = samples.iter().map(|s| s.power_peak_w).fold(0.0_f64, f64::max);
        let short_name: String = name.chars().take(28).collect();
        println!("  {:<6} {:<28} {:>10}MB {:>10}MB {:>9}% {:>8.0}W",
            idx, short_name, vram_peak, vram_avg, util_peak, power_peak);
    }
    println!();
}

/// Side-by-side comparison across two targets, one block per (model, num_ctx, prompt).
fn print_sweep_comparison(cells: &[CellResult], targets: &[ServerTarget]) {
    if targets.len() != 2 { return; }
    // Group cells by (model, num_ctx, prompt)
    let mut groups: BTreeMap<(String, usize, String), Vec<&CellResult>> = BTreeMap::new();
    for cell in cells {
        let key = (cell.model.clone(), cell.num_ctx, cell.prompt_name.clone());
        groups.entry(key).or_default().push(cell);
    }

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════════════╗");
    println!("  ║ SWEEP COMPARISON ({} vs {})", targets[0].label, targets[1].label);
    println!("  ╚══════════════════════════════════════════════════════════════════════╝");

    for ((model, num_ctx, prompt), group) in &groups {
        let cell_a = group.iter().find(|c| c.target.label == targets[0].label);
        let cell_b = group.iter().find(|c| c.target.label == targets[1].label);
        if let (Some(a), Some(b)) = (cell_a, cell_b) {
            println!();
            println!("  --- model={} | num_ctx={} | prompt={} ---", model, num_ctx, prompt);
            stats::print_comparison(&a.target.label, &a.stats, &b.target.label, &b.stats);
        }
    }
}

// JSON output -----------------------------------------------------------------

fn save_json(path: &str, args: &Args, cells: &[CellResult], idle_energy: Option<&EnergyWindow>) {
    let results: Vec<serde_json::Value> = cells.iter().map(|c| {
        let stats_map: serde_json::Map<String, serde_json::Value> = c.stats.iter().map(|s| {
            (s.label.clone(), serde_json::json!({
                "mean": s.mean,
                "stddev": s.stddev,
                "min": s.min,
                "max": s.max,
                "p50": s.p50,
                "p95": s.p95,
                "unit": s.unit,
                "count": s.count,
            }))
        }).collect();

        let iterations_json: Vec<serde_json::Value> = c.iterations.iter().enumerate().map(|(i, m)| {
            let gpu = c.gpu_samples_per_iter.get(i).cloned().unwrap_or_default();
            let host = c.host_samples_per_iter.get(i).cloned().flatten();
            // Energy: derive per-token metrics from the window + this iter's tokens.
            let energy = c.energy_per_iter.get(i).and_then(|e| e.as_ref()).map(|w| {
                let toks = m.tokens_generated.unwrap_or(0);
                serde_json::json!({
                    "energy_j": w.energy_j,
                    "gpu_energy_j": w.gpu_energy_j,
                    "cpu_pkg_energy_j": w.cpu_pkg_energy_j,
                    "dram_energy_j": w.dram_energy_j,
                    "duration_s": w.duration_s,
                    "j_per_tok": w.j_per_tok(toks),
                    "wh_per_tok": w.wh_per_tok(toks),
                    "gco2_per_tok": w.gco2_per_tok(toks, c.carbon_intensity),
                    "carbon_intensity_gco2_per_kwh": c.carbon_intensity,
                    "gpu_path": w.gpu_path,
                    "domains_counted": w.domains_counted,
                    "cpu_rapl_available": w.cpu_rapl_available,
                    "dram_rapl_available": w.dram_rapl_available,
                    "note": w.note,
                    // Decode-phase (prefill-excluded) — the cache-confound-free metric.
                    "gpu_decode_j": w.gpu_decode_j,
                    "cpu_pkg_decode_j": w.cpu_pkg_decode_j,
                    "dram_decode_j": w.dram_decode_j,
                    "decode_energy_j": w.decode_energy_j,
                    "decode_duration_s": w.decode_duration_s,
                    "j_per_decode_tok": w.j_per_decode_tok(toks),
                })
            });
            serde_json::json!({
                "iteration": i + 1,
                "wall_clock_ms": m.wall_clock_ms,
                "load_time_ms": m.load_time_ms,
                "ttft_ms": m.ttft_ms,
                "prompt_tokens": m.prompt_tokens,
                "prompt_tok_s": m.prompt_tok_s,
                "completion_tok_s": m.completion_tok_s,
                "e2e_latency_ms": m.e2e_latency_ms,
                "tokens_generated": m.tokens_generated,
                "inter_token_latencies_ms": m.inter_token_latencies_ms,
                "response_preview": m.response_preview,
                "gpu_samples": gpu,
                "host_sample": host,
                "energy": energy,
            })
        }).collect();

        let fp = c.model_fingerprint.as_ref().map(|(m, p, q)| {
            serde_json::json!({"modified_at": m, "parameter_size": p, "quantization_level": q})
        });
        serde_json::json!({
            "target": c.target.label,
            "url": c.target.url,
            "model": c.model,
            "model_fingerprint": fp,
            "num_ctx": c.num_ctx,
            "prompt": c.prompt_name,
            "prompt_chars": c.prompt_chars,
            "load_time_ms": c.load_time_ms,
            "first_response_preview": c.first_response_preview,
            "coherence_pass": c.coherence_pass,
            "stats": serde_json::Value::Object(stats_map),
            "iterations": iterations_json,
        })
    }).collect();

    let payload = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "config": {
            "models": args.models,
            "num_ctx": args.num_ctx,
            "prompts": args.prompts,
            "custom_prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "streaming": args.stream,
            "gpu_sample": !args.no_gpu_sample,
            "energy_measure": !args.no_energy,
            "carbon_intensity_gco2_per_kwh": args.carbon_intensity,
        },
        "idle_energy_baseline": idle_energy.map(|w| serde_json::json!({
            "energy_j": w.energy_j,
            "gpu_energy_j": w.gpu_energy_j,
            "cpu_pkg_energy_j": w.cpu_pkg_energy_j,
            "dram_energy_j": w.dram_energy_j,
            "duration_s": w.duration_s,
            "idle_watts": if w.duration_s > 0.0 { w.energy_j / w.duration_s } else { 0.0 },
            "domains_counted": w.domains_counted,
            "note": w.note,
        })),
        "results": results,
    });

    match std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap()) {
        Ok(()) => println!("  Results saved to {}", path),
        Err(e) => eprintln!("  Failed to save JSON: {}", e),
    }
}

// CSV output ------------------------------------------------------------------

fn save_csv(path: &str, cells: &[CellResult]) {
    let mut out = String::new();
    out.push_str("target,model,num_ctx,prompt,iteration,wall_clock_ms,ttft_ms,prompt_tok_s,completion_tok_s,e2e_latency_ms,tokens_generated,itl_count,itl_p50_ms,itl_p95_ms\n");
    for c in cells {
        for (i, m) in c.iterations.iter().enumerate() {
            let (itl_p50, itl_p95) = if let Some(s) = Stats::compute("itl", "ms", &m.inter_token_latencies_ms) {
                (format!("{:.3}", s.p50), format!("{:.3}", s.p95))
            } else {
                (String::new(), String::new())
            };
            out.push_str(&format!(
                "{},{},{},{},{},{:.3},{},{},{},{:.3},{},{},{},{}\n",
                csv_esc(&c.target.label),
                csv_esc(&c.model),
                c.num_ctx,
                csv_esc(&c.prompt_name),
                i + 1,
                m.wall_clock_ms,
                m.ttft_ms.map(|v| format!("{:.3}", v)).unwrap_or_default(),
                m.prompt_tok_s.map(|v| format!("{:.3}", v)).unwrap_or_default(),
                m.completion_tok_s.map(|v| format!("{:.3}", v)).unwrap_or_default(),
                m.e2e_latency_ms,
                m.tokens_generated.map(|v| v.to_string()).unwrap_or_default(),
                m.inter_token_latencies_ms.len(),
                itl_p50,
                itl_p95,
            ));
        }
    }
    match std::fs::write(path, out) {
        Ok(()) => println!("  Per-iteration CSV saved to {}", path),
        Err(e) => eprintln!("  Failed to save CSV: {}", e),
    }
}

fn csv_esc(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// Helpers ---------------------------------------------------------------------

/// Normalize a URL: accept "host:port", "http://host:port", or just a port number
fn normalize_url(input: &str) -> String {
    let trimmed = input.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else if let Ok(port) = trimmed.parse::<u16>() {
        format!("http://localhost:{}", port)
    } else {
        format!("http://{}", trimmed)
    }
}
