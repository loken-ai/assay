//! LLM Benchmark — compare Ollama vs LOKEN performance
//!
//! Sweeps over models × num_ctx × prompts and reports per-token streaming
//! metrics + per-GPU sampling for each combination.
//!
//! Usage:
//!   # Single model, single context, default prompts
//!   assay --ollama 11434 --loken 11435 --models qwen3:latest --stream
//!
//!   # Full sweep with JSON + CSV output
//!   assay --ollama 11434 --loken 11435 \
//!       --models qwen3:latest,devstral-small-2:latest \
//!       --num-ctx 2048,4096 --stream -n 3 \
//!       -o sweep.json --output-csv sweep.csv

mod api;
mod energy;
mod gpu_sampler;
mod host_sampler;
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

// The `2.5k` prompt: a long single request that fills the KV cache while asking for the
// same completion length as the other cells, so the column isolates decode-rate falloff as
// KV grows rather than mixing in a different output size. The text is a technical article on
// processor architecture followed by comprehension questions — long, ordinary English prose,
// chosen because it tokenises the way real input does. Kept in-tree rather than passed by
// path, so a run is reproducible from the repository alone.
const PROMPT_2_5K: &str = include_str!("../prompts/p2_5k.txt");

// Context fillers. Uniform word salad rather than prose, on purpose: the point is to occupy a
// known number of tokens without the model finding structure to latch onto, so the column
// measures decode falloff as the KV cache grows and nothing else. Ordinary text of the same
// length would let a model predict its way through and confound the two.
const PROMPT_CTX_2000: &str = include_str!("../prompts/ctx_2000.txt");
const PROMPT_CTX_8000: &str = include_str!("../prompts/ctx_8000.txt");

/// Resolve a built-in prompt name to its content; returns None if unknown.
fn builtin_prompt(name: &str) -> Option<&'static str> {
    match name {
        "short" => Some(PROMPT_SHORT),
        "medium" => Some(PROMPT_MEDIUM),
        "long" => Some(PROMPT_LONG),
        "2.5k" | "2.5K" => Some(PROMPT_2_5K),
        "ctx2000" => Some(PROMPT_CTX_2000),
        "ctx8000" => Some(PROMPT_CTX_8000),
        "vision_short" => Some(PROMPT_VISION_SHORT),
        "vision_medium" => Some(PROMPT_VISION_MEDIUM),
        "vision_long" => Some(PROMPT_VISION_LONG),
        _ => None,
    }
}

// CLI -------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "assay")]
#[command(
    about = "Inference server benchmark — compare Ollama vs LOKEN across models, contexts, and prompts"
)]
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

    /// Force Ollama's `num_gpu` (layers offloaded to GPU). Ollama-only — the other engines
    /// never receive it, and are put on the CPU by how they were launched instead. Which
    /// cards each engine may use is decided at launch, not here: see scripts/gpu-policy.sh.
    #[arg(long)]
    num_gpu: Option<usize>,

    /// Comma-separated list of prompt names. Text suite: short, medium, long
    /// (the default). Vision suite: vision_short, vision_medium, vision_long —
    /// require --image to be passed too.
    #[arg(long, value_delimiter = ',', default_values_t = vec!["short".to_string(), "medium".to_string(), "long".to_string()])]
    prompts: Vec<String>,

    /// Give every iteration a prompt no engine has seen, by prefixing a counter.
    ///
    /// Engines reuse a computed prefix, so replaying one prompt measures that cache rather
    /// than the prefill — the figure that comes back is a cache hit wearing a rate's units.
    /// Use this whenever the prefill number is the point.
    #[arg(long)]
    pub unique_prompt: bool,

    /// Custom prompt text (overrides --prompts; runs once per sweep cell)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum completion tokens
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Number of benchmark iterations per (target, model, num_ctx, prompt) cell
    #[arg(short = 'n', long, default_value = "3")]
    iterations: usize,

    /// Requests in flight at once. 1 (the default) keeps the sequential protocol exactly
    /// as it was; above 1 the iterations of a cell are issued together and the cell also
    /// reports an AGGREGATE rate - total tokens over the wall time of the flight.
    ///
    /// Per-request rates and aggregate rate answer different questions and neither
    /// substitutes for the other. A single request cannot go faster for a second node
    /// existing - it can only pay a hop. What a second node buys is served requests per
    /// second, and that is invisible to a sequential sweep.
    #[arg(long, default_value = "1")]
    concurrency: usize,

    /// Number of warmup iterations (not counted in stats)
    #[arg(short, long, default_value = "1")]
    warmup: usize,

    /// Save aggregated results to JSON file
    #[arg(short, long)]
    output: Option<String>,

    /// Compare current run against a previous bench JSON. Per-cell
    /// completion_tok_s deltas printed at the end. Warns if model
    /// fingerprints differ between runs (model was re-pulled — the
    /// "regression" is then a different set of weights, not different code).
    #[arg(long)]
    baseline: Option<String>,

    /// Save per-iteration data to CSV file
    #[arg(long)]
    output_csv: Option<String>,

    /// Use streaming mode (measures real TTFT + per-token ITL)
    #[arg(long)]
    stream: bool,

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
    /// comma-separated substrings. For runs where a model can answer fast and wrong: a
    /// degenerate completion produces an ordinary token rate, so throughput alone cannot
    /// tell it from a good one.
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

    /// Process names to weigh for the host-footprint columns, comma-separated. The default
    /// covers the three engines this tool drives; vLLM runs as `python`, so a list that omits
    /// it reports a footprint for two engines and none for the third.
    #[arg(long, value_delimiter = ',', default_values_t = ["loken".to_string(), "ollama".to_string(), "python".to_string(), "vllm".to_string()])]
    host_proc_names: Vec<String>,

    /// Sample idle (no-request) power for this many seconds at startup and record it in the
    /// JSON output as `idle_energy_baseline`. It is NOT subtracted from the reported figures:
    /// every J/token here includes the machine's idle draw, and this is the number to subtract
    /// it with, in your own analysis. 0 = off.
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
    /// verify model identity: a tag that has been re-pulled between two runs can point at
    /// different weights, and the delta then measures the checkpoint rather than the code.
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
    /// for the coherence check by eye — a fast answer and a fast wrong answer have the
    /// same rate, and only the text separates them.
    first_response_preview: Option<String>,
    /// Coherence pass/fail flag. Always set when the cell produced text: a
    /// degenerate answer fails on its own shape, and `--require-substr` adds an
    /// expected-content gate on top when a caller supplies one.
    coherence_pass: Option<bool>,
}

/// Whether a response is a repeating pattern rather than an answer.
///
/// This is the check that needs no expected answer, which is the only reason it will
/// actually be run: a gate that has to be told what the model should say is a gate every
/// sweep leaves off, and one did - `coherence_pass` was null in every cell of a whole
/// campaign while gemma4:31b emitted "--- --- --- ---" for 128 tokens and the table
/// recorded it as a throughput result against ollama's prose.
///
/// A degenerate answer repeats a handful of units over and over, so the share of DISTINCT
/// units collapses. Real prose and real code both sit far above the threshold - the
/// separation is not delicate, which matters because a false accusation of garbage is
/// worse than the gap it closes.
fn looks_degenerate(text: &str) -> bool {
    // Undecodable bytes. gemma4:31b answered "kingdom" then 50 replacement characters and
    // the first version of this check passed it: they are all DISTINCT as units, so a
    // distinct-ratio sees variety where there is only damage.
    let chars = text.chars().count();
    if chars >= 20 && text.chars().filter(|c| *c == '\u{FFFD}').count() * 10 > chars {
        return true;
    }
    let units: Vec<&str> = text.split_whitespace().collect();
    if units.len() < 12 {
        return false; // too short to tell repetition from brevity
    }
    // A single unit repeated: "--- --- ---", "olde olde olde". Needs a longer run than the
    // phrase check below to be conclusive, so it is gated separately rather than by an early
    // return - the first version returned at 15 units and never reached the phrase check,
    // which is how a 12-unit "time-olde olde olde ..." passed.
    let distinct: std::collections::HashSet<&str> = units.iter().copied().collect();
    if units.len() >= 15 && (distinct.len() as f64) / (units.len() as f64) < 0.15 {
        return true;
    }
    // A PHRASE repeated, which the ratio above cannot see: "the end of a person, the end of
    // a person, ..." has a perfectly ordinary share of distinct words. Four of the five
    // gemma4 and lfm2 answers that were visibly looping passed the first check for exactly
    // this reason. Count how much of the text one three-word window covers.
    // Counting DISTINCT windows rather than the commonest one: a cycle of k words gives every
    // window a share of 1/k, so a "commonest window" threshold has to be tuned per cycle
    // length and misses the long ones - "the end of a person, ..." repeats five words and no
    // single window covers more than 22%. Distinct windows over total lands near 1.0 for
    // prose and collapses toward k/n for anything looping, whatever k is.
    let windows: Vec<[&str; 3]> = units.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    let distinct_windows: std::collections::HashSet<&[&str; 3]> = windows.iter().collect();
    if (distinct_windows.len() as f64) / (windows.len() as f64) < 0.5 {
        return true;
    }
    // A sane opening followed by a looping tail - "king who was able to see a time, that,
    // that, that, ..." - keeps a high share of distinct windows because the head supplies
    // them. What gives it away is one unit owning most of the answer.
    let mut unit_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for u in &units {
        *unit_counts.entry(*u).or_insert(0) += 1;
    }
    unit_counts
        .values()
        .max()
        .map(|m| (*m as f64) / (units.len() as f64) > 0.40)
        .unwrap_or(false)
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
                    eprintln!("Error: unknown prompt name '{}' (valid: short, medium, long, 2.5k, ctx2000, ctx8000, vision_short, vision_medium, vision_long)", name);
                    std::process::exit(1);
                }
            }
        }
        out
    };

    // Load + base64-encode the image once at startup (if --image was given).
    // We hold base64 in memory across the entire sweep so per-iteration
    // latency only includes the HTTP send — decoding the file per iteration would add
    // variance that has nothing to do with the model.
    let image_b64: Option<Vec<String>> = match args.image.as_deref() {
        Some(path) => {
            use base64::Engine;
            match std::fs::read(path) {
                Ok(bytes) => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    println!(
                        "  Image:      {} ({} bytes raw, {} bytes b64)",
                        path,
                        bytes.len(),
                        encoded.len()
                    );
                    // A vision run without a session id pays a cold prefill on every
                    // iteration, so the figure describes the first-token outlier rather than
                    // the steady state. It is NOT set automatically: prefix-KV reuse is a
                    // feature some engines implement and others do not, and switching it on
                    // by default would turn it on for one side of the comparison only.
                    if args.session_id.is_none() {
                        eprintln!(
                            "\n  ℹ️  Vision run without --session-id: every iteration pays a\n  \
                                       \x20  cold image prefill, which measures the first-token\n  \
                                       \x20  outlier rather than the steady state. Passing an id\n  \
                                       \x20  enables prefix-KV reuse on engines that implement it\n  \
                                       \x20  — and on those alone, so the run is then no longer\n  \
                                       \x20  a like-for-like comparison.\n"
                        );
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
        targets.push(ServerTarget {
            label: "Ollama".into(),
            url: normalize_url(url),
            protocol: Protocol::Ollama,
        });
    }
    if let Some(ref url) = args.loken {
        targets.push(ServerTarget {
            label: "LOKEN".into(),
            url: normalize_url(url),
            protocol: Protocol::Ollama,
        });
    }
    if let Some(ref url) = args.vllm {
        targets.push(ServerTarget {
            label: "vLLM".into(),
            url: normalize_url(url),
            protocol: Protocol::OpenAI,
        });
    }

    // vLLM reports no prefill/decode timing split — without --stream the decode
    // rate is wall-clock-inflated by prefill. Warn so a non-stream vLLM run
    // isn't misread as a loss.
    if !args.stream && targets.iter().any(|t| t.protocol == Protocol::OpenAI) {
        eprintln!(
            "\n  ⚠️  vLLM target without --stream: decode tok/s will include prefill time\n  \
                       \x20  (the OpenAI completions API gives no prefill/decode split).\n  \
                       \x20  Pass --stream for a fair, length-invariant decode comparison.\n"
        );
    }

    let clients: Vec<BenchClient> = targets
        .iter()
        .map(|t| {
            let mut c = BenchClient::with_protocol(t.url.clone(), args.verbose, t.protocol);
            c.set_num_gpu(args.num_gpu);
            c
        })
        .collect();

    // Header
    println!();
    println!("  LLM Benchmark v{}", env!("CARGO_PKG_VERSION"));
    println!("  Models:     {}", args.models.join(", "));
    println!(
        "  num_ctx:    {}",
        args.num_ctx
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  Prompts:    {}",
        prompt_specs
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  Max tokens: {}", args.max_tokens);
    println!(
        "  Iterations: {} (+ {} warmup)",
        args.iterations, args.warmup
    );
    if args.concurrency > 1 {
        println!("  Concurrency: {} requests in flight", args.concurrency);
    }
    println!("  Streaming:  {}", if args.stream { "yes" } else { "no" });
    println!(
        "  GPU sample: {}",
        if args.no_gpu_sample { "off" } else { "on" }
    );
    println!(
        "  Energy:     {}",
        if args.no_energy {
            "off".to_string()
        } else {
            format!("on (CI={} gCO2/kWh)", args.carbon_intensity)
        }
    );
    for t in &targets {
        println!("  {:<11} {}", format!("{}:", t.label), t.url);
    }
    println!();

    // Optional idle-energy baseline: sample energy with no request in flight so
    // the later report can separate active from idle draw. Done once, up front.
    let idle_energy: Option<EnergyWindow> = if !args.no_energy && args.idle_energy_secs > 0 {
        println!(
            "  Sampling idle energy baseline for {}s (no request)...",
            args.idle_energy_secs
        );
        let s = EnergySampler::start(args.gpu_sample_interval_ms);
        tokio::time::sleep(Duration::from_secs(args.idle_energy_secs)).await;
        let w = s.stop(None).await;
        let idle_w = if w.duration_s > 0.0 {
            w.energy_j / w.duration_s
        } else {
            0.0
        };
        println!(
            "  Idle baseline: {:.2} J over {:.1}s = {:.1} W ({}){}",
            w.energy_j,
            w.duration_s,
            idle_w,
            if w.domains_counted.is_empty() {
                "no counters".to_string()
            } else {
                w.domains_counted.join("+")
            },
            w.note
                .as_deref()
                .map(|n| format!("  [{}]", n))
                .unwrap_or_default(),
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
                print!(
                    "  [{}] Unloading {} loaded model(s)... ",
                    other.label,
                    loaded.len()
                );
                for m in &loaded {
                    match clients[j].unload_model(m).await {
                        Ok(()) => ok += 1,
                        Err(_) => skipped += 1,
                    }
                }
                println!("ok={} skip={}", ok, skipped);
            }
            // Wait for VRAM to actually drop before loading the next
            // model. An unload request can return before the memory is actually back:
            // the runner process holding it exits on its own schedule. Polling is more
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
                        .output()
                        .ok()
                        .and_then(|o| String::from_utf8(o.stdout).ok())
                        .and_then(|s| s.lines().next().and_then(|l| l.trim().parse::<u32>().ok()));
                    if let Some(mb) = used_mb {
                        // < 2 GB: GPU0 is essentially idle (~hundreds of MB
                        // baseline used by drivers + the bench's own server).
                        if mb < 2048 {
                            println!("  [vram-wait] GPU0 freed at {}MB after {}s", mb, poll + 1);
                            break;
                        }
                        if poll == 29 {
                            println!(
                                "  [vram-wait] GPU0 still at {}MB after 30s — proceeding anyway",
                                mb
                            );
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
                print!(
                    "  [{}] Checking vLLM server (model resident at launch)... ",
                    target.label
                );
                match clients[idx].load_model(model).await {
                    Ok(_) => {
                        println!("ready");
                        None
                    }
                    Err(e) => {
                        println!("FAILED: {} (skipping)", e);
                        continue;
                    }
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
                    println!(
                        "  ===== {} | model={} | num_ctx={} | prompt={} =====",
                        target.label, model, num_ctx, prompt_name
                    );

                    let cell = run_cell(
                        &clients[idx],
                        target,
                        model,
                        num_ctx,
                        prompt_name,
                        prompt_text,
                        args.unique_prompt,
                        args.max_tokens,
                        args.iterations,
                        args.concurrency,
                        args.warmup,
                        args.stream,
                        !args.no_gpu_sample,
                        args.gpu_sample_interval_ms,
                        !args.no_energy,
                        args.carbon_intensity,
                        load_time,
                        image_b64.as_deref(),
                        &args.require_substr,
                        args.session_id.as_deref(),
                        &args.host_proc_names,
                    )
                    .await;
                    all_cells.push(cell);
                }
            }
        }
    }

    // Final cleanup
    for (j, _target) in targets.iter().enumerate() {
        for model in &args.models {
            if failed_models.contains(&(j, model.clone())) {
                continue;
            }
            let _ = clients[j].unload_model(model).await;
        }
    }

    // Per-cell tables
    for cell in &all_cells {
        stats::print_table(
            &format!("{} ({})", cell.target.label, cell.target.url),
            &format!(
                "{} | num_ctx={} | prompt={}",
                cell.model, cell.num_ctx, cell.prompt_name
            ),
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
            println!(
                "  Coherence gate:    {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        // GPU summary (if any sampler ran)
        print_gpu_summary(&cell.gpu_samples_per_iter);
        print_host_summary(&cell.host_samples_per_iter);
    }

    // Per (model, num_ctx, prompt) comparison across targets
    // Pairwise, so a three-engine sweep is not silently left without any comparison at all —
    // which is what a bare `== 2` did, on the very run this tool exists for.
    for (i, a) in targets.iter().enumerate() {
        for b in targets.iter().skip(i + 1) {
            print_sweep_comparison(&all_cells, a, b);
        }
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
        Err(e) => {
            eprintln!("  Could not read baseline: {}", e);
            return;
        }
    };
    let baseline: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  Could not parse baseline JSON: {}", e);
            return;
        }
    };
    let results = match baseline.get("results").and_then(|v| v.as_array()) {
        Some(r) => r,
        None => {
            eprintln!("  Baseline missing 'results' array");
            return;
        }
    };
    let mut drift_warned = false;
    for c in cells {
        // Find matching cell in baseline
        let cell_key = format!(
            "{}|{}|{}|{}",
            c.target.label, c.model, c.num_ctx, c.prompt_name
        );
        let matched = results.iter().find(|b| {
            let target = b.get("target").and_then(|v| v.as_str()).unwrap_or("");
            let model = b.get("model").and_then(|v| v.as_str()).unwrap_or("");
            let num_ctx = b.get("num_ctx").and_then(|v| v.as_u64()).unwrap_or(0);
            let prompt = b.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
            format!("{}|{}|{}|{}", target, model, num_ctx, prompt) == cell_key
        });
        let m = match matched {
            Some(m) => m,
            None => {
                println!("  [{}] NO BASELINE — skipping", cell_key);
                continue;
            }
        };
        // Fingerprint check
        let cur_fp = c
            .model_fingerprint
            .as_ref()
            .map(|(m, _, _)| m.as_str())
            .unwrap_or("?");
        let base_fp = m
            .pointer("/model_fingerprint/modified_at")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let drift = cur_fp != base_fp;
        if drift {
            drift_warned = true;
            println!(
                "  ⚠️  [{}] DRIFT: model modified_at differs (baseline={}, current={})",
                cell_key, base_fp, cur_fp
            );
        }
        // Completion tok/s delta (find stat named "Completion tok/s")
        let cur_tok_s = c
            .stats
            .iter()
            .find(|s| s.label == "Completion tok/s")
            .map(|s| s.mean);
        // JSON pointer per RFC 6901: '/' inside key is escaped as ~1
        let base_tok_s = m
            .pointer("/stats/Completion tok~1s/mean")
            .and_then(|v| v.as_f64());
        match (cur_tok_s, base_tok_s) {
            (Some(cur), Some(base)) => {
                let delta_pct = ((cur - base) / base) * 100.0;
                let arrow = if delta_pct > 1.0 {
                    "↑"
                } else if delta_pct < -1.0 {
                    "↓"
                } else {
                    "="
                };
                println!(
                    "  {} [{}] {} {:.1} → {:.1} tok/s  ({:+.1}%)",
                    arrow,
                    cell_key,
                    if drift { "(DRIFT)" } else { "" },
                    base,
                    cur,
                    delta_pct,
                );
            }
            _ => println!("  ? [{}] stats unavailable for delta", cell_key),
        }
    }
    if drift_warned {
        println!();
        println!("  NOTE: Some cells had model fingerprint drift — those deltas are NOT");
        println!("  code-vs-code comparisons — the weights themselves differ between runs.");
    }
}

/// Pair each iteration's energy window with the token count of that same iteration.
///
/// The pairing is positional, so it is checked rather than trusted. Both vectors are filled in
/// one arm of one match and nowhere else; if that ever stops being true, every figure derived
/// here would be a real window divided by a different request's tokens — arithmetic that looks
/// entirely plausible and is wrong. Length drift yields nothing and says so, because a missing
/// energy column is a visible problem and a wrong one is not.
fn per_token(
    metrics: &[IterationMetrics],
    energy: &[Option<EnergyWindow>],
    f: impl Fn(&EnergyWindow, u64) -> Option<f64>,
) -> Vec<f64> {
    if metrics.len() != energy.len() {
        eprintln!(
            "    ⚠️  {} iterations against {} energy windows — per-token energy withheld for this cell",
            metrics.len(),
            energy.len()
        );
        return Vec::new();
    }
    metrics
        .iter()
        .zip(energy.iter())
        .filter_map(|(m, e)| {
            e.as_ref()
                .and_then(|w| f(w, m.tokens_generated.unwrap_or(0)))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_cell<'a>(
    client: &BenchClient,
    target: &'a ServerTarget,
    model: &str,
    num_ctx: usize,
    prompt_name: &str,
    prompt: &str,
    unique_prompt: bool,
    max_tokens: usize,
    iterations: usize,
    concurrency: usize,
    warmup: usize,
    stream: bool,
    gpu_sample: bool,
    gpu_interval_ms: u64,
    energy_measure: bool,
    carbon_intensity: f64,
    load_time_ms: Option<f64>,
    images: Option<&[String]>,
    require_substr: &[String],
    session_id: Option<&str>,
    host_proc_names: &[String],
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
                client
                    .generate_stream(model, prompt, max_tokens, Some(num_ctx), images, session_id)
                    .await
            } else {
                client
                    .generate(model, prompt, max_tokens, Some(num_ctx), images, session_id)
                    .await
            };
            match result {
                Ok(m) => {
                    let tok_s = m
                        .completion_tok_s
                        .map(|t| format!("{:.1}", t))
                        .unwrap_or_default();
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
    println!(
        "  [{}] Running {} {} iterations...",
        target.label, iterations, mode
    );
    let mut all_metrics: Vec<IterationMetrics> = Vec::new();
    let mut all_gpu_samples: Vec<Vec<GpuSample>> = Vec::new();
    let mut all_host_samples: Vec<Option<HostSample>> = Vec::new();
    let mut all_energy: Vec<Option<EnergyWindow>> = Vec::new();

    // Concurrent flights. Requests are issued together and the cell reports what the
    // engine SERVED per second, which is the only figure a second node can move: one
    // request never gets faster for a peer existing, it only pays a hop.
    //
    // Sampling is per flight rather than per request - overlapping windows would each
    // attribute the whole machine's draw to one request and count the same joules N times.
    if concurrency > 1 {
        let flights = iterations.div_ceil(concurrency);
        for f in 0..flights {
            let n = concurrency.min(iterations - f * concurrency);
            let prompts: Vec<String> = (0..n)
                .map(|k| {
                    let idx = f * concurrency + k;
                    if unique_prompt {
                        format!("Request {} of a benchmark series.\n\n{}", idx + 1, prompt)
                    } else {
                        prompt.to_string()
                    }
                })
                .collect();
            let t0 = std::time::Instant::now();
            let results = futures::future::join_all(prompts.iter().map(|p| async move {
                if stream {
                    client
                        .generate_stream(model, p, max_tokens, Some(num_ctx), images, session_id)
                        .await
                } else {
                    client
                        .generate(model, p, max_tokens, Some(num_ctx), images, session_id)
                        .await
                }
            }))
            .await;
            let wall_s = t0.elapsed().as_secs_f64();
            let mut served = 0usize;
            let mut tokens = 0u64;
            for r in results {
                match r {
                    Ok(m) => {
                        served += 1;
                        tokens += m.tokens_generated.unwrap_or(0);
                        all_metrics.push(m);
                        all_gpu_samples.push(Vec::new());
                        all_host_samples.push(None);
                        all_energy.push(None);
                    }
                    // A failure is reported and not counted: a flight that served fewer
                    // requests in the same wall time would otherwise read as a slower engine
                    // rather than as a broken one.
                    Err(e) => println!("    [{}] flight request failed: {e}", target.label),
                }
            }
            println!(
                "    [{}] flight {}/{}: {}/{} served, {} tokens in {:.2}s -> {:.1} tok/s aggregate",
                target.label,
                f + 1,
                flights,
                served,
                n,
                tokens,
                wall_s,
                tokens as f64 / wall_s.max(1e-9)
            );
        }
    }

    // Sequential path. Concurrency has its own loop above; expressing "skip this" as a
    // zero-length range read as a bug every time anyone looked at it.
    //
    // Iterations deliberately do not unload between themselves. Models are isolated from one
    // another at the perimeter, and unloading per iteration forces a cold cache on every
    // measured run — which inflates exactly the variance the isolation exists to control.
    for i in (0..iterations).take_while(|_| concurrency <= 1) {
        // A prefix no engine holds, so the prefill is computed rather than recalled. The
        // counter goes in FRONT: a shared prefix with a different tail would still hit.
        let iter_prompt: String = if unique_prompt {
            format!("Request {} of a benchmark series.\n\n{}", i + 1, prompt)
        } else {
            prompt.to_string()
        };
        let prompt = iter_prompt.as_str();

        // Start GPU sampler before request, stop after
        let sampler = if gpu_sample {
            Some(GpuSampler::start(gpu_interval_ms))
        } else {
            None
        };
        // Host footprint of the engine processes (RSS/swap/CPU via /proc).
        // Which processes to weigh. vLLM runs as python, so it was never matched and its
        // host footprint column came back empty while the other two reported one — an
        // asymmetry that reads as a finding. The default names each engine's own process;
        // it is settable because a name that is generic enough to be someone else's — the
        // LOKEN server used to be called plain "server" — silently sums a stranger's memory
        // into the engine's footprint.
        let host_names: Vec<&str> = host_proc_names.iter().map(String::as_str).collect();
        let host_sampler = HostSampler::start(gpu_interval_ms.max(200), &host_names);
        // Energy/carbon window (NVML energy counter + RAPL). Spans the whole
        // request including prefill; per-token energy uses tokens_generated.
        let energy_sampler = if energy_measure {
            Some(EnergySampler::start(gpu_interval_ms))
        } else {
            None
        };

        let result = if stream {
            client
                .generate_stream(model, prompt, max_tokens, Some(num_ctx), images, session_id)
                .await
        } else {
            client
                .generate(model, prompt, max_tokens, Some(num_ctx), images, session_id)
                .await
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
        let gpu_summary = if let Some(s) = sampler {
            s.stop().await
        } else {
            Vec::new()
        };
        let host_summary = host_sampler.stop().await;

        match result {
            Ok(metrics) => {
                let tok_s = metrics
                    .completion_tok_s
                    .map(|t| format!("{:.1}", t))
                    .unwrap_or("?".into());
                let ttft = metrics
                    .ttft_ms
                    .map(|t| format!("{:.0}ms", t))
                    .unwrap_or("?".into());
                let tokens = metrics.tokens_generated.unwrap_or(0);
                // Show content tokens alongside total when they differ (template-leak warning).
                let tokens_str = match metrics.content_tokens {
                    Some(c) if c != tokens => {
                        format!("{}={}+{}tpl", tokens, c, tokens.saturating_sub(c))
                    }
                    _ => format!("{}", tokens),
                };
                let itl_str = format_itl_summary(&metrics.inter_token_latencies_ms);
                println!(
                    "    #{}: {} tok/s, TTFT {}, {} tokens, ITL[{}], {:.0}ms total",
                    i + 1,
                    tok_s,
                    ttft,
                    tokens_str,
                    itl_str,
                    metrics.e2e_latency_ms
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
                    let dom = if ew.domains_counted.is_empty() {
                        "none".to_string()
                    } else {
                        ew.domains_counted.join("+")
                    };
                    match jpt {
                        Some(j) => {
                            let gpt = if toks > 0 {
                                ew.gpu_energy_j / toks as f64
                            } else {
                                0.0
                            };
                            let cpt = if toks > 0 {
                                ew.cpu_pkg_energy_j / toks as f64
                            } else {
                                0.0
                            };
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
                        None => println!(
                            "        energy: counters unavailable ({})",
                            ew.note.as_deref().unwrap_or("none")
                        ),
                    }
                }
                all_metrics.push(metrics);
                all_gpu_samples.push(gpu_summary);
                all_host_samples.push(host_summary);
                all_energy.push(energy_window);
            }
            Err(e) => {
                // Record nothing for a failed iteration. The four vectors are read positionally
                // — j_per_tok zips energy against tokens_generated — so a slot pushed here
                // without a matching `all_metrics` entry does not "keep alignment", it shifts
                // every later window onto the wrong request's token count. A failure has no
                // token count to pair with, and its window covers a request that did not run
                // to completion, so it belongs in neither.
                eprintln!("    #{}: ERROR -- {}", i + 1, e);
            }
        }
    }
    debug_assert_eq!(
        all_metrics.len(),
        all_gpu_samples.len(),
        "gpu samples desynchronised"
    );
    debug_assert_eq!(
        all_metrics.len(),
        all_host_samples.len(),
        "host samples desynchronised"
    );

    // Coherence gate. If --require-substr was set, each iteration's preview
    // must contain at least one required substring (case-insensitive). A
    // single failure marks the whole cell incoherent, so a run that was fast and wrong
    // cannot be reported as a rate.
    let needles: Vec<String> = require_substr.iter().map(|s| s.to_lowercase()).collect();
    let mut degenerate = None;
    let mut missing_substr = false;
    let mut judged = false;
    for m in &all_metrics {
        let preview = m.response_preview.as_deref().unwrap_or("");
        if preview.trim().is_empty() {
            continue;
        }
        judged = true;
        if looks_degenerate(preview) {
            degenerate = Some(preview.chars().take(46).collect::<String>());
            break;
        }
        if !needles.is_empty() {
            let hay = preview.to_lowercase();
            if !needles.iter().any(|n| hay.contains(n)) {
                missing_substr = true;
                break;
            }
        }
    }
    let coherence_pass = if judged {
        Some(degenerate.is_none() && !missing_substr)
    } else {
        None
    };
    if let Some(sample) = &degenerate {
        println!("    [coherence] FAIL — the answer repeats itself: {sample:?}");
        println!("    [coherence] a rate measured on this is not a rate. Treat the cell as empty.");
    } else if missing_substr {
        println!("    [coherence] FAIL — an iteration contained none of the required substrings");
    } else if coherence_pass == Some(true) {
        println!(
            "    [coherence] PASS ({} iter{})",
            all_metrics.len(),
            if needles.is_empty() {
                String::new()
            } else {
                format!(", {} substring gate", needles.len())
            }
        );
    }

    // Compute stats
    let mut stats = Vec::new();
    if let Some(load) = load_time_ms {
        if let Some(s) = Stats::compute("Model load", "ms", &[load]) {
            stats.push(s);
        }
    }
    let ttft: Vec<f64> = all_metrics.iter().filter_map(|m| m.ttft_ms).collect();
    if let Some(s) = Stats::compute("TTFT", "ms", &ttft) {
        stats.push(s);
    }
    let prompt_tps: Vec<f64> = all_metrics.iter().filter_map(|m| m.prompt_tok_s).collect();
    if let Some(s) = Stats::compute("Prompt tok/s", "tok/s", &prompt_tps) {
        stats.push(s);
    }
    let comp_tps: Vec<f64> = all_metrics
        .iter()
        .filter_map(|m| m.completion_tok_s)
        .collect();
    if let Some(s) = Stats::compute("Completion tok/s", "tok/s", &comp_tps) {
        stats.push(s);
    }
    // Length-invariant decode rate. Two servers that stop at different lengths still get
    // compared on the same quantity: milliseconds spent per token produced, which does not
    // move when one of them emits twice as many tokens.
    let decode_ms: Vec<f64> = all_metrics
        .iter()
        .filter_map(|m| m.decode_ms_per_token)
        .collect();
    if let Some(s) = Stats::compute("Decode ms/tok", "ms", &decode_ms) {
        stats.push(s);
    }
    let e2e: Vec<f64> = all_metrics.iter().map(|m| m.e2e_latency_ms).collect();
    if let Some(s) = Stats::compute("E2E latency", "ms", &e2e) {
        stats.push(s);
    }
    let tokens: Vec<f64> = all_metrics
        .iter()
        .filter_map(|m| m.tokens_generated.map(|t| t as f64))
        .collect();
    if let Some(s) = Stats::compute("Tokens generated", "count", &tokens) {
        stats.push(s);
    }
    // ITL stats: aggregate every inter-token gap from every iteration into one population
    let all_itl: Vec<f64> = all_metrics
        .iter()
        .flat_map(|m| m.inter_token_latencies_ms.iter().copied())
        .collect();
    if let Some(s) = Stats::compute("ITL (per token)", "ms", &all_itl) {
        stats.push(s);
    }
    // Energy metrics. Every one pairs a window with the token count of the SAME iteration,
    // through `per_token`, which refuses to pair at all when the two sides have drifted apart.
    //
    // The label carries WHICH domains the joules cover, because that differs by platform and
    // by permissions: RAPL is readable from userspace on Linux and nowhere else, so a Windows
    // run counts the GPU alone and its J/tok is mechanically lower than a Linux one. The two
    // are not the same quantity, and since the comparison table matches rows by label, an
    // unqualified "Energy J/tok" would let them be subtracted from each other in silence.
    let domains = all_energy
        .iter()
        .flatten()
        .next()
        .map(|w| w.domains_counted.join("+"))
        .unwrap_or_default();
    let lbl = |base: &str| {
        if domains.is_empty() {
            base.to_string()
        } else {
            format!("{base} [{domains}]")
        }
    };
    let j_per_tok = per_token(&all_metrics, &all_energy, |w, t| w.j_per_tok(t));
    if let Some(s) = Stats::compute(&lbl("Energy J/tok"), "J", &j_per_tok) {
        stats.push(s);
    }
    // Decode-phase J/tok (prefill excluded) — the cross-engine energy metric.
    let j_per_decode_tok = per_token(&all_metrics, &all_energy, |w, t| w.j_per_decode_tok(t));
    if let Some(s) = Stats::compute(&lbl("Energy decode J/tok"), "J", &j_per_decode_tok) {
        stats.push(s);
    }
    let wh_per_tok = per_token(&all_metrics, &all_energy, |w, t| w.wh_per_tok(t));
    if let Some(s) = Stats::compute(&lbl("Energy Wh/tok"), "Wh", &wh_per_tok) {
        stats.push(s);
    }
    let gco2_per_tok = per_token(&all_metrics, &all_energy, |w, t| {
        w.gco2_per_tok(t, carbon_intensity)
    });
    if let Some(s) = Stats::compute(&lbl("Carbon gCO2/tok"), "g", &gco2_per_tok) {
        stats.push(s);
    }

    let first_response_preview = all_metrics
        .iter()
        .find_map(|m| m.response_preview.clone().filter(|s| !s.is_empty()));

    // Capture model fingerprint so JSON output is cross-session attributable.
    let model_fingerprint = client.model_fingerprint(model).await;
    if let Some((mod_at, params, quant)) = model_fingerprint.as_ref() {
        println!(
            "  [{}] Model fingerprint: modified_at={} params={} quant={}",
            target.label, mod_at, params, quant
        );
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

/// Per-GPU summary across all iterations of one cell.
fn print_gpu_summary(samples_per_iter: &[Vec<GpuSample>]) {
    // Aggregate per gpu_index across iterations.
    let mut by_idx: BTreeMap<u32, (String, Vec<&GpuSample>)> = BTreeMap::new();
    for iter_samples in samples_per_iter {
        for s in iter_samples {
            let entry = by_idx
                .entry(s.gpu_index)
                .or_insert_with(|| (s.gpu_name.clone(), Vec::new()));
            entry.1.push(s);
        }
    }
    if by_idx.is_empty() {
        return;
    }
    println!("  GPU usage during this cell:");
    println!(
        "  {:<6} {:<28} {:>12} {:>12} {:>10} {:>10}",
        "GPU", "Name", "VRAM peak", "VRAM avg", "Util pk", "Power pk"
    );
    for (idx, (name, samples)) in &by_idx {
        let vram_peak = samples.iter().map(|s| s.vram_peak_mb).max().unwrap_or(0);
        let vram_avg = if !samples.is_empty() {
            samples.iter().map(|s| s.vram_avg_mb).sum::<u64>() / samples.len() as u64
        } else {
            0
        };
        let util_peak = samples.iter().map(|s| s.util_peak_pct).max().unwrap_or(0);
        let power_peak = samples
            .iter()
            .map(|s| s.power_peak_w)
            .fold(0.0_f64, f64::max);
        let short_name: String = name.chars().take(28).collect();
        println!(
            "  {:<6} {:<28} {:>10}MB {:>10}MB {:>9}% {:>8.0}W",
            idx, short_name, vram_peak, vram_avg, util_peak, power_peak
        );
    }
    println!();
}

/// Side-by-side comparison across two targets, one block per (model, num_ctx, prompt).
fn print_sweep_comparison(cells: &[CellResult], target_a: &ServerTarget, target_b: &ServerTarget) {
    let targets = [target_a, target_b];
    // Group cells by (model, num_ctx, prompt)
    let mut groups: BTreeMap<(String, usize, String), Vec<&CellResult>> = BTreeMap::new();
    for cell in cells {
        let key = (cell.model.clone(), cell.num_ctx, cell.prompt_name.clone());
        groups.entry(key).or_default().push(cell);
    }

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════════════╗");
    println!(
        "  ║ SWEEP COMPARISON ({} vs {})",
        targets[0].label, targets[1].label
    );
    println!("  ╚══════════════════════════════════════════════════════════════════════╝");

    for ((model, num_ctx, prompt), group) in &groups {
        let cell_a = group.iter().find(|c| c.target.label == targets[0].label);
        let cell_b = group.iter().find(|c| c.target.label == targets[1].label);
        if let (Some(a), Some(b)) = (cell_a, cell_b) {
            println!();
            println!(
                "  --- model={} | num_ctx={} | prompt={} ---",
                model, num_ctx, prompt
            );
            stats::print_comparison(&a.target.label, &a.stats, &b.target.label, &b.stats);
        }
    }
}

// JSON output -----------------------------------------------------------------

fn save_json(path: &str, args: &Args, cells: &[CellResult], idle_energy: Option<&EnergyWindow>) {
    let results: Vec<serde_json::Value> = cells
        .iter()
        .map(|c| {
            let stats_map: serde_json::Map<String, serde_json::Value> = c
                .stats
                .iter()
                .map(|s| {
                    (
                        s.label.clone(),
                        serde_json::json!({
                            "mean": s.mean,
                            "stddev": s.stddev,
                            "min": s.min,
                            "max": s.max,
                            "p50": s.p50,
                            "p95": s.p95,
                            "unit": s.unit,
                            "count": s.count,
                        }),
                    )
                })
                .collect();

            let iterations_json: Vec<serde_json::Value> = c
                .iterations
                .iter()
                .enumerate()
                .map(|(i, m)| {
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
                })
                .collect();

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
        })
        .collect();

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
            // Recorded because it changes what the cell measures: a session id enables
            // prefix-KV reuse on the engines that implement it, and on those alone. A
            // reader comparing two result files has to be able to see it was set.
            "session_id": args.session_id,
            "unique_prompt": args.unique_prompt,
            "num_gpu": args.num_gpu,
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
            let (itl_p50, itl_p95) =
                if let Some(s) = Stats::compute("itl", "ms", &m.inter_token_latencies_ms) {
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
                m.prompt_tok_s
                    .map(|v| format!("{:.3}", v))
                    .unwrap_or_default(),
                m.completion_tok_s
                    .map(|v| format!("{:.3}", v))
                    .unwrap_or_default(),
                m.e2e_latency_ms,
                m.tokens_generated
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
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

#[cfg(test)]
mod pairing_tests {
    use super::per_token;
    use crate::api::IterationMetrics;
    use crate::energy::{EnergyWindow, GpuEnergyPath};

    fn window(joules: f64) -> Option<EnergyWindow> {
        Some(EnergyWindow {
            gpu_energy_j: joules,
            cpu_pkg_energy_j: 0.0,
            dram_energy_j: 0.0,
            energy_j: joules,
            duration_s: 1.0,
            gpu_path: GpuEnergyPath::NvmlCounter,
            cpu_rapl_available: false,
            dram_rapl_available: false,
            domains_counted: vec!["gpu".into()],
            note: None,
            gpu_decode_j: None,
            cpu_pkg_decode_j: None,
            dram_decode_j: None,
            decode_energy_j: None,
            decode_duration_s: None,
        })
    }

    fn ran(tokens: u64) -> IterationMetrics {
        IterationMetrics {
            tokens_generated: Some(tokens),
            ..Default::default()
        }
    }

    /// Each window divides the tokens of ITS OWN iteration. A failed request used to push a
    /// window without pushing metrics, after which every later window divided a different
    /// request's token count — arithmetic that reads perfectly and is wrong.
    #[test]
    fn a_window_is_divided_by_its_own_iterations_tokens() {
        let m = vec![ran(10), ran(100)];
        let e = vec![window(20.0), window(50.0)];
        assert_eq!(per_token(&m, &e, |w, t| w.j_per_tok(t)), vec![2.0, 0.5]);
    }

    /// If the two sides ever drift, the column is withheld. A missing energy figure is a
    /// visible problem; a confidently wrong one is not.
    #[test]
    fn drift_between_the_two_sides_yields_nothing() {
        let m = vec![ran(10), ran(100)];
        let e = vec![window(20.0)];
        assert!(per_token(&m, &e, |w, t| w.j_per_tok(t)).is_empty());
    }
}

#[cfg(test)]
mod coherence_tests {
    use super::looks_degenerate;

    /// The check has to fire on what was actually observed, and stay silent on what a
    /// working model writes. A gate that cannot do both is worse than none: it either
    /// misses the garbage it was written for, or it accuses good cells and gets removed.
    #[test]
    fn it_recognises_a_repeating_answer_and_leaves_real_text_alone() {
        // gemma4:31b, verbatim from the campaign's recorded preview.
        assert!(looks_degenerate(&"--- ".repeat(40)));
        // The four that slipped through the first version, verbatim from the sweep. Each
        // has an ordinary share of distinct words and is plainly not an answer.
        assert!(looks_degenerate(
            "time-olde olde olde olde olde olde olde olde olde olde olde ol"
        ));
        assert!(looks_degenerate(
            "person, the end of a person, the end of a person, the end of a person, the end of a"
        ));
        assert!(looks_degenerate(
            "king who was able to see a time, that, that, that, that, that, that, that, that,"
        ));
        assert!(looks_degenerate(
            "time, in a kingdom far far away, to a time, in a kingdom far far away, to a time, \
             in a kingdom far far away, to a"
        ));
        assert!(looks_degenerate(&format!(
            "kingdom{}",
            "\u{FFFD}".repeat(55)
        )));
        // Single-token loops - the other shape this takes.
        assert!(looks_degenerate(&"three ".repeat(30)));
        assert!(looks_degenerate(&"the the the ".repeat(12)));

        assert!(!looks_degenerate(
            "To understand how a modern CPU pipeline works, we must first divide the \
             instruction into stages: fetch, decode, execute, memory access and write \
             back, each handled by a different part of the chip while the next \
             instruction is already entering the one before it."
        ));
        // Code repeats structure without repeating units, and must not be accused.
        assert!(!looks_degenerate(
            "fn main() { let mut total = 0; for i in 0..10 { total += i * 2; } \
             println!(\"{}\", total); let names = vec![\"ana\", \"bo\", \"cy\"]; \
             for n in names { println!(\"hello {}\", n); } }"
        ));
        // Brevity is not degeneracy - a short correct answer must pass.
        assert!(!looks_degenerate("Paris."));
        assert!(!looks_degenerate(
            "The capital of France is Paris, on the Seine."
        ));
    }
}
