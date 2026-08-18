//! Minimal API client for benchmarking. Speaks two protocols:
//!   * `Protocol::Ollama` — Ollama-native `/api/generate` NDJSON (Ollama, LOKEN).
//!   * `Protocol::OpenAI` — OpenAI-compatible `/v1/completions` SSE (vLLM).
//! Only includes the types and methods needed for generate + load/unload.

use futures::StreamExt;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Which wire protocol a target server speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Ollama-native `/api/generate` (Ollama + LOKEN). Raw prompt, no chat
    /// template; server reports nanosecond prefill/decode durations.
    Ollama,
    /// OpenAI-compatible `/v1/completions` (vLLM). Raw prompt completion (no
    /// chat template — matches the Ollama raw path for apples-to-apples decode
    /// comparison). The model is fixed at server launch: there is no load /
    /// unload / pull lifecycle, and `num_ctx` is set via `--max-model-len` when
    /// the server starts, so per-request `num_ctx` is ignored. The OpenAI
    /// completions API reports no prefill/decode split, so decode rate is
    /// derived from wall-clock minus TTFT — use `--stream` for fair numbers.
    OpenAI,
}

/// Ollama generate request (POST /api/generate)
#[derive(Debug, Clone, Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Base64-encoded images for vision models. Ollama's /api/generate accepts
    /// `images[]` as plain base64 strings (no `data:` prefix); LOKEN matches
    /// that shape. None for text-only benches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Disable thinking so reasoning models (gemma4, qwen3, deepcoder) emit their
    /// DIRECT answer into `response` instead of a separate thinking channel.
    /// Without this, ollama returns an empty `response` for gemma4 (its thinking
    /// output is dropped — looked "broken") and a thinking-vs-content mismatch for
    /// qwen3. `think` is ollama's field, `thinking` is LOKEN's; each engine
    /// ignores the other's, so both end up emitting content → fair comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    /// Send the prompt UNWRAPPED, which is what this bench has always said it does.
    ///
    /// `/api/generate` applies the model's chat template by default - ollama's contract,
    /// and LOKEN matches it - so without this those two answer a story opening presented
    /// as a user message while vLLM, benched on `/v1/completions`, continues it. Three
    /// engines, two different questions, on prompts written as continuations.
    ///
    /// It shows up as models that look like they stop early: qwen3-coder-next replied
    /// "It seems your message was cut off. Could you please complete the sentence?" and
    /// the cell recorded a token deficit against ollama's 128.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
}

/// Ollama generate response
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct GenerateResponse {
    pub model: String,
    #[serde(default)]
    pub response: String,
    /// Reasoning models (e.g. qwen3 with thinking mode) emit content here
    /// instead of in `response`. We count it as a token for streaming metrics.
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Explicit error field (some servers use this); we also detect in-band
    /// errors that are stuffed into `response` below.
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl GenerateResponse {
    /// Detect server-side failures that arrive as HTTP 200 with an error
    /// description in the response body.
    ///
    /// LOKEN (and occasionally Ollama) pack generation failures into the
    /// `response` field with a `"Error generating response: ..."` prefix and
    /// set `done=true` without any timing fields. Earlier bench runs silently
    /// treated these as 0-token successes; catch them here so the bench fails
    /// the cell instead of logging a zero.
    pub fn as_server_error(&self) -> Option<String> {
        if let Some(err) = &self.error {
            if !err.is_empty() {
                return Some(err.clone());
            }
        }
        // Known LOKEN in-band error format.
        if self.response.starts_with("Error generating response:")
            || self.response.starts_with("Error: ")
        {
            return Some(self.response.clone());
        }
        None
    }
}

// OpenAI-compatible /v1/completions types (vLLM) ------------------------------

/// OpenAI text-completion request (POST /v1/completions). We use the raw
/// completions endpoint (not /v1/chat/completions) so no chat template is
/// applied — matching the Ollama `/api/generate` raw path for a fair decode
/// comparison on the continuation-style bench prompts.
#[derive(Debug, Clone, Serialize)]
struct OpenAiCompletionRequest {
    model: String,
    prompt: String,
    max_tokens: usize,
    stream: bool,
    /// Greedy decode for determinism across servers.
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
}

/// Token accounting block returned by both stream and non-stream responses.
#[derive(Debug, Clone, Deserialize, Default)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

/// Non-streaming /v1/completions response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAiCompletionResponse {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct OpenAiChoice {
    #[serde(default)]
    text: String,
}

/// Streaming /v1/completions chunk (one per `data: {...}` SSE line). The final
/// chunk (when `stream_options.include_usage` is set) carries `usage` with an
/// empty `choices` array.
#[derive(Debug, Clone, Deserialize)]
struct OpenAiCompletionChunk {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

/// Metrics extracted from a single benchmark iteration
#[derive(Debug, Clone, Default)]
pub struct IterationMetrics {
    /// Wall-clock time for the entire request
    pub wall_clock_ms: f64,
    /// Model load time (ms), if reported
    pub load_time_ms: Option<f64>,
    /// Time to first token / prompt eval (ms)
    pub ttft_ms: Option<f64>,
    /// Prompt tokens processed
    pub prompt_tokens: Option<u64>,
    /// Prompt throughput (tok/s)
    pub prompt_tok_s: Option<f64>,
    /// Completion throughput (tok/s)
    pub completion_tok_s: Option<f64>,
    /// Decode time per generated token (ms). Server-reported
    /// eval_duration / eval_count. This is the length-invariant
    /// per-token decode rate — DOES NOT depend on prompt length or
    /// num_predict ceiling. Use this for fair cross-server comparison
    /// when models have different EOS behavior (chat-tuned models
    /// where Ollama generates to cap and LOKEN respects EOS).
    /// See project_bench_apples_oranges_2026_05_26.md.
    pub decode_ms_per_token: Option<f64>,
    /// End-to-end latency (ms) from server
    pub e2e_latency_ms: f64,
    /// Tokens generated (server's eval_count — includes template/control tokens).
    pub tokens_generated: Option<u64>,
    /// Non-empty content chunks received via streaming. Differs from
    /// `tokens_generated` for chat-template-leaking models (gemma4/qwen3
    /// emit `<start_of_turn>`/`<end_of_turn>`/`model:` template tokens that
    /// have empty response content but count toward server's eval_count).
    /// When tokens_generated >> content_tokens, the model is wasting cycles
    /// on template tokens — bench tok/s based on tokens_generated is
    /// CORRECT (real decode work) but the user-visible output is short.
    pub content_tokens: Option<u64>,
    /// Inter-token latencies in ms (gap between consecutive non-empty chunks).
    /// Only populated for streaming requests. Empty for non-streaming.
    pub inter_token_latencies_ms: Vec<f64>,
    /// First ~120 chars of generated text. Used for the per-cell coherence
    /// preview so a fast-but-garbage response (Z-Image black-PNG, gemma4
    /// "three three three") doesn't get logged as a WIN.
    pub response_preview: Option<String>,
    /// Client wall-clock time (ms, from request start) of the first generated
    /// token. Marks the prefill→decode boundary so the energy sampler can
    /// isolate decode-phase energy (excluding prefill). Streaming only.
    pub first_token_wall_ms: Option<f64>,
}

/// HTTP client for benchmarking
pub struct BenchClient {
    base_url: String,
    http: HttpClient,
    verbose: bool,
    protocol: Protocol,
    /// When set, injects `num_gpu` into the Ollama `options` (0 = force CPU).
    /// LOKEN ignores it; the CPU build is already CPU-only. Used for the
    /// fair CPU-vs-CPU benchmark (Ollama otherwise auto-offloads to GPU).
    num_gpu: Option<usize>,
    /// When set, injects `main_gpu` into the Ollama `options` - pins ollama to that
    /// index in its own enumeration while every card stays visible.
    ///
    /// It exists for a box whose cards are not equivalent. Ollama's scheduler is free
    /// to put a model that fits either card on the slower one, and then the two sides
    /// of the benchmark are not running on the same hardware - which is the one thing
    /// a comparison may not do. LOKEN and vLLM ignore it; they take the fastest card.
    main_gpu: Option<usize>,
}

impl BenchClient {
    pub fn with_protocol(base_url: String, verbose: bool, protocol: Protocol) -> Self {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(600)) // 10 min global timeout
            .tcp_nodelay(true) // deliver each streamed token promptly, no Nagle batching
            .build()
            .expect("Failed to create HTTP client");
        Self {
            base_url,
            http,
            verbose,
            protocol,
            num_gpu: None,
            main_gpu: None,
        }
    }

    /// Force `num_gpu` in the Ollama options (e.g. 0 for a CPU-only bench).
    pub fn set_num_gpu(&mut self, num_gpu: Option<usize>) {
        self.num_gpu = num_gpu;
    }

    /// Force `main_gpu` in the Ollama options (which physical GPU ollama uses).
    pub fn set_main_gpu(&mut self, main_gpu: Option<usize>) {
        self.main_gpu = main_gpu;
    }

    fn log_request(&self, method: &str, url: &str, body: &impl Serialize) {
        if self.verbose {
            let json = serde_json::to_string_pretty(body).unwrap_or_default();
            eprintln!("    --> {} {}", method, url);
            eprintln!("    --> {}", json);
        }
    }

    fn log_response_status(&self, url: &str, status: reqwest::StatusCode) {
        if self.verbose {
            eprintln!("    <-- {} {}", status, url);
        }
    }

    fn log_response_body(&self, body: &impl std::fmt::Debug) {
        if self.verbose {
            eprintln!("    <-- {:?}", body);
        }
    }

    /// Check if a model exists on the server.
    ///   * Ollama: POST `/api/show`.
    ///   * OpenAI/vLLM: GET `/v1/models` and look for the id (vLLM serves the
    ///     one model it was launched with; an exact id mismatch is tolerated as
    ///     long as the endpoint is reachable and serving something, since users
    ///     often launch with a HF repo id that differs from the bench alias).
    pub async fn model_exists(&self, model: &str) -> bool {
        if self.protocol == Protocol::OpenAI {
            let url = format!("{}/v1/models", self.base_url);
            let resp = match self.http.get(&url).send().await {
                Ok(r) if r.status().is_success() => r,
                _ => return false,
            };
            let v: serde_json::Value = match resp.json().await {
                Ok(v) => v,
                Err(_) => return false,
            };
            let ids: Vec<&str> = v
                .get("data")
                .and_then(|d| d.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
                        .collect()
                })
                .unwrap_or_default();
            // Exact match preferred; otherwise accept any served model so the
            // bench alias need not equal the HF repo id.
            return ids.iter().any(|id| *id == model) || !ids.is_empty();
        }
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({ "model": model });
        match self.http.post(&url).json(&body).send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    /// Capture model fingerprint via /api/show: returns (modified_at,
    /// parameter_size, quantization_level). Used to tag bench output JSON
    /// so cross-session comparisons can verify model identity (Ollama
    /// auto-pulls update the underlying GGUF mid-bench-history, causing
    /// phantom regressions — see project_deepcoder_bisect_2026_05_25).
    /// Returns None if the model doesn't exist or /api/show fails.
    pub async fn model_fingerprint(&self, model: &str) -> Option<(String, String, String)> {
        // vLLM has no /api/show equivalent; identity is fixed at server launch.
        if self.protocol == Protocol::OpenAI {
            return None;
        }
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({ "model": model });
        let resp = self.http.post(&url).json(&body).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: serde_json::Value = resp.json().await.ok()?;
        let modified = v
            .get("modified_at")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let params = v
            .pointer("/details/parameter_size")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let quant = v
            .pointer("/details/quantization_level")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if modified.is_empty() && params.is_empty() {
            return None;
        }
        Some((modified, params, quant))
    }

    /// Pull a model from the registry (POST /api/pull), streaming progress to stdout.
    /// Returns Ok when the pull completes successfully.
    pub async fn pull_model(&self, model: &str) -> Result<(), String> {
        let url = format!("{}/api/pull", self.base_url);
        let body = serde_json::json!({ "name": model, "stream": true });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(3600)) // 1 hour for large models
            .send()
            .await
            .map_err(|e| format!("Pull request failed: {}", e))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Pull failed: {}", text));
        }

        // Read NDJSON progress stream
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut last_status = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Pull stream error: {}", e))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();
                if line.is_empty() {
                    continue;
                }
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
                    let status = obj.get("status").and_then(|s| s.as_str()).unwrap_or("");
                    if status != last_status {
                        if !last_status.is_empty() {
                            println!();
                        }
                        print!("    {}", status);
                        last_status = status.to_string();
                    }
                    // Show download progress
                    if let (Some(completed), Some(total)) = (
                        obj.get("completed").and_then(serde_json::Value::as_u64),
                        obj.get("total").and_then(serde_json::Value::as_u64),
                    ) {
                        if total > 0 {
                            let pct = (completed as f64 / total as f64) * 100.0;
                            print!(" {:.0}%", pct);
                        }
                    }
                }
            }
        }
        println!();
        Ok(())
    }

    /// Ensure a model is available — pull it if not found.
    pub async fn ensure_model(&self, model: &str) -> Result<(), String> {
        if self.model_exists(model).await {
            return Ok(());
        }
        // vLLM serves exactly the model it was launched with; the bench can't
        // pull or swap it. A miss means the server is down or serving a
        // different model — surface that rather than attempting an /api/pull.
        if self.protocol == Protocol::OpenAI {
            return Err(format!(
                "vLLM target at {} is not serving '{}' (start it with `vllm serve {}`)",
                self.base_url, model, model
            ));
        }
        println!("  Model '{}' not found, pulling...", model);
        self.pull_model(model).await
    }

    /// Query /api/ps for the list of models currently loaded on this server.
    /// Returns an empty list if the endpoint is unreachable or returns a
    /// non-JSON body — never an error, since this is best-effort cleanup.
    pub async fn loaded_models(&self) -> Vec<String> {
        let url = format!("{}/api/ps", self.base_url);
        let resp = match self.http.get(&url).send().await {
            Ok(r) if r.status().is_success() => r,
            _ => return Vec::new(),
        };
        let body: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        body.get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Unload any loaded model by sending empty prompt with keep_alive: 0.
    /// No-op for vLLM: the model is resident for the server's lifetime and
    /// there is no unload API.
    pub async fn unload_model(&self, model: &str) -> Result<(), String> {
        if self.protocol == Protocol::OpenAI {
            return Ok(());
        }
        let url = format!("{}/api/generate", self.base_url);
        // Send keep_alive as a JSON-Value-typed body so we can use the integer
        // form. Ollama accepts both string and integer; some Ollama versions
        // ignore the string form silently and never actually unload, leaving
        // the runner subprocess holding VRAM. The integer form is universally
        // honoured. LOKEN now also accepts both forms after the
        // deserialize_keep_alive helper landed.
        let body = serde_json::json!({
            "model": model,
            "prompt": "",
            "stream": false,
            "keep_alive": 0,
        });
        self.log_request("POST", &url, &body);
        let resp = self.http.post(&url).json(&body).send().await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(r) => {
                let status = r.status();
                let text = r.text().await.unwrap_or_default();
                // 404 is fine — model wasn't loaded
                if status.as_u16() == 404 {
                    Ok(())
                } else {
                    Err(format!("Unload returned {}: {}", status, text))
                }
            }
            Err(e) => Err(format!("Unload request failed: {}", e)),
        }
    }

    /// Load a model and return load time metrics. For vLLM the model is
    /// already resident (loaded at server launch), so this just confirms the
    /// server is reachable and reports no load time.
    pub async fn load_model(&self, model: &str) -> Result<(f64, Option<f64>), String> {
        if self.protocol == Protocol::OpenAI {
            return if self.model_exists(model).await {
                Ok((0.0, None))
            } else {
                Err(format!(
                    "vLLM target at {} not reachable / not serving '{}'",
                    self.base_url, model
                ))
            };
        }
        let url = format!("{}/api/generate", self.base_url);
        let req = GenerateRequest {
            model: model.to_string(),
            prompt: String::new(),
            stream: false,
            options: None,
            keep_alive: Some("30m".to_string()),
            images: None,
            think: None,
            thinking: None,
            // The load probe sends an empty prompt; there is nothing to wrap either way.
            raw: None,
        };
        self.log_request("POST", &url, &req);

        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .json(&req)
            .timeout(Duration::from_secs(600))
            .send()
            .await
            .map_err(|e| format!("Load request failed: {}", e))?;

        self.log_response_status(&url, resp.status());
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            if self.verbose {
                eprintln!("    <-- {}", text);
            }
            return Err(format!("Load failed: {}", text));
        }

        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let body: GenerateResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;
        self.log_response_body(&body);
        let server_load_ms = body.load_duration.map(|ns| ns as f64 / 1_000_000.0);

        Ok((wall_ms, server_load_ms))
    }

    /// Build the `options` JSON for an Ollama request, including optional num_ctx.
    fn build_options(
        max_tokens: usize,
        num_ctx: Option<usize>,
        session_id: Option<&str>,
        num_gpu: Option<usize>,
        main_gpu: Option<usize>,
    ) -> serde_json::Value {
        let mut opts = serde_json::Map::new();
        opts.insert("num_predict".into(), serde_json::Value::from(max_tokens));
        // Greedy decode for benchmarking: deterministic, reproducible, and it
        // measures pure decode speed. Without an explicit temperature each
        // engine falls back to its own default (ollama 0.8, LOKEN's config
        // default) and runs the full sampling path — softmax + top-k/top-p +
        // multinomial — which is ~1.7 ms/token slower than argmax and varies by
        // engine, confounding the decode-rate comparison. Temperature 0 = argmax
        // on every engine (ollama, LOKEN, vLLM).
        opts.insert("temperature".into(), serde_json::Value::from(0.0));
        if let Some(ctx) = num_ctx {
            opts.insert("num_ctx".into(), serde_json::Value::from(ctx));
        }
        if let Some(ng) = num_gpu {
            opts.insert("num_gpu".into(), serde_json::Value::from(ng));
        }
        if let Some(mg) = main_gpu {
            opts.insert("main_gpu".into(), serde_json::Value::from(mg));
        }
        if let Some(sid) = session_id {
            // LOKEN uses session_id for prefix-KV reuse (vision and text).
            // Ollama ignores unknown options, so this is safe to pass to both.
            opts.insert("session_id".into(), serde_json::Value::from(sid));
        }
        serde_json::Value::Object(opts)
    }

    /// Run a single generate request and return metrics. `images` is an optional
    /// slice of base64-encoded JPEGs/PNGs (no `data:` prefix) — vision models
    /// only. For text-only models pass `None`.
    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        num_ctx: Option<usize>,
        images: Option<&[String]>,
        session_id: Option<&str>,
    ) -> Result<IterationMetrics, String> {
        if self.protocol == Protocol::OpenAI {
            // num_ctx (fixed at server launch) and session_id (no prefix-KV
            // reuse API) don't apply to the OpenAI completions path.
            return self
                .generate_openai(model, prompt, max_tokens, images)
                .await;
        }
        let url = format!("{}/api/generate", self.base_url);
        let req = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
            options: Some(Self::build_options(
                max_tokens,
                num_ctx,
                session_id,
                self.num_gpu,
                self.main_gpu,
            )),
            keep_alive: Some("30m".to_string()),
            images: images.map(<[String]>::to_vec),
            think: Some(false),
            thinking: Some(false),
            // Text prompts are continuations and must reach every engine unwrapped, the
            // way vLLM's /v1/completions receives them. Vision prompts are instructions
            // about an image and keep their template - stripping it there would compare
            // a different task, not a fairer one.
            raw: images.is_none().then_some(true),
        };
        self.log_request("POST", &url, &req);

        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .json(&req)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| format!("Generate request failed: {}", e))?;

        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

        self.log_response_status(&url, resp.status());
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            if self.verbose {
                eprintln!("    <-- {}", text);
            }
            return Err(format!("Generate failed: {}", text));
        }

        let body: GenerateResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;
        self.log_response_body(&body);

        // Surface in-band server errors (HTTP 200 with an error string in `response`).
        if let Some(err) = body.as_server_error() {
            return Err(format!("server error: {}", err));
        }
        // Completed response with zero generated tokens is also a failure — the server
        // said "done" without emitting anything. Empty prompt requests (used for load/unload)
        // hit a different code path and don't go through this function.
        if body.done && body.eval_count.unwrap_or(0) == 0 && body.response.is_empty() {
            return Err("server returned done=true with 0 tokens generated".to_string());
        }

        // Extract metrics from response (all durations in nanoseconds)
        let ttft_ms = body.prompt_eval_duration.map(|ns| ns as f64 / 1_000_000.0);
        let prompt_tokens = body.prompt_eval_count;
        let prompt_tok_s = match (body.prompt_eval_count, body.prompt_eval_duration) {
            (Some(count), Some(dur)) if dur > 0 => Some(count as f64 / (dur as f64 / 1e9)),
            _ => None,
        };
        let completion_tok_s = match (body.eval_count, body.eval_duration) {
            (Some(count), Some(dur)) if dur > 0 => Some(count as f64 / (dur as f64 / 1e9)),
            _ => None,
        };
        let decode_ms_per_token = match (body.eval_count, body.eval_duration) {
            (Some(count), Some(dur)) if count > 0 => Some((dur as f64 / 1e6) / count as f64),
            _ => None,
        };
        let e2e_latency_ms = body
            .total_duration
            .map(|ns| ns as f64 / 1_000_000.0)
            .unwrap_or(wall_ms);
        let load_time_ms = body.load_duration.map(|ns| ns as f64 / 1_000_000.0);

        Ok(IterationMetrics {
            wall_clock_ms: wall_ms,
            load_time_ms,
            ttft_ms,
            prompt_tokens,
            prompt_tok_s,
            completion_tok_s,
            decode_ms_per_token,
            e2e_latency_ms,
            tokens_generated: body.eval_count,
            content_tokens: None, // non-streaming has no chunk count
            inter_token_latencies_ms: Vec::new(),
            response_preview: Some(truncate_preview(&body.response)),
            first_token_wall_ms: None, // non-streaming has no per-token timing
        })
    }

    /// Run a streaming generate request and return metrics.
    /// TTFT is measured as wall-clock time to receive the first chunk.
    /// The final chunk contains the full timing fields.
    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        num_ctx: Option<usize>,
        images: Option<&[String]>,
        session_id: Option<&str>,
    ) -> Result<IterationMetrics, String> {
        if self.protocol == Protocol::OpenAI {
            return self
                .generate_stream_openai(model, prompt, max_tokens, images)
                .await;
        }
        let url = format!("{}/api/generate", self.base_url);
        let req = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: true,
            options: Some(Self::build_options(
                max_tokens,
                num_ctx,
                session_id,
                self.num_gpu,
                self.main_gpu,
            )),
            keep_alive: Some("30m".to_string()),
            images: images.map(<[String]>::to_vec),
            think: Some(false),
            thinking: Some(false),
            // Text prompts are continuations and must reach every engine unwrapped, the
            // way vLLM's /v1/completions receives them. Vision prompts are instructions
            // about an image and keep their template - stripping it there would compare
            // a different task, not a fairer one.
            raw: images.is_none().then_some(true),
        };
        self.log_request("POST", &url, &req);

        // Stream over a raw TcpStream line reader: it reliably captures the final
        // done line (and thus the server's eval_count/eval_duration, which drives
        // the decode rate) without depending on hyper's chunk-boundary handling,
        // and reads the per-token timestamps used for the ITL diagnostics.
        let host_port = self
            .base_url
            .trim_end_matches('/')
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .to_string();
        let body_bytes = serde_json::to_vec(&req).map_err(|e| format!("serialize: {}", e))?;
        let max_tok = max_tokens;
        let (first_chunk_time, last_done_line, token_count, token_times_ms, text_acc, wall_ms) =
            tokio::task::spawn_blocking(move || -> Result<(Option<f64>, Option<String>, u64, Vec<f64>, String, f64), String> {
                use std::io::{BufRead, BufReader, Write};
                use std::net::TcpStream;
                let start = Instant::now();
                let mut sock = TcpStream::connect(&host_port)
                    .map_err(|e| format!("connect {host_port}: {e}"))?;
                sock.set_nodelay(true).ok();
                // Minimal HTTP/1.1 request. `Connection: close` keeps the framing
                // simple; we stop reading at the done line regardless.
                let head = format!(
                    "POST /api/generate HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body_bytes.len()
                );
                sock.write_all(head.as_bytes()).map_err(|e| format!("write head: {e}"))?;
                sock.write_all(&body_bytes).map_err(|e| format!("write body: {e}"))?;
                sock.flush().ok();

                let mut reader = BufReader::new(sock);
                // Skip HTTP response headers (up to the blank line); capture status.
                let mut status_ok = None;
                let mut hl = String::new();
                loop {
                    hl.clear();
                    let n = reader.read_line(&mut hl).map_err(|e| format!("read headers: {e}"))?;
                    if n == 0 {
                        return Err("connection closed during headers".to_string());
                    }
                    if status_ok.is_none() {
                        // "HTTP/1.1 200 OK"
                        status_ok = Some(hl.split(' ').nth(1) == Some("200"));
                    }
                    if hl.trim_end().is_empty() {
                        break;
                    }
                }
                if status_ok != Some(true) {
                    return Err(format!("stream HTTP status not 200 ({})", status_ok.map_or("?", |_| "non-200")));
                }

                let mut first_chunk_time: Option<f64> = None;
                let mut last_done_line: Option<String> = None;
                let mut token_count: u64 = 0;
                let mut token_times_ms: Vec<f64> = Vec::with_capacity(max_tok);
                let mut text_acc = String::with_capacity(256);
                let mut line = String::new();
                loop {
                    line.clear();
                    let n = reader.read_line(&mut line).map_err(|e| format!("Stream read error: {e}"))?;
                    if n == 0 {
                        break; // EOF (Connection: close)
                    }
                    let l = line.trim();
                    // Skip HTTP chunked-transfer framing lines (hex size, CRLF) and
                    // any blank lines: only NDJSON objects start with '{'.
                    if !l.starts_with('{') {
                        continue;
                    }
                    let now = start.elapsed().as_secs_f64() * 1000.0;
                    if first_chunk_time.is_none() {
                        first_chunk_time = Some(now);
                    }
                    // Content first: a chunk carrying response/thinking text is a
                    // token even if that text contains `"done":true`. The terminal
                    // stats line always has an EMPTY response → done branch.
                    if has_nonempty_field(l, "\"response\":\"")
                        || has_nonempty_field(l, "\"thinking\":\"")
                    {
                        token_count += 1;
                        token_times_ms.push(now);
                        if text_acc.len() < 256 {
                            if let Ok(r) = serde_json::from_str::<GenerateResponse>(l) {
                                if let Some(err) = r.as_server_error() {
                                    return Err(format!("server error: {}", err));
                                } else if !r.response.is_empty() {
                                    text_acc.push_str(&r.response);
                                } else if let Some(t) = r.thinking.as_deref() {
                                    text_acc.push_str(t);
                                }
                            }
                        }
                    } else if l.contains("\"done\":true") || l.contains("\"done\": true") {
                        last_done_line = Some(l.to_string());
                        break; // generation complete — don't wait on EOF / keep-alive
                    }
                }
                let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                Ok((first_chunk_time, last_done_line, token_count, token_times_ms, text_acc, wall_ms))
            })
            .await
            .map_err(|e| format!("stream task panicked: {}", e))??;

        // Final done line carries the server's prompt/eval stats: prefill
        // throughput, TTFT, and the authoritative eval_count/eval_duration used
        // for the decode rate below.
        let last_response: Option<GenerateResponse> = last_done_line
            .as_deref()
            .and_then(|l| serde_json::from_str(l).ok());
        if let Some(ref body) = last_response {
            if let Some(err) = body.as_server_error() {
                return Err(format!("server error: {}", err));
            }
            self.log_response_body(body);
        }
        if token_count == 0 {
            return Err("streaming request returned 0 tokens (server silently failed)".to_string());
        }

        let inter_token_latencies_ms: Vec<f64> =
            token_times_ms.windows(2).map(|w| w[1] - w[0]).collect();

        // UNIFORM decode rate across ALL engines: the client-observed first-to-
        // last generated-token span (steady_decode_tok_s). vLLM's /v1 API exposes
        // no server prefill/decode split, so reading the server eval_count/
        // eval_duration here would measure Ollama/loken by a DIFFERENT
        // criterion than vLLM and bias the cross-engine comparison. Every engine
        // is therefore scored by the same client-side rate. The server eval
        // fields are kept only as a fallback when the client span is unusable
        // (<2 tokens or zero span).
        let completion_tok_s = steady_decode_tok_s(&token_times_ms).or_else(|| {
            last_response
                .as_ref()
                .and_then(|b| match (b.eval_count, b.eval_duration) {
                    (Some(c), Some(d)) if d > 0 => Some(c as f64 / (d as f64 / 1e9)),
                    _ => None,
                })
        });
        let decode_ms_per_token = completion_tok_s.map(|t| 1000.0 / t);

        let ttft_ms = last_response
            .as_ref()
            .and_then(|b| b.prompt_eval_duration.map(|ns| ns as f64 / 1e6))
            .or(first_chunk_time);
        let prompt_tokens = last_response.as_ref().and_then(|b| b.prompt_eval_count);
        let prompt_tok_s = last_response.as_ref().and_then(|b| {
            match (b.prompt_eval_count, b.prompt_eval_duration) {
                (Some(count), Some(dur)) if dur > 0 => Some(count as f64 / (dur as f64 / 1e9)),
                _ => None,
            }
        });
        let eval_count = last_response
            .as_ref()
            .and_then(|b| b.eval_count)
            .unwrap_or(token_count);
        let e2e_latency_ms = last_response
            .as_ref()
            .and_then(|b| b.total_duration.map(|ns| ns as f64 / 1e6))
            .unwrap_or(wall_ms);
        let load_time_ms = last_response
            .as_ref()
            .and_then(|b| b.load_duration.map(|ns| ns as f64 / 1e6));

        Ok(IterationMetrics {
            wall_clock_ms: wall_ms,
            load_time_ms,
            ttft_ms,
            prompt_tokens,
            prompt_tok_s,
            completion_tok_s,
            decode_ms_per_token,
            e2e_latency_ms,
            tokens_generated: Some(eval_count),
            content_tokens: Some(token_count),
            inter_token_latencies_ms,
            response_preview: Some(truncate_preview(&text_acc)),
            first_token_wall_ms: token_times_ms.first().copied(),
        })
    }

    // OpenAI /v1/completions path (vLLM) --------------------------------------

    /// Non-streaming OpenAI completion. The completions API reports token
    /// counts (`usage`) but no prefill/decode timing split, so decode rate is
    /// derived from wall-clock (which still includes prefill — prefer
    /// `generate_stream_openai` / `--stream` for a fair decode number).
    async fn generate_openai(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        images: Option<&[String]>,
    ) -> Result<IterationMetrics, String> {
        if images.is_some() {
            return Err("vLLM vision benchmarking via --vllm is not supported \
                        (the bench uses the raw /v1/completions text path)"
                .to_string());
        }
        let url = format!("{}/v1/completions", self.base_url);
        let req = OpenAiCompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            stream: false,
            temperature: 0.0,
            stream_options: None,
        };
        self.log_request("POST", &url, &req);

        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .json(&req)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| format!("Completion request failed: {}", e))?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

        self.log_response_status(&url, resp.status());
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            if self.verbose {
                eprintln!("    <-- {}", text);
            }
            return Err(format!("Completion failed: {}", text));
        }
        let body: OpenAiCompletionResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;
        self.log_response_body(&body);

        let text = body
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();
        let usage = body.usage.unwrap_or_default();
        let completion_tokens = usage.completion_tokens.unwrap_or(0);
        if completion_tokens == 0 && text.is_empty() {
            return Err("vLLM returned 0 completion tokens".to_string());
        }
        // No server timing split: wall-clock includes prefill. Decode rate here
        // is an upper bound on latency; streaming gives the clean number.
        let completion_tok_s = if wall_ms > 0.0 && completion_tokens > 0 {
            Some(completion_tokens as f64 / (wall_ms / 1000.0))
        } else {
            None
        };

        Ok(IterationMetrics {
            wall_clock_ms: wall_ms,
            load_time_ms: None,
            ttft_ms: None,
            prompt_tokens: usage.prompt_tokens,
            prompt_tok_s: None,
            completion_tok_s,
            decode_ms_per_token: None,
            e2e_latency_ms: wall_ms,
            tokens_generated: Some(completion_tokens),
            content_tokens: None,
            inter_token_latencies_ms: Vec::new(),
            response_preview: Some(truncate_preview(&text)),
            first_token_wall_ms: None, // non-streaming has no per-token timing
        })
    }

    /// Streaming OpenAI completion (SSE). Measures real TTFT (time to first
    /// content chunk) and derives the decode rate from generation-only
    /// wall-clock (wall − TTFT), mirroring the Ollama streaming fallback so the
    /// two protocols are compared on the same length-invariant basis. Token
    /// counts come from the final `usage` chunk (`stream_options.include_usage`).
    async fn generate_stream_openai(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        images: Option<&[String]>,
    ) -> Result<IterationMetrics, String> {
        if images.is_some() {
            return Err("vLLM vision benchmarking via --vllm is not supported \
                        (the bench uses the raw /v1/completions text path)"
                .to_string());
        }
        let url = format!("{}/v1/completions", self.base_url);
        let req = OpenAiCompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            stream: true,
            temperature: 0.0,
            stream_options: Some(serde_json::json!({ "include_usage": true })),
        };
        self.log_request("POST", &url, &req);

        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .json(&req)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| format!("Stream request failed: {}", e))?;

        self.log_response_status(&url, resp.status());
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            if self.verbose {
                eprintln!("    <-- {}", text);
            }
            return Err(format!("Stream failed: {}", text));
        }

        let mut stream = resp.bytes_stream();
        let mut first_chunk_time: Option<f64> = None;
        let mut token_count: u64 = 0;
        let mut usage: Option<OpenAiUsage> = None;
        let mut buffer = String::new();
        let mut token_times_ms: Vec<f64> = Vec::with_capacity(max_tokens);
        let mut text_acc = String::with_capacity(256);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Stream read error: {}", e))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // SSE frames are separated by newlines; each data line is
            // `data: {json}` or the terminal `data: [DONE]`.
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();
                let payload = match line.strip_prefix("data:") {
                    Some(p) => p.trim(),
                    None => continue, // blank lines / comments
                };
                if payload.is_empty() || payload == "[DONE]" {
                    continue;
                }
                let parsed: OpenAiCompletionChunk = match serde_json::from_str(payload) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if let Some(u) = parsed.usage {
                    usage = Some(u); // final usage chunk
                }
                let delta = parsed
                    .choices
                    .first()
                    .map(|c| c.text.as_str())
                    .unwrap_or("");
                if !delta.is_empty() {
                    if first_chunk_time.is_none() {
                        first_chunk_time = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    token_count += 1;
                    token_times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                    if text_acc.len() < 256 {
                        text_acc.push_str(delta);
                    }
                }
            }
        }

        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let inter_token_latencies_ms: Vec<f64> =
            token_times_ms.windows(2).map(|w| w[1] - w[0]).collect();

        let completion_tokens = usage
            .as_ref()
            .and_then(|u| u.completion_tokens)
            .unwrap_or(token_count);
        if completion_tokens == 0 {
            return Err("vLLM streaming returned 0 tokens".to_string());
        }
        // UNIFORM decode rate — identical definition to the Ollama-protocol path
        // (first-to-last generated token, client-side), so vLLM is measured by
        // exactly the same criterion as Ollama and LOKEN.
        let completion_tok_s = steady_decode_tok_s(&token_times_ms);
        let decode_ms_per_token = completion_tok_s.map(|t| 1000.0 / t);

        Ok(IterationMetrics {
            wall_clock_ms: wall_ms,
            load_time_ms: None,
            ttft_ms: first_chunk_time,
            prompt_tokens: usage.as_ref().and_then(|u| u.prompt_tokens),
            prompt_tok_s: None,
            completion_tok_s,
            decode_ms_per_token,
            e2e_latency_ms: wall_ms,
            tokens_generated: Some(completion_tokens),
            content_tokens: Some(token_count),
            inter_token_latencies_ms,
            response_preview: Some(truncate_preview(&text_acc)),
            first_token_wall_ms: token_times_ms.first().copied(),
        })
    }
}

/// Take the first ~120 chars of a model's output and squash whitespace.
/// Used only for the per-cell coherence preview — never affects metrics.
fn truncate_preview(s: &str) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed: String = collapsed.chars().take(120).collect();
    if collapsed.chars().count() > 120 {
        format!("{trimmed}…")
    } else {
        trimmed
    }
}

/// Uniform decode-rate criterion used for EVERY engine (Ollama, LOKEN,
/// vLLM): tokens-after-the-first divided by the client-observed wall time from
/// the first to the last generated token. It deliberately ignores any
/// server-reported timing field so the comparison is identical across engines
/// ("mêmes critères pour tous"), and it excludes prefill/TTFT by starting the
/// clock at the first token. Accuracy depends on the read loop not throttling
/// the stream — hence the lean reader in `generate_stream`.
fn steady_decode_tok_s(token_times_ms: &[f64]) -> Option<f64> {
    if token_times_ms.len() < 2 {
        return None;
    }
    let span_ms = token_times_ms[token_times_ms.len() - 1] - token_times_ms[0];
    if span_ms <= 0.0 {
        return None;
    }
    Some((token_times_ms.len() as f64 - 1.0) / (span_ms / 1000.0))
}

/// Cheap test for a non-empty JSON string field, e.g. `field_pat = "\"response\":\""`.
/// Avoids a full per-token deserialize in the streaming hot loop (a heavy
/// per-token parse would itself set the read cadence and under-report fast
/// engines). `field_pat` must be a literal so there is no per-call allocation.
fn has_nonempty_field(line: &str, field_pat: &str) -> bool {
    line.find(field_pat)
        .is_some_and(|i| line.as_bytes().get(i + field_pat.len()) != Some(&b'"'))
}
