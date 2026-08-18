//! Background NVML sampler — periodically reads VRAM/utilization/power for every CUDA GPU.
//!
//! Usage:
//!   let sampler = GpuSampler::start(100); // sample every 100ms
//!   // ... do work ...
//!   let summary = sampler.stop().await;   // returns `Vec<GpuSample>`, one per GPU
//!
//! NVML may be unavailable (no driver, no GPU, container without GPU passthrough).
//! In that case `start()` returns a no-op sampler and `stop()` returns an empty Vec.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Per-GPU summary computed from a stream of samples taken during one iteration.
#[derive(Debug, Clone, Serialize)]
pub struct GpuSample {
    pub gpu_index: u32,
    pub gpu_name: String,
    pub vram_peak_mb: u64,
    pub vram_avg_mb: u64,
    pub util_peak_pct: u32,
    pub util_avg_pct: u32,
    pub power_peak_w: f64,
    pub power_avg_w: f64,
    pub samples: usize,
}

/// Raw measurement for one GPU at one point in time.
#[derive(Debug, Clone, Copy)]
struct RawSample {
    vram_mb: u64,
    util_pct: u32,
    power_w: f64,
}

pub struct GpuSampler {
    stop_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<GpuSample>>>,
}

impl GpuSampler {
    /// Spawn a background task that samples every `interval_ms` milliseconds.
    /// If NVML is unavailable, returns a sampler whose `stop()` yields an empty Vec.
    pub fn start(interval_ms: u64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        // Try to initialize NVML on the spawned thread; if it fails we still return
        // a valid (but empty) sampler so callers don't have to special-case the error.
        let handle = tokio::task::spawn_blocking(move || sample_loop(stop_flag_clone, interval_ms));

        Self {
            stop_flag,
            handle: Some(handle),
        }
    }

    /// Signal the background task to stop and return the per-GPU summary.
    pub async fn stop(mut self) -> Vec<GpuSample> {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.await.unwrap_or_default()
        } else {
            Vec::new()
        }
    }
}

impl Drop for GpuSampler {
    fn drop(&mut self) {
        // Make sure the background task exits even if `stop()` was never awaited.
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}

/// Sample NVML in a tight loop until `stop_flag` is set, then return per-GPU summaries.
fn sample_loop(stop_flag: Arc<AtomicBool>, interval_ms: u64) -> Vec<GpuSample> {
    use nvml_wrapper::Nvml;

    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };
    let count = nvml.device_count().unwrap_or(0);
    if count == 0 {
        return Vec::new();
    }

    // Cache GPU names and pre-allocate sample buffers.
    let mut names: Vec<String> = Vec::with_capacity(count as usize);
    for i in 0..count {
        let name = nvml
            .device_by_index(i)
            .ok()
            .and_then(|d| d.name().ok())
            .unwrap_or_else(|| format!("GPU {}", i));
        names.push(name);
    }
    let mut buffers: Vec<Vec<RawSample>> = (0..count).map(|_| Vec::new()).collect();

    let interval = Duration::from_millis(interval_ms.max(10));
    while !stop_flag.load(Ordering::Relaxed) {
        for i in 0..count {
            let dev = match nvml.device_by_index(i) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let vram_mb = dev
                .memory_info()
                .map(|m| m.used / (1024 * 1024))
                .unwrap_or(0);
            let util_pct = dev.utilization_rates().map(|u| u.gpu).unwrap_or(0);
            // power_usage returns milliwatts
            let power_w = dev
                .power_usage()
                .map(|mw| mw as f64 / 1000.0)
                .unwrap_or(0.0);
            buffers[i as usize].push(RawSample {
                vram_mb,
                util_pct,
                power_w,
            });
        }
        std::thread::sleep(interval);
    }

    // Aggregate per-GPU
    buffers
        .into_iter()
        .enumerate()
        .map(|(i, samples)| {
            let n = samples.len();
            if n == 0 {
                return GpuSample {
                    gpu_index: i as u32,
                    gpu_name: names[i].clone(),
                    vram_peak_mb: 0,
                    vram_avg_mb: 0,
                    util_peak_pct: 0,
                    util_avg_pct: 0,
                    power_peak_w: 0.0,
                    power_avg_w: 0.0,
                    samples: 0,
                };
            }
            let vram_peak = samples.iter().map(|s| s.vram_mb).max().unwrap_or(0);
            let vram_avg = samples.iter().map(|s| s.vram_mb).sum::<u64>() / n as u64;
            let util_peak = samples.iter().map(|s| s.util_pct).max().unwrap_or(0);
            let util_avg = samples.iter().map(|s| s.util_pct).sum::<u32>() / n as u32;
            let power_peak = samples.iter().map(|s| s.power_w).fold(0.0_f64, f64::max);
            let power_avg = samples.iter().map(|s| s.power_w).sum::<f64>() / n as f64;
            GpuSample {
                gpu_index: i as u32,
                gpu_name: names[i].clone(),
                vram_peak_mb: vram_peak,
                vram_avg_mb: vram_avg,
                util_peak_pct: util_peak,
                util_avg_pct: util_avg,
                power_peak_w: power_peak,
                power_avg_w: power_avg,
                samples: n,
            }
        })
        .collect()
}
