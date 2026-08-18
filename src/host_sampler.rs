//! Background host sampler — periodically reads RSS/swap/CPU of the engine
//! processes (matched by /proc/<pid>/comm) so each bench reports the host
//! footprint next to the GPU one. Mirrors `gpu_sampler`'s start/stop shape.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Host footprint summary for the engine process set during one run.
#[derive(Debug, Clone, Serialize)]
pub struct HostSample {
    /// comm names matched (e.g. "server", "ollama")
    pub procs: Vec<String>,
    pub rss_peak_mb: u64,
    pub rss_avg_mb: u64,
    pub swap_peak_mb: u64,
    /// whole-process-set CPU usage (100% = one core)
    pub cpu_avg_pct: u32,
    pub cpu_peak_pct: u32,
    pub samples: usize,
}

pub struct HostSampler {
    stop_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<Option<HostSample>>>,
}

impl HostSampler {
    /// Sample every `interval_ms` ms the processes whose /proc comm is in `names`.
    pub fn start(interval_ms: u64, names: &[&str]) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let flag = stop_flag.clone();
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        let handle = tokio::task::spawn_blocking(move || sample_loop(flag, interval_ms, names));
        Self {
            stop_flag,
            handle: Some(handle),
        }
    }

    pub async fn stop(mut self) -> Option<HostSample> {
        self.stop_flag.store(true, Ordering::Relaxed);
        match self.handle.take() {
            Some(h) => h.await.unwrap_or(None),
            None => None,
        }
    }
}

impl Drop for HostSampler {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}

/// (rss_kb + swap_kb, swap_kb, cpu_jiffies) summed over the matching pids.
fn snapshot(names: &[String]) -> (u64, u64, u64) {
    let (mut rss, mut swap, mut jiffies) = (0u64, 0u64, 0u64);
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return (0, 0, 0);
    };
    for e in dir.flatten() {
        let p = e.path();
        let Some(pid) = p.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if !pid.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(comm) = std::fs::read_to_string(p.join("comm")) else {
            continue;
        };
        if !names.iter().any(|n| comm.trim() == n) {
            continue;
        }
        if let Ok(st) = std::fs::read_to_string(p.join("status")) {
            for l in st.lines() {
                if let Some(v) = l.strip_prefix("VmRSS:") {
                    rss += v
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse::<u64>()
                        .unwrap_or(0);
                } else if let Some(v) = l.strip_prefix("VmSwap:") {
                    swap += v
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse::<u64>()
                        .unwrap_or(0);
                }
            }
        }
        if let Ok(stat) = std::fs::read_to_string(p.join("stat")) {
            // fields 14+15 (utime+stime), after the parenthesized comm
            if let Some(rest) = stat.rsplit(')').next() {
                let f: Vec<&str> = rest.split_whitespace().collect();
                if f.len() > 12 {
                    jiffies +=
                        f[11].parse::<u64>().unwrap_or(0) + f[12].parse::<u64>().unwrap_or(0);
                }
            }
        }
    }
    (rss + swap, swap, jiffies)
}

fn sample_loop(stop: Arc<AtomicBool>, interval_ms: u64, names: Vec<String>) -> Option<HostSample> {
    let interval = Duration::from_millis(interval_ms.max(50));
    let clk = 100.0f64; // USER_HZ on Linux
    let mut rss_v: Vec<u64> = Vec::new();
    let mut swap_peak = 0u64;
    let mut cpu_v: Vec<f64> = Vec::new();
    let (_, _, mut prev_j) = snapshot(&names);
    let mut prev_t = std::time::Instant::now();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(interval);
        let (rss_kb, swap_kb, j) = snapshot(&names);
        let dt = prev_t.elapsed().as_secs_f64();
        prev_t = std::time::Instant::now();
        if rss_kb > 0 {
            rss_v.push(rss_kb / 1024);
            swap_peak = swap_peak.max(swap_kb / 1024);
            if dt > 0.0 && j >= prev_j {
                cpu_v.push((j - prev_j) as f64 / clk / dt * 100.0);
            }
        }
        prev_j = j;
    }
    if rss_v.is_empty() {
        return None;
    }
    let n = rss_v.len();
    Some(HostSample {
        procs: names,
        rss_peak_mb: rss_v.iter().copied().max().unwrap_or(0),
        rss_avg_mb: rss_v.iter().sum::<u64>() / n as u64,
        swap_peak_mb: swap_peak,
        cpu_avg_pct: (cpu_v.iter().sum::<f64>() / cpu_v.len().max(1) as f64) as u32,
        cpu_peak_pct: cpu_v.iter().fold(0.0f64, |a, &b| a.max(b)) as u32,
        samples: n,
    })
}
