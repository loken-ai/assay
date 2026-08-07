//! Statistics computation for benchmark results.

/// Descriptive statistics for a series of measurements.
#[derive(Debug, Clone)]
pub struct Stats {
    pub label: String,
    pub unit: String,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub count: usize,
}

impl Stats {
    /// Compute stats from a slice of values. Returns None if empty.
    pub fn compute(label: &str, unit: &str, values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);

        Some(Stats {
            label: label.to_string(),
            unit: unit.to_string(),
            mean,
            stddev,
            min,
            max,
            p50,
            p95,
            count: n,
        })
    }

    /// Format value with unit for display
    pub fn fmt_val(&self, val: f64) -> String {
        if self.unit == "ms" {
            if val >= 1000.0 {
                format!("{:.1}s", val / 1000.0)
            } else {
                format!("{:.0}ms", val)
            }
        } else if self.unit == "tok/s" {
            format!("{:.1}", val)
        } else if self.unit == "J" {
            // Energy per token is sub-joule to a few J — keep 3 decimals.
            format!("{:.3}", val)
        } else if self.unit == "Wh" || self.unit == "g" {
            // Wh/tok and gCO2/tok are tiny (1e-4..1e-6) — scientific notation.
            format!("{:.3e}", val)
        } else {
            format!("{:.0}", val)
        }
    }

    /// Format mean +/- stddev
    pub fn fmt_mean_std(&self) -> String {
        format!("{} +/-{}", self.fmt_val(self.mean), self.fmt_val(self.stddev))
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Print a stats table for a single server run
pub fn print_table(server_url: &str, model: &str, prompt_len: usize, max_tokens: usize, iterations: usize, stats: &[Stats]) {
    let width = 72;
    let border = "=".repeat(width);
    let thin = "-".repeat(width);

    println!();
    println!("  {}", border);
    println!("  LLM Benchmark Results");
    println!("  {}", thin);
    println!("  Server:     {}", server_url);
    println!("  Model:      {}", model);
    println!("  Prompt:     ~{} chars, max {} tokens, {} iterations", prompt_len, max_tokens, iterations);
    println!("  {}", thin);
    println!(
        "  {:<22} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Metric", "Mean", "StdDev", "Min", "Max", "P95"
    );
    println!("  {}", thin);

    for s in stats {
        println!(
            "  {:<22} {:>10} {:>10} {:>10} {:>10} {:>10}",
            s.label,
            s.fmt_val(s.mean),
            format!("+/-{}", s.fmt_val(s.stddev)),
            s.fmt_val(s.min),
            s.fmt_val(s.max),
            s.fmt_val(s.p95),
        );
    }
    println!("  {}", border);
    println!();
}

/// Print a comparison table between two server runs
pub fn print_comparison(
    label_a: &str,
    stats_a: &[Stats],
    label_b: &str,
    stats_b: &[Stats],
) {
    let width = 80;
    let border = "=".repeat(width);
    let thin = "-".repeat(width);

    println!();
    println!("  {}", border);
    println!("  Comparison: {} vs {}", label_a, label_b);
    println!("  {}", thin);
    println!(
        "  {:<22} {:>16} {:>16} {:>10}",
        "Metric",
        label_a,
        label_b,
        "Speedup"
    );
    println!("  {}", thin);

    for sa in stats_a {
        if let Some(sb) = stats_b.iter().find(|s| s.label == sa.label) {
            let speedup = if sa.unit == "tok/s" {
                // Higher is better for throughput
                if sa.mean > 0.0 {
                    (sb.mean / sa.mean - 1.0) * 100.0
                } else {
                    0.0
                }
            } else {
                // Lower is better for latency/time
                if sb.mean > 0.0 {
                    (sa.mean / sb.mean - 1.0) * 100.0
                } else {
                    0.0
                }
            };

            let speedup_str = if speedup > 0.0 {
                format!("+{:.1}%", speedup)
            } else {
                format!("{:.1}%", speedup)
            };

            println!(
                "  {:<22} {:>16} {:>16} {:>10}",
                sa.label,
                sa.fmt_mean_std(),
                sb.fmt_mean_std(),
                speedup_str,
            );
        }
    }
    println!("  {}", border);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn compute_returns_none_on_empty() {
        assert!(Stats::compute("x", "ms", &[]).is_none());
    }

    #[test]
    fn compute_single_value_has_zero_stddev() {
        let s = Stats::compute("x", "ms", &[42.0]).unwrap();
        approx(s.mean, 42.0);
        approx(s.stddev, 0.0);
        approx(s.min, 42.0);
        approx(s.max, 42.0);
        approx(s.p50, 42.0);
        approx(s.p95, 42.0);
        assert_eq!(s.count, 1);
    }

    #[test]
    fn compute_uses_sample_stddev_not_population() {
        // values: 2, 4, 4, 4, 5, 5, 7, 9
        // mean = 5; sum_sq_dev = 9+1+1+1+0+0+4+16 = 32
        // sample variance = 32 / (n-1) = 32/7 ≈ 4.5714
        // sample stddev   = sqrt(32/7)    ≈ 2.1381
        // (population stddev would be sqrt(32/8) = 2.0 — n-1 denominator
        // is the correct estimator for an unknown distribution like ours.)
        let s = Stats::compute("x", "ms", &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
        approx(s.mean, 5.0);
        approx(s.stddev, (32.0_f64 / 7.0).sqrt());
        assert_eq!(s.count, 8);
    }

    #[test]
    fn compute_min_max_picked_correctly_when_unsorted_input() {
        let s = Stats::compute("x", "ms", &[10.0, 1.0, 100.0, 50.0, 5.0]).unwrap();
        approx(s.min, 1.0);
        approx(s.max, 100.0);
    }

    #[test]
    fn percentile_interpolates_between_indices() {
        // sorted [1, 2, 3, 4, 5]: p50 = index 2 = 3
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        approx(percentile(&sorted, 50.0), 3.0);
        // p25 of 5 values: rank = 0.25 * 4 = 1.0 → exact index 1 = 2
        approx(percentile(&sorted, 25.0), 2.0);
        // p95 of 5 values: rank = 0.95 * 4 = 3.8 → interp(3, 4) at frac 0.8
        // = 4 * 0.2 + 5 * 0.8 = 0.8 + 4 = 4.8
        approx(percentile(&sorted, 95.0), 4.8);
    }

    #[test]
    fn percentile_single_value_returns_that_value() {
        approx(percentile(&[42.0], 0.0), 42.0);
        approx(percentile(&[42.0], 95.0), 42.0);
    }

    #[test]
    fn fmt_val_picks_seconds_for_long_ms() {
        let s = Stats::compute("x", "ms", &[1.0]).unwrap();
        assert_eq!(s.fmt_val(500.0), "500ms");
        assert_eq!(s.fmt_val(1500.0), "1.5s");
        assert_eq!(s.fmt_val(10_000.0), "10.0s");
        // tok/s and unknown units don't fold to seconds.
        let t = Stats::compute("x", "tok/s", &[1.0]).unwrap();
        assert_eq!(t.fmt_val(123.45), "123.5");
        let u = Stats::compute("x", "%", &[1.0]).unwrap();
        assert_eq!(u.fmt_val(85.6), "86"); // {:.0} round
    }

    #[test]
    fn fmt_mean_std_combines_both() {
        let s = Stats::compute("x", "ms", &[100.0, 110.0, 120.0]).unwrap();
        let out = s.fmt_mean_std();
        // mean = 110, stddev = 10 → "110ms +/-10ms"
        assert!(out.contains("110ms"), "{out}");
        assert!(out.contains("10ms"), "{out}");
        assert!(out.contains("+/-"), "{out}");
    }
}
