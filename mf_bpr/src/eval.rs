use std::time::Duration;
use rand::Rng;
use anyhow::Result;
use crate::data::MovieLensData;

// Hit ratio function
pub fn calculate_hit_ratio<F>(data: &MovieLensData, k: usize, mut predict_fn: F) -> Result<f32>
    where F: FnMut(u32) -> Result<(Vec<f32>, Duration)> {

    let mut hits = 0;
    let total_test = data.test_set.len();
    
    // Useful to see the progress and to avoid %0
    let mut processed = 0;
    let log_interval = std::cmp::max(1, total_test / 10);

    println!("\nEvaluation on {} users", total_test);

    for &(u_idx, target_item_idx) in &data.test_set {
        
        // Call to the closure
        let (mut scores, _duration) = predict_fn(u_idx as u32)?;

        // Save the computed score for the item
        let target_score = scores[target_item_idx];

        // Mask with -inf all the seen items
        if let Some(history) = data.user_history.get(u_idx) {
            for &seen_item in history {
                if seen_item != target_item_idx {
                    scores[seen_item] = f32::NEG_INFINITY;
                }
            }
        }
        scores[target_item_idx] = target_score;

        // Counts how many items have the score higher than the test item
        let count_better = scores.iter().filter(|&&s| s > target_score).count();
        if count_better < k {
            hits += 1;
        }

        processed += 1;
        // Update the bar every 10% of test
        if processed % log_interval == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();

    Ok(hits as f32 / total_test as f32)
}

// Get avg, p50, p90, p95, p99 inference time
pub fn benchmark_latency<F>(data: &MovieLensData, num_requests: usize, mut predict_fn: F) -> Result<(f64, f64, f64, f64, f64)>
    where F: FnMut(u32) -> Result<(Vec<f32>, Duration)> {
    
    println!("\nStarting Benchmark on {} requests", num_requests);
    
    let mut rng = rand::thread_rng();
    let mut latencies: Vec<f64> = Vec::with_capacity(num_requests);

    for _ in 0..num_requests {
        // Get a random user
        let u_idx = rng.gen_range(0..data.num_users);


        // Call the closure
        let (_, duration) = predict_fn(u_idx as u32)?;

        latencies.push(duration.as_secs_f64() * 1000.0);
    }

    // All the statistics (computed using formulas (Python had numpy functions))
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = latencies.iter().sum();
    let avg = sum / num_requests as f64;
    
    // Closure to obtain quantiles
    let get_p = |p: f64| {
        let idx = (latencies.len() as f64 * p) as usize;
        latencies[idx.min(latencies.len() - 1)]
    };

    Ok((avg, get_p(0.50), get_p(0.90), get_p(0.95), get_p(0.99)))
}

