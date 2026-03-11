use candle_core::{Device};
use candle_nn::{Optimizer, VarMap, VarBuilder};
use anyhow::Result;
use std::time::{Duration, Instant};
use mf_bpr::data::MovieLensData;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use mf_bpr::eval::{calculate_hit_ratio, benchmark_latency};
use mf_bpr::model::{BPRModel, bpr_loss};


// Function that returns the recommended items for a user
fn recommend_for_user(model: &BPRModel, data: &MovieLensData, user_idx: usize,
    top_k: usize, device: &candle_core::Device) -> anyhow::Result<(Vec<(usize, f32)>, Duration)> {
    
    // Start the count
    let start_time = Instant::now();
    
    // Score for each item
    let scores_tensor = model.predict(user_idx as u32, device)?;
    
    // Cast into vec to handle operations then
    let scores: Vec<f32> = scores_tensor.to_vec1()?;

    // Save the time
    let inference_time = start_time.elapsed();
    
    let seen_items = &data.user_history[user_idx];
    let mut candidates: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|(item_idx, _score)| !seen_items.contains(item_idx))
        .map(|(idx, &s)| (idx, s))
        .collect();

    // Sort for decremental score
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_items = candidates.into_iter().take(top_k).collect();

    // Return first k items
    Ok((top_items, inference_time))
}

fn main() -> Result<()> {
    // Constants used
    let device = Device::cuda_if_available(0)?;
    let embedding_dim = 32;
    let batch_size = 1024;
    let lr = 0.005;
    let epochs = 20;

    println!("Using: {:?}", device);

    // rng for deterministic results
    let seed = 42;
    let mut rng = SmallRng::seed_from_u64(seed);

    // Loading data
    let data = MovieLensData::load("ml-latest-small")?;

    // Building the model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    let model = BPRModel::new(data.num_users, data.num_items, embedding_dim, vb)?;
    
    // Optimizer with weight decay
    let params = candle_nn::ParamsAdamW {
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), params)?;
    opt.set_learning_rate(lr);

    // Training loop
    let steps = data.interactions.len() / batch_size;

    // Start the count
    let start_time = Instant::now();

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        
        for _ in 0..steps {
            // Data Sampling
            let (u, i, j) = data.get_batch(batch_size, &device, &mut rng)?;
            
            // Forward and Loss
            let diff = model.forward(&u, &i, &j)?;
            let loss = bpr_loss(&diff)?;
            
            // Backward
            opt.backward_step(&loss)?;
            
            total_loss += loss.to_scalar::<f32>()?;
        }
        
        println!("Epoch {:>2} | Avg Loss: {:.5}", epoch, total_loss / steps as f32);
    }
    // Save the time
    let training_time = start_time.elapsed();

    // Closure called during the evaluation of the model
    let predict_closure = |u_idx: u32| -> Result<(Vec<f32>, Duration)> {
        let start = Instant::now();
        let scores_tensor = model.predict(u_idx, &device)?;
        let scores = scores_tensor.to_vec1()?;
        let duration = start.elapsed();
        Ok((scores, duration))
    };

    let hit_ratio = calculate_hit_ratio(&data, 10, predict_closure)?;
    println!("\n------------------------------------------------");
    println!("Rust Native Hit Ratio: {:.4} ({:.2})", hit_ratio, hit_ratio*100.00);
    println!("------------------------------------------------");

    let (avg, p50, p90, p95, p99) = benchmark_latency(&data, 1000, predict_closure)?;
    println!("------------------------------------------------");
    println!("Rust Native Inference Stats");
    println!("------------------------------------------------");
    println!("Avg: {:>8.4} ms", avg);
    println!("P50: {:>8.4} ms (Median)", p50);
    println!("P90: {:>8.4} ms", p90);
    println!("P95: {:>8.4} ms", p95);
    println!("P99: {:>8.4} ms", p99);
    println!("------------------------------------------------");

    let test_user_idx = 100;
    
    println!("\n------------------------------------------------");
    println!("First 10 items seen by a random user (idx {})", test_user_idx);
    println!("------------------------------------------------");
    let history = &data.user_history[test_user_idx];

    for (i, item_idx) in history.iter().take(10).enumerate() {
        let title = data.movie_titles.get(item_idx).map(|s| s.as_str()).unwrap_or("Unknown");
        println!("{}. {}", i+1, title);
    }
    println!("------------------------------------------------");

    // Item recommendations
    let (recommendations, inference_time) = recommend_for_user(&model, &data, test_user_idx, 10, &device)?;

    println!("Example of recommendations for a random user (idx {})", test_user_idx);
    for (i, (item_idx, score)) in recommendations.iter().enumerate() {
        let title = data.movie_titles.get(item_idx).map(|s| s.as_str()).unwrap_or("Unknown");
        println!("{}. {} (Score: {:.4})", i+1, title, score);
    }

    println!("Training time: {:.2?}", training_time);
    println!("Inference time: {:.2?}ms", inference_time.as_secs_f64() * 1000.0);

    Ok(())
}
