use anyhow::Result;
use ort::session::Session;
use std::time::Duration;
use ort::value::Value;
use std::collections::HashSet;
use std::time::Instant;
use mf_bpr::data::MovieLensData;
use mf_bpr::eval::{calculate_hit_ratio, benchmark_latency};

fn main() -> Result<()> {
    // Load data
    let data = MovieLensData::load("ml-1m-csv")?;
    
    // Load the model
    let mut model = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file("bpr_model_1m.onnx")?;

    // Random user (In {} to use the model later)
    {
        let test_user_internal_idx = 100;

        // Input Vec and shape
        let input_array = vec![test_user_internal_idx as i64];
        let input_shape = vec![1]; // Batch size 1
        
        // ONNX tensor
        let input_tensor = Value::from_array((input_shape, input_array))?;
        
        // Input map
        let inputs = ort::inputs!["user_id" => input_tensor];

        // Start the count
        let start_time = Instant::now();

        // Inference
        let outputs = model.run(inputs)?;

        // Extract the output tuple
        let output_tuple = outputs["scores"].try_extract_tensor::<f32>()?;
        let scores = output_tuple.1;

        // Save the time
        let inference_time = start_time.elapsed();

        // Get the movies already seen
        let empty_set = HashSet::new();

        println!("\n------------------------------------------------");
        println!("\nFirst 10 items seen by a random user (idx {})", test_user_internal_idx);
        println!("\n------------------------------------------------");
        let seen_items = data.user_history.get(test_user_internal_idx).unwrap_or(&empty_set);
        
        let mut seen_titles: Vec<&str> = seen_items.iter()
            .map(|&idx| data.movie_titles.get(&idx).map(|s| s.as_str()).unwrap_or("Unknown"))
            .collect();
        seen_titles.sort();

        for (i, title) in seen_titles.iter().take(10).enumerate() {
            println!("{}. {}", i+1, title);
        }

        // Candidates list (idx, score)
        let mut candidates: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| !seen_items.contains(idx))
            .map(|(idx, &score)| (idx, score))
            .collect();

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        println!("------------------------------------------------");
        println!("Example of recommendations for a random user (idx {})", test_user_internal_idx);

        for (rank, (idx, score)) in candidates.iter().take(10).enumerate() {
            let title = data.movie_titles.get(idx).map(|s| s.as_str()).unwrap_or("Unknown");
            let raw_id = data.idx_to_item_raw.get(idx).unwrap_or(&0);
            
            println!("{}. {} (ID: {}, Score: {:.2})", rank + 1, title, raw_id, score);
        }

        println!("Inference time: {:.2?}ms", inference_time.as_secs_f64() * 1000.0);
    }
    
    // Closure called during the evaluation of the model
    let mut predict_closure = |u_idx: u32| -> Result<(Vec<f32>, Duration)> {
        
        let input_array = vec![u_idx as i64];
        let input_shape = vec![1];
        let input_tensor = Value::from_array((input_shape, input_array))?;
        let inputs = ort::inputs!["user_id" => input_tensor];

        let start = Instant::now();
        let outputs = model.run(inputs)?;
        let output_tuple = outputs["scores"].try_extract_tensor::<f32>()?;
        
        let scores_slice = output_tuple.1;
        let scores = scores_slice.to_vec();
        let duration = start.elapsed();

        Ok((scores, duration))
    };

    let hit_ratio = calculate_hit_ratio(&data, 10, &mut predict_closure)?;
    
    println!("\n------------------------------------------------");
    println!("Imported Python model Hit Ratio: {:.4} ({:.2})", hit_ratio, hit_ratio*100.00);
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

    Ok(())
}