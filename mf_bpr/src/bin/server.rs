use axum::{extract::{Path, State}, routing::get, Json, Router, http::StatusCode};
use std::sync::Arc;
use std::collections::HashSet;
use std::time::Instant;
use ort::{session::Session, value::Value};
use async_channel::{Sender, Receiver};
use mf_bpr::data::MovieLensData;
use mf_bpr::api::{MovieRec, RecommendationResponse};
use tower_http::cors::CorsLayer;

// Shared state
struct AppState {
    pool_rx: Receiver<Session>,
    pool_tx: Sender<Session>,
    data: MovieLensData,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting server");

    // Load the dataset
    let data = MovieLensData::load("ml-1m-csv")?;
    
    // Number of parallel sessions = number of core of the cpu
    let num_workers = num_cpus::get();
    println!("Configuration Pool: {} parallel sessions.", num_workers);

    // Channel
    let (tx, rx) = async_channel::bounded(num_workers);
    
    for i in 0..num_workers {
        // A session for each worker
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file("bpr_model_1m.onnx")?;

        tx.send(session).await.expect("Error with the initialization.");
        println!("Instance {} loaded.", i + 1);
    }

    // Start the Server
    let shared_state = Arc::new(AppState { 
        pool_rx: rx,
        pool_tx: tx,
        data
    });

    let app = Router::new()
        .route("/recommend/:user_id", get(handle_recommend))
        .with_state(shared_state)
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Server started on http://localhost:3000");
    axum::serve(listener, app).await?;

    Ok(())
}

// Function to handle a single request
async fn handle_recommend(Path(user_id_raw): Path<u32>, State(state): State<Arc<AppState>>,)
-> Result<Json<RecommendationResponse>, (StatusCode, String)> {
    
    // Start the time (to run the entire function)
    let start = Instant::now();

    // Search the user idx
    let internal_idx = match state.data.user_raw_to_idx.get(&user_id_raw) {
        Some(&idx) => idx,
        None => return Err((StatusCode::NOT_FOUND, format!("User ID {} not found", user_id_raw))),
    };

    // Get session from the pool
    let mut session = state.pool_rx.recv().await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Pool error".to_string()))?;

    let input_array = vec![internal_idx as i64];
    let input_tensor = Value::from_array((vec![1], input_array))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let inputs = ort::inputs!["user_id" => input_tensor];

    // Inference
    let outputs = session.run(inputs)
        .map_err(|e| {
            // Return the error if it fails
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    // Extract and copying the results
    let scores_vec: Vec<f32> = outputs["scores"]
        .try_extract_tensor::<f32>()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .1
        .to_vec();

    // Drop the output to stop reading
    drop(outputs);

    // Return the session
    let _ = state.pool_tx.send(session).await;
    
    // Get the top 10 candidates
    let empty_set = HashSet::new();
    let seen_items = state.data.user_history.get(internal_idx).unwrap_or(&empty_set);

    let mut candidates: Vec<(usize, f32)> = scores_vec
        .iter()
        .enumerate()
        .filter(|(idx, _)| !seen_items.contains(idx))
        .map(|(idx, &s)| (idx, s))
        .collect();

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    let top_k = 10;
    let recs: Vec<MovieRec> = candidates.iter().take(top_k).enumerate().map(|(rank, (idx, score))| {
        let title = state.data.movie_titles.get(idx).cloned().unwrap_or_else(|| "Unknown".to_string());
        MovieRec {
            rank: rank + 1,
            title,
            score: *score,
        }
    }).collect();

    let duration = start.elapsed();

    Ok(Json(RecommendationResponse {
        user_id_raw,
        recommendations: recs,
        inference_time_ms: duration.as_secs_f64() * 1000.0,
    }))
}