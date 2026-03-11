use std::time::Instant;
use reqwest::Client;
use futures::future::join_all;
use mf_bpr::api::RecommendationResponse;

// Function to do a single request
async fn call_api(client: &Client, user_id: u32) {
    let start = Instant::now();
    let url = format!("http://localhost:3000/recommend/{}", user_id);

    // Send the request
    match client.get(&url).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                // Parse the JSON
                match resp.json::<RecommendationResponse>().await {
                    Ok(data) => {
                        let elapsed = start.elapsed().as_millis();
                        println!(
                            "User {:<4} | {} recs | {:>3}ms (Server: {:.2}ms)",
                            data.user_id_raw,
                            data.recommendations.len(),
                            elapsed,
                            data.inference_time_ms
                        );
                    }
                    Err(e) => println!("User {:<4} | Errore parsing JSON: {}", user_id, e),
                }
            } else {
                println!("User {:<4} | Status Error: {}", user_id, resp.status());
            }
        },
        Err(e) => println!("User {:<4} | Connection Error: {}", user_id, e),
    }
}

#[tokio::main]
async fn main() {
    println!("---------------------------------------");
    println!("Starting stress test (50 requests).");
    
    // Single client for all the requests
    let client = Client::new();
    
    // Count the whole time
    let start_global = Instant::now();
    let mut tasks = Vec::new();

    for user_id in 1..=50 {
        // Clone the client ref
        let client_ref = client.clone();
        
        let task = tokio::spawn(async move {
            call_api(&client_ref, user_id).await;
        });
        
        tasks.push(task);
    }

    // Wait for all the tasks
    join_all(tasks).await;

    let duration = start_global.elapsed();
    println!("---------------------------------------");
    println!("Test finished in {:.2?}", duration);
}