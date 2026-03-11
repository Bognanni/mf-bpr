use reqwest::Client;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time;

// Configuration consts
// Time for each RPS level
const TEST_DURATION_SECS: u64 = 5;
const TIMEOUT_MS: u64 = 2000;


const RPS_STEPS: &[u64] = &[100, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000];

struct RequestResult {
    latency: f64,
    success: bool,
    status_code: Option<u16>,
}

async fn call_api(client: &Client, user_id: u32) -> RequestResult {
    let start = Instant::now();
    let url = format!("http://localhost:3000/recommend/{}", user_id);

    // Client request with timeout
    let request = client.get(&url).timeout(Duration::from_millis(TIMEOUT_MS));

    match request.send().await {
        Ok(resp) => {
            let status = resp.status();
            let success = status.is_success();
            
            RequestResult {
                latency: start.elapsed().as_secs_f64() * 1000.0,
                success,
                status_code: Some(status.as_u16()),
            }
        }
        Err(_) => RequestResult {
            latency: start.elapsed().as_secs_f64() * 1000.0,
            success: false,
            status_code: None, // Timeout o Connection Refused
        },
    }
}

#[tokio::main]
async fn main() {
    println!("Ramp up stress test");
    println!("-------------------------------------------------------------------------------------");
    println!("Target RPS | Real RPS |   Avg   |   P50   |   P90   |   P95   |   P99   | Errors");
    println!("-------------------------------------------------------------------------------------");

    // Single client for all the requests (to avoid connection time)
    let client = Client::builder()
        .pool_max_idle_per_host(5000)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_nodelay(true)
        .build()
        .unwrap();

    // Warm-up to avoid bigger times at the beginning
    println!("Status: Warming up connection pool...");
    warm_up(&client, 500).await;
    println!("Status: Warm-up complete. Starting benchmark.\n");

    for &target_rps in RPS_STEPS {
        let passed = run_stage(&client, target_rps).await;
        if !passed {
            println!("-------------------------------------------------------------------------------------");
            println!("Breaking point reached at {} RPS target.", target_rps);
            println!("Too many errors or server too slow");
            break;
        }

        time::sleep(Duration::from_secs(1)).await;
    }
}

async fn warm_up(client: &Client, count: u32) {
    let mut tasks = Vec::new();
    // Parallel requests to open the connections
    for i in 0..count {
        let c = client.clone();
        tasks.push(tokio::spawn(async move {
            // Ignore the results
            let _ = c.get(format!("http://localhost:3000/recommend/{}", (i % 100) + 1))
                .send()
                .await;
        }));
    }
    
    // Attendiamo che tutte le connessioni di warm-up siano finite
    for task in tasks {
        let _ = task.await;
    }
}

async fn run_stage(client: &Client, target_rps: u64) -> bool {
    let (tx, mut rx) = mpsc::unbounded_channel();
    
    // Every 10 ms a block of requests
    let batch_interval = Duration::from_millis(10);
    let reqs_per_batch = (target_rps as f64 * 0.010).ceil() as u64;

    let sender_handle = tokio::spawn({
        let client = client.clone();
        let tx = tx.clone();
        async move {
            let mut interval = time::interval(batch_interval);
            let start_sending = Instant::now();

            while start_sending.elapsed().as_secs() < TEST_DURATION_SECS {
                interval.tick().await;

                for i in 0..reqs_per_batch {
                    let client_ref = client.clone();
                    let tx_ref = tx.clone();
                    // Changing id to avoid caching
                    let user_id = (i % 600) + 1;

                    tokio::spawn(async move {
                        let res = call_api(&client_ref, user_id as u32).await;
                        let _ = tx_ref.send(res);
                    });
                }
            }
        }
    });

    let mut latencies = Vec::new();
    let mut errors = 0;
    let mut total_reqs = 0;

    let collection_timeout = time::sleep(Duration::from_secs(TEST_DURATION_SECS + 1));
    tokio::pin!(collection_timeout);

    loop {
        tokio::select! {
            Some(res) = rx.recv() => {
                total_reqs += 1;
                if res.success {
                    latencies.push(res.latency);
                } else {
                    errors += 1;
                }
            }
            _ = &mut collection_timeout => {
                break;
            }
        }
    }

    sender_handle.abort();

    // Statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let real_rps = total_reqs as f64 / TEST_DURATION_SECS as f64;
    let avg = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
    
    let get_p = |p: f64| {
        if latencies.is_empty() { 0.0 }
        else { latencies[(latencies.len() as f64 * p) as usize] }
    };

    let p50 = get_p(0.50);
    let p95 = get_p(0.95);
    let p90 = get_p(0.90);
    let p99 = get_p(0.99);

    let error_rate = (errors as f64 / total_reqs as f64) * 100.0;

    println!(
        "{:>10} | {:>8.0} | {:>7.2}ms | {:>7.2}ms | {:>7.2}ms | {:>7.2}ms | {:>7.2}ms | {:>5.1}%",
        target_rps, real_rps, avg, p50, p90, p95, p99, error_rate
    );

    // Breaking criteria
    if error_rate > 5.0 {
        println!("   -> Too many errors (>5%)");
        return false;
    }
    if p99 > 1000.0 {
        println!("   -> Too slow (P99 > 1s)");
        return false;
    }

    true
}