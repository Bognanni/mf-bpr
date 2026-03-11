use serde::{Deserialize, Serialize};

// Struct of the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationResponse {
    pub user_id_raw: u32,
    pub recommendations: Vec<MovieRec>,
    pub inference_time_ms: f64,
}

// Struct of a single position in the ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovieRec {
    pub rank: usize,
    pub title: String,
    pub score: f32,
}