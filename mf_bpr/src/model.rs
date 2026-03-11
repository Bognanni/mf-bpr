use candle_core::{Result, Tensor, Module};
use candle_nn::{Embedding, VarBuilder, Init};

// Model to embed user and item
pub struct BPRModel {
    user_embeddings: Embedding,
    item_embeddings: Embedding,
}

impl BPRModel {
    // Initialization
    pub fn new(num_users: usize, num_items: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Initialize using Normal Distribution like Python
        let init_config = Init::Randn {
            mean: 0.0,
            stdev: 0.01
        };
        let u_weight = vb.pp("user_emb").get_with_hints((num_users, embedding_dim), "weight", init_config)?;
        let u_emb = Embedding::new(u_weight, embedding_dim);

        let i_weight = vb.pp("item_emb").get_with_hints((num_items, embedding_dim), "weight", init_config)?;
        let i_emb = Embedding::new(i_weight, embedding_dim);
        
        Ok(Self {
            user_embeddings: u_emb,
            item_embeddings: i_emb,
        })
    }

    // Forward pass: score difference (x_ui - x_uj)
    pub fn forward(&self, u_idxs: &Tensor, pos_i_idxs: &Tensor, neg_j_idxs: &Tensor) -> Result<Tensor> {
        // Embeddings for the current batch
        let u = self.user_embeddings.forward(u_idxs)?;      // Shape: [batch, dim]
        let i = self.item_embeddings.forward(pos_i_idxs)?;  // Shape: [batch, dim]
        let j = self.item_embeddings.forward(neg_j_idxs)?;  // Shape: [batch, dim]

        // dot product (score)
        // Multiply element-wise e sum on dim
        let x_ui = (u.clone() * i)?.sum(1)?; // Shape: [batch]
        let x_uj = (u * j)?.sum(1)?;         // Shape: [batch]

        // Score difference
        let diff = (x_ui - x_uj)?;
        Ok(diff)
    }

    // Predict item for a user
    pub fn predict(&self, u_idx: u32, device: &candle_core::Device) -> Result<Tensor> {
        // Embed the idx of the user
        let u_idx_tensor = Tensor::new(&[u_idx], device)?;
        let u_emb = self.user_embeddings.forward(&u_idx_tensor)?; // Shape: [1, dim]

        // Take the embeddings of all the items [num_items, dim]
        let all_items = self.item_embeddings.embeddings();
        
        // Matrix mul to have a list of items [1, dim] * [dim, num_items] = [1, num_items]
        let all_items_t = all_items.t()?;
        let scores = u_emb.matmul(&all_items_t)?;

        scores.squeeze(0)
    }
}

// Loss function to get the BPR
pub fn bpr_loss(diff: &Tensor) -> Result<Tensor> {
    // Formula: Loss = Softplus(-diff)
    // Softplus(x) = log(1 + exp(x))
    
    let neg_diff = diff.neg()?;

    let exp_neg_diff = neg_diff.exp()?;
    let one = Tensor::ones_like(&neg_diff)?;
    let softplus = (one + exp_neg_diff)?.log()?;
    
    let loss = softplus.mean(0)?;
    
    Ok(loss)
}