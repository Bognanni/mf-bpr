import torch
import torch.nn as nn

class BPRModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()

        self.u_emb = nn.Embedding(num_users, embedding_dim)
        self.i_emb = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.u_emb.weight, std=0.01)
        nn.init.normal_(self.i_emb.weight, std=0.01)

    # Forward pass: score difference (x_ui - x_uj)
    def calculate_loss(self, u, i, j):
        u_v = self.u_emb(u)  # Shape: [batch, dim]
        i_v = self.i_emb(i)
        j_v = self.i_emb(j)

        # dot product (score)
        # Multiply element-wise e sum on dim
        x_ui = (u_v * i_v).sum(dim=1)
        x_uj = (u_v * j_v).sum(dim=1)

        return x_ui - x_uj

    # Predict item for a user
    def forward(self, u_idx_tensor):
        with torch.no_grad():
            # Embed the idx of the user
            u_v = self.u_emb(u_idx_tensor)

            # Take the embeddings of all the items [num_items, dim]
            all_items = self.i_emb.weight

            # Matrix mul to have a list of items [1, dim] * [dim, num_items] = [1, num_items]
            scores = torch.matmul(u_v, all_items.T)

            return scores.squeeze()
