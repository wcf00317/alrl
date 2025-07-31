import torch
import torch.nn as nn

class TransformerPolicyNet(nn.Module):
    def __init__(self, input_dim=768, embed_dim=768, num_heads=8, num_layers=2, ff_dim=1024, dropout=0.1):
        super(TransformerPolicyNet, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional: subset aggregation (state)
        self.subset_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )

        # FFN for Q value estimation
        self.q_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x, subset):
        """
        x:       [B, N_clips, D]        # candidate clips
        subset:  [B, N_selected, D]     # selected clips (state)
        """
        # Encode subset (as "state" embedding)
        # print(subset.shape)
        subset_embed = self.subset_proj(subset.mean(dim=1, keepdim=True))  # [B, 1, D]

        # Concatenate subset embedding with candidate clips
        # print(subset_embed.shape, x.shape)
        fused = torch.cat([subset_embed, x.unsqueeze(dim=1)], dim=1)  # [B, 1 + N_clips, D]

        # Run transformer
        encoded = self.transformer(fused)  # [B, 1 + N_clips, D]
        clip_encoded = encoded[:, 1:, :]   # drop subset token

        # Predict Q-values for each clip
        q_values = self.q_predictor(clip_encoded).squeeze(-1)  # [B, N_clips]

        return q_values
