"""Real small PyTorch attention-based age regressor: learns a
per-channel attention weight (softmax over a learned scoring layer)
that pools the 16 channel features into a prediction -- the real
mechanism the resume's 'attention visualization module' inspects to
identify which cortical regions the model found most predictive."""
from __future__ import annotations

import torch
import torch.nn as nn

N_CHANNELS = 16


class AttentionAgeRegressor(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, hidden: int = 16):
        super().__init__()
        self.n_channels = n_channels
        self.value_embed = nn.Linear(1, hidden)
        self.channel_id_embed = nn.Embedding(n_channels, hidden)  # gives each channel a distinct identity
        self.attn_score = nn.Linear(hidden, 1)
        self.output_head = nn.Sequential(nn.Linear(hidden, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x, return_attention: bool = False):
        # x: (B, C) -> treat each channel's scalar feature as a token,
        # with a learned per-channel identity embedding so attention
        # can learn channel-specific importance, not just value magnitude
        channel_ids = torch.arange(self.n_channels, device=x.device)
        tokens = self.value_embed(x.unsqueeze(-1)) + self.channel_id_embed(channel_ids)[None, :, :]
        scores = self.attn_score(tokens).squeeze(-1)  # (B, C)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, C)
        pooled = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)
        pred = self.output_head(pooled).squeeze(-1)

        if return_attention:
            return pred, attn_weights
        return pred
