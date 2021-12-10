import math
import torch
import torch.nn as nn


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_dict=1500, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_pos_dict, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)

    def forward(self, x, mask):
        x_len = x.shape[-1]
        i = torch.arange(x_len, device=x.device)
        x_emb = self.row_embed(i)
        pos = x_emb.unsqueeze(0).permute(0, 2, 1).repeat(x.shape[0], 1, 1)
        return pos
