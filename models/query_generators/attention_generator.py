# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the vvvb-github/AVSegFormer
# library
# ---------------------------------------------------------------

import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query):
        out1 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + out1)
        out3 = self.ffn(query)
        query = self.norm2(query + out3)
        return query


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, audio_feat):
        out1 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + out1)
        out2 = self.cross_attn(query, audio_feat, audio_feat)[0]
        query = self.norm2(query + out2)
        out3 = self.ffn(query)
        query = self.norm3(query + out3)
        return query


# TODO: Add positional encoding
# TODO: Queries from encoded image that is also used in upsampler?
class AttentionGenerator(nn.Module):
    def __init__(
        self,
        num_layers,
        query_num,
        embed_dim=320,
        num_heads=8,
        hidden_dim=None,
        project_output=True,
        project_output_dim=320,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or (embed_dim * 4)
        self.query = nn.Embedding(query_num, embed_dim)
        self.layers = nn.ModuleList(
            [
                AttentionLayer(embed_dim, num_heads, self.hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.output_dim = project_output_dim
        project_output = project_output or (project_output_dim != embed_dim)
        if project_output:
            self.output_proj = nn.Linear(embed_dim, project_output_dim)
        else:
            self.output_proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, audio_feat):
        bs = audio_feat.shape[0]
        query = self.query.weight[None, :, :].repeat(bs, 1, 1)
        for layer in self.layers:
            query = layer(query, audio_feat)
        query = self.output_proj(query)
        return query


class SelfAttentionGenerator(nn.Module):
    def __init__(
        self,
        num_layers,
        query_num,
        embed_dim=1024,
        num_heads=8,
        hidden_dim=2048,
        project_input=True,
        project_input_dim=128,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Embedding(query_num, embed_dim)
        self.layers = nn.ModuleList(
            [
                SelfAttentionLayer(embed_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        project_input = project_input or (project_input_dim != embed_dim)
        if project_input:
            self.input_proj = nn.Linear(project_input_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, audio_feat):
        bs = audio_feat.shape[0]
        audio_feat = self.input_proj(audio_feat)
        query = self.query.weight[None, :, :].repeat(bs, 1, 1)
        query = torch.cat([query, audio_feat], dim=1)
        for layer in self.layers:
            query = layer(query)
        return query[:, 1:]
