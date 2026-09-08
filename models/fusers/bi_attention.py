# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the yannqi/COMBO repository,
# which is licensed under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath


def permute_and_flatten(layer, N, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2).contiguous()
    layer = layer.reshape(N, -1, C)
    return layer


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class BiMultiHeadAttention(nn.Module):
    def __init__(
        self, v_dim, a_dim, embed_dim, num_heads, dropout=0.1, use_sigmoid=False
    ):
        """
        v_dim: visual feature dimension
        a_dim: audio feature dimension
        embed_dim: embedding dimension
        num_heads: number of heads
        """
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.use_sigmoid = use_sigmoid

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.a_proj = nn.Linear(self.a_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_a_proj = nn.Linear(self.a_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_a_proj = nn.Linear(self.embed_dim, self.a_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a_proj.weight)
        self.a_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_a_proj.weight)
        self.values_a_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_a_proj.weight)
        self.out_a_proj.bias.data.fill_(0)

    def forward(self, v, a, pos_v=None, pos_a=None):
        bsz, tgt_len, _ = v.size()

        if pos_v is None:
            query_states = self.v_proj(v) * self.scale
        else:
            query_states = (self.v_proj(v + pos_v)) * self.scale

        if pos_a is None:
            key_states = self._shape(self.a_proj(a), -1, bsz)
        else:
            key_states = self._shape((self.a_proj(a + pos_a)), -1, bsz)

        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_a_states = self._shape(self.values_a_proj(a), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        # For bi-attention, we need the value_v_states and value_a_states
        value_v_states = value_v_states.view(*proj_shape)
        value_a_states = value_a_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2).contiguous())

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_a = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        # The purpose of this operation is to prevent underflow or overflow during numerical calculations, which can lead to inaccurate results or errors.
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2).contiguous()
        # Max-Normalization
        attn_weights_a = (
            attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        )

        if self.clamp_min_for_underflow:
            attn_weights_a = torch.clamp(
                attn_weights_a, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_a = torch.clamp(
                attn_weights_a, max=50000
            )  # Do not increase 50000, data type half has quite limited range
        if self.use_sigmoid:
            attn_weights_a = attn_weights_a.sigmoid()
            attn_weights_v = nn.functional.sigmoid(attn_weights)
        else:
            attn_weights_a = attn_weights_a.softmax(dim=-1)
            attn_weights_v = nn.functional.softmax(attn_weights, dim=1)
        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_a = F.dropout(attn_weights_a, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_a_states)
        attn_output_a = torch.bmm(attn_probs_a, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_a.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_a` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_a.size()}"
            )

        # Reshape back to initial dim
        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).contiguous()
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_a = attn_output_a.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_a = attn_output_a.transpose(1, 2).contiguous()
        attn_output_a = attn_output_a.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_a = self.out_a_proj(attn_output_a)

        return attn_output_v, attn_output_a


class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        audio_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
        use_sigmoid: bool = False,
    ):
        """Bi-Way Attention across audio and visual features.

        Args:
            vision_dim (int): Dimensionality of visual features
            audio_dim (int): Dimensionality of audio features
            embed_dim (int): Dimensionality of the embedding space
            num_heads (int): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
            init_values (float, optional): Initialization values for layer scales. Defaults to 1e-4.
            use_sigmoid (bool, optional): Whether to use sigmoid normalization for attention weights. Defaults to False.
        """
        super().__init__()

        # Pre-layer norm
        self.layer_norm_v = nn.LayerNorm(vision_dim)
        self.layer_norm_a = nn.LayerNorm(audio_dim)

        self.attn = BiMultiHeadAttention(
            v_dim=vision_dim,
            a_dim=audio_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_sigmoid=use_sigmoid,
        )

        # Add layer scale for training stability
        self.gamma_v = nn.Parameter(
            init_values * torch.ones((vision_dim)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_a = nn.Parameter(
            init_values * torch.ones((audio_dim)), requires_grad=True
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_feature: torch.Tensor,
        pos_v: Optional[torch.Tensor] = None,
        pos_a: Optional[torch.Tensor] = None,
    ):

        v = self.layer_norm_v(visual_features)
        a = self.layer_norm_a(audio_feature)
        delta_v, delta_a = self.attn(v, a, pos_v, pos_a)
        v = v + self.drop_path(self.gamma_v * delta_v)
        a = a + self.drop_path(self.gamma_a * delta_a)
        return v, a


class BiAttentionFuser(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        audio_dim: int,
        embed_dim: int,
        layer_count: int = 1,
        audio_token_num: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
        use_sigmoid: bool = False,
    ):
        """Bi-Attention based audio-visual feature fusion module.

        Args:
            vision_dim (int): Dimensionality of visual features.
            audio_dim (int): Dimensionality of audio features.
            embed_dim (int): Dimensionality of the embedding space.
            layer_count (int, optional): Number of bi-attention layers. Defaults to 1.
            audio_token_num (int, optional): Number of audio tokens. Defaults to 1.
            num_heads (int): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
            init_values (float, optional): Initialization values for layer scales. Defaults to 1e-4.
            use_sigmoid (bool, optional): Whether to use sigmoid normalization for attention weights. Defaults to False.
        """

        super().__init__()

        self.layers = nn.ModuleList(
            [
                BiAttentionBlock(
                    vision_dim=vision_dim,
                    audio_dim=audio_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                    init_values=init_values,
                    use_sigmoid=use_sigmoid,
                )
                for _ in range(layer_count)
            ]
        )

        # Position embeddings
        self.audio_pos = nn.Embedding(audio_token_num, audio_dim)
        self.pe_layer = PositionEmbeddingSine(vision_dim // 2, normalize=True)

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_feature: torch.Tensor,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse audio and visual features using bi-attention.

        Args:
            visual_features (torch.Tensor): Visual features of shape (B, N, C_v) or (B, C_v, H, W).
            audio_feature (torch.Tensor): Audio features of shape (B, T, C_a).
            h (int, optional): Height of the visual feature map. Needed if visual_features is (B, N, C_v). Defaults to None.
            w (int, optional): Width of the visual feature map. Needed if visual_features is (B, N, C_v). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Fused visual and audio features.
        """
        reshape_v = True  # reshape visual features back to (B, C, H, W) if needed
        if len(visual_features.shape) == 3:
            assert h is not None and w is not None, "Height and width must be provided."
            visual_features = rearrange(
                visual_features, "b (h w) c -> b c h w", h=h, w=w
            )
            reshape_v = False

        B, C, H, W = visual_features.shape
        pos_a = self.audio_pos.weight[None, :, :].expand(audio_feature.size(0), -1, -1)
        pos_v = self.pe_layer(visual_features).flatten(2).permute(0, 2, 1).contiguous()
        visual_features = permute_and_flatten(visual_features, B, C, H, W)

        for layer in self.layers:
            visual_features, audio_feature = layer(
                visual_features, audio_feature, pos_v=pos_v, pos_a=pos_a
            )
            pos_a, pos_v = None, None  # only use positional encoding at first layer

        if reshape_v:
            visual_features = visual_features.transpose(1, 2).reshape(B, C, H, W)

        return visual_features, audio_feature
