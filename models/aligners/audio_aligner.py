from typing import Optional
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Sequential2(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.mod_list = nn.ModuleList(modules)

    def forward(self, x, y):
        results = (x, y)
        for m in self.mod_list:
            results = m(*results)
        return results


class FrequencyAvg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spatial, cls):
        return spatial.mean(2, keepdim=True), cls


class ChannelNorm(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward_spatial(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x, cls):
        return self.forward_spatial(x), self.forward_cls(cls)

    def forward_cls(self, cls):
        if cls is not None:
            return self.norm(cls)
        else:
            return None


class LearnedTimePool2(nn.Module):
    def __init__(self, in_dim, out_dim, width, maxpool, use_cls_layer):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        if maxpool:
            self.layer = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=width, stride=1, padding="same"),
                nn.MaxPool2d(kernel_size=(1, width), stride=(1, width)),
            )
        else:
            self.layer = nn.Conv2d(
                in_dim, out_dim, kernel_size=(1, width), stride=(1, width)
            )

        self.use_cls_layer = use_cls_layer
        if use_cls_layer:
            self.cls_layer = nn.Linear(in_dim, out_dim)

    def forward(self, spatial, cls):
        if cls is not None:
            if self.use_cls_layer:
                aligned_cls = self.cls_layer(cls)
            else:
                aligned_cls = cls
        else:
            aligned_cls = None

        return self.layer(spatial), aligned_cls


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_qkv=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, qkv


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False, return_qkv=False):
        y, attn, qkv = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_qkv:
            return x, attn, qkv
        return x


class SelfAttentionAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.num_heads = 6
        if dim % self.num_heads != 0:
            self.padding = self.num_heads - (dim % self.num_heads)
        else:
            self.padding = 0

        self.block = Block(
            dim + self.padding,
            num_heads=self.num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-4),  # type: ignore
        )

    def forward(self, spatial, cls):
        padded_feats = F.pad(spatial, [0, 0, 0, 0, self.padding, 0])

        B, C, H, W = padded_feats.shape
        proj_feats = padded_feats.reshape(B, C, H * W).permute(0, 2, 1)

        if cls is not None:
            assert len(cls.shape) == 2
            padded_cls = F.pad(cls, [self.padding, 0])
            proj_feats = torch.cat([padded_cls.unsqueeze(1), proj_feats], dim=1)

        aligned_feat, attn, qkv = self.block(proj_feats, return_qkv=True)

        if cls is not None:
            aligned_cls = aligned_feat[:, 0, :]
            aligned_spatial = aligned_feat[:, 1:, :]
        else:
            aligned_cls = None
            aligned_spatial = aligned_feat

        aligned_spatial = aligned_spatial.reshape(
            B, H, W, self.dim + self.padding
        ).permute(0, 3, 1, 2)

        aligned_spatial = aligned_spatial[:, self.padding :, :, :]
        if aligned_cls is not None:
            aligned_cls = aligned_cls[:, self.padding :]

        return aligned_spatial, aligned_cls


class AudioSA1x1Pool2(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mod_list = Sequential2(
            FrequencyAvg(),
            ChannelNorm(in_dim),
            LearnedTimePool2(in_dim, out_dim, 1, False, True),
            LearnedTimePool2(out_dim, out_dim, 1, False, False),
            SelfAttentionAligner(out_dim),
        )

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.mod_list.load_state_dict(state_dict, strict=True)

    def forward(self, *args, **kwargs):
        return self.mod_list(*args, **kwargs)
