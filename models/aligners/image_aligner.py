from typing import Optional

import torch
from torch import nn


class Identity2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x, y


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


def id_conv(dim, strength=0.9):
    conv = nn.Conv2d(dim, dim, 1, padding="same")
    start_w = conv.weight.data
    conv.weight.data = nn.Parameter(
        torch.eye(dim, device=start_w.device).unsqueeze(-1).unsqueeze(-1) * strength
        + start_w * (1 - strength)
    )
    conv.bias.data = nn.Parameter(conv.bias.data * (1 - strength))
    return conv


class LinearAligner(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_norm: bool = True,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if use_norm:
            self.norm = ChannelNorm(in_dim)
        else:
            self.norm = Identity2()

        if in_dim == out_dim:
            self.layer = id_conv(in_dim, 0)
        else:
            self.layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)

        self.cls_layer = nn.Linear(in_dim, out_dim)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=True)

    def forward(self, spatial, cls=None):
        norm_spatial, _ = self.norm(spatial, cls)
        if cls is not None:
            cls = self.cls_layer(cls)
        spatial = self.layer(norm_spatial)
        return spatial, cls
