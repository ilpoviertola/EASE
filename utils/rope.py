# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the MMAudio library,
# used under MIT License.
# ---------------------------------------------------------------

from typing import Union

import torch
from einops import rearrange
from torch import Tensor


def compute_rope_rotations(
    length: int,
    dim: int,
    theta: int,
    *,
    freq_scaling: float = 1.0,
    device: Union[torch.device, str] = "cpu"
) -> Tensor:
    assert dim % 2 == 0

    # with torch.amp.autocast(device_type="cuda", enabled=False):
    pos = torch.arange(length, dtype=torch.float32, device=device)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    freqs *= freq_scaling

    rot = torch.einsum("..., f -> ... f", pos, freqs)
    rot = torch.stack(
        [torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1
    )
    rot = rearrange(rot, "n d (i j) -> 1 n d i j", i=2, j=2)
    return rot


def apply_rope(x: Tensor, rot: Tensor) -> Tensor:
    # with torch.amp.autocast(device_type="cuda", enabled=False):
    # rot: [1, T, 32, 2, 2] x: [B, 16, 1029, 64]
    B, _, _, _ = x.shape
    _, T, _, _, _ = rot.shape
    _x = x.permute(2, 1, 0, 3).float()  # [1029, 16, B, 64]
    _x = _x.view(*_x.shape[:-1], -1, 1, 2)  # [1029, 16, B, 32, 1, 2]
    # B = N*T -> apply rope N times over B dim
    if B > T:
        N = B // T
        rot = rot.repeat(1, N, 1, 1, 1)
    x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
    x_out = x_out.permute(2, 1, 0, 3, 4)
    return x_out.reshape(*x.shape).to(dtype=x.dtype)
