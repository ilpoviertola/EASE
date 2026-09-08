# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import typing as tp
import collections

import torch
from torch import nn
import omegaconf

from .model import JAFAR

torch.serialization.add_safe_globals(
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Metadata,
        omegaconf.nodes.AnyNode,
        omegaconf.dictconfig.DictConfig,
        collections.defaultdict,
        tp.Any,
        list,
        dict,
        int,
    ]
)


class JAFARWrapper(nn.Module):
    def __init__(
        self,
        v_dim: int,
        ckpt_path: tp.Optional[str] = None,
        input_dim: int = 3,
        qk_dim: int = 128,
        kernel_size: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.v_dim = v_dim
        self.jafar = JAFAR(
            input_dim=input_dim,
            qk_dim=qk_dim,
            v_dim=v_dim,
            kernel_size=kernel_size,
            num_heads=num_heads,
        )
        self.jafar.eval()

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "jafar" in ckpt:
                ckpt = ckpt["jafar"]
            for key in list(ckpt.keys()):
                if not key.startswith("jafar"):
                    ckpt[f"jafar.{key}"] = ckpt.pop(key)

            self.load_state_dict(ckpt, strict=True)

    @torch.no_grad()
    def forward(
        self, features: torch.Tensor, image: torch.Tensor, output_size: tuple[int, int]
    ):
        return self.jafar(image, features, output_size)
