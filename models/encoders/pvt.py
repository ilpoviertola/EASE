# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import Optional
import timm
import torch
import torch.nn as nn
from lightning_fabric.utilities import rank_zero_info

bold_green = "\033[1;32m"
reset = "\033[0m"


class PvT(nn.Module):
    def __init__(
        self,
        backbone_name="pvt_v2_b5",
        ckpt_path: Optional[str] = None,
        num_fusion_blocks: int = 4,
        disable_sr: bool = True,
        prune_last_stage: bool = True,
        **kwargs,
    ):
        super().__init__()

        if ckpt_path is not None:
            rank_zero_info(
                f"{bold_green}Loading PvT-v2 weights from {ckpt_path}{reset}"
            )

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0 if ckpt_path is None else 1000,
            checkpoint_path=ckpt_path,
        )
        self.num_fusion_blocks = num_fusion_blocks

        # We need to init classifier head if loading from custom checkpoint
        if type(self.backbone.head) is nn.Linear:
            self.backbone.head = nn.Identity()
            self.backbone.num_classes = 0  # type: ignore

        # prune the last stage of PvTv2
        if prune_last_stage:
            self.backbone.stages = self.backbone.stages[:-1]  # type: ignore
            self.backbone.depths = self.backbone.depths[:-1]  # type: ignore
            self.backbone.feature_info = self.backbone.feature_info[:-1]  # type: ignore
            self.backbone.num_features = (
                self.backbone.head_hidden_size
            ) = self.backbone.feature_info[-1][
                "num_chs"  # type: ignore
            ]

            if disable_sr:
                for blk in self.backbone.stages[-1].blocks[-num_fusion_blocks:]:  # type: ignore
                    blk.attn.sr = None
                    blk.attn.norm = None

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(  # type: ignore
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)  # type: ignore

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def get_depth_to_stage(self, stage: int) -> int:
        if stage < 0 or stage >= len(self.backbone.depths):  # type: ignore
            raise ValueError(f"Invalid stage: {stage}")
        return sum(self.backbone.depths[:stage])  # type: ignore

    @property
    def num_backbone_blocks(self):
        return sum(self.backbone.depths)  # type: ignore
