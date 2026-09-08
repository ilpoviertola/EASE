# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License and from tue-mps/EoMT,
# used under the MIT License.
# ---------------------------------------------------------------

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ease import EASE
from utils.rope import apply_rope
from models.feat_upscalers.mask_logits_identity_proj import MaskLogitsIdentityProj


class EASEPvT(EASE):
    def __init__(
        self,
        encoder: nn.Module,
        feat_upscaler: nn.Module,
        num_classes,
        num_q,
        final_dropout_rate: float = 0.1,
        rope_seq_len=1,
        query_generator: Optional[nn.Module] = None,
        num_blocks=4,
        masked_attn_enabled=True,
        similarity: Optional[nn.Module] = None,
        collapse_to_binary_logits: bool = False,
        audio_enhance: Optional[nn.Module] = None,
        pvt_stage_as_img_batch: int = -1,
        img_batch_proj: Optional[nn.Module] = None,
        mask_logit_proj: Optional[nn.Module] = None,
        fuser: Optional[nn.Module] = None,
        fusion_at_blocks: Optional[list[int]] = None,
        late_fusion: bool = False,
        propagate_aembs: bool = False,
    ):
        super().__init__(
            encoder=encoder,
            feat_upscaler=feat_upscaler,
            num_classes=num_classes,
            num_q=num_q,
            rope_seq_len=rope_seq_len,
            query_generator=query_generator,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
            similarity=similarity,
            collapse_to_binary_logits=collapse_to_binary_logits,
        )
        assert (
            self.encoder.num_fusion_blocks == num_blocks
        ), "Number of fusion blocks in PvT must match num_blocks of EASEPvT"

        if audio_enhance is not None:
            self.audio_enhance = audio_enhance
        else:
            self.audio_enhance = nn.Identity()  # type: ignore

        if img_batch_proj is not None:
            self.img_batch_proj = img_batch_proj
        else:
            self.img_batch_proj = nn.Identity()  # type: ignore

        if mask_logit_proj is not None:
            self.mask_logit_proj = mask_logit_proj
        else:
            self.mask_logit_proj = MaskLogitsIdentityProj()  # type: ignore

        self.pvt_stage_as_img_batch = pvt_stage_as_img_batch

        self.late_fusion = late_fusion
        self.fusion_at_blocks = fusion_at_blocks if fusion_at_blocks is not None else []
        if len(self.fusion_at_blocks) > 0:
            assert fuser is not None, "Fuser module must be provided."
            assert max(self.fusion_at_blocks) < self.encoder.num_backbone_blocks, "Fusion block index is out of bounds."  # type: ignore
            assert (
                min(self.fusion_at_blocks) >= 0
            ), "Fusion blocks must be non-negative."
            self.fuser = fuser

        if self.late_fusion:
            assert fuser is not None, "Fuser module must be provided."
            if not hasattr(self, "fuser"):
                self.fuser = fuser  # type: ignore

        if final_dropout_rate > 0:
            self.dropout = nn.Dropout2d(0.1)
        else:
            self.dropout = nn.Identity()  # type: ignore

        self.propagate_aembs = propagate_aembs

    def _sanity_check_feat_upscaler(self):
        if self.upscale.__class__.__name__ == "ConvScaler":
            assert (
                self.upscale.encoder_patch_size
                == self.encoder.backbone.patch_embed.patch_size  # type: ignore
            ), f"{self.upscale.encoder_patch_size=} != {self.encoder.backbone.patch_embed.patch_size=}"  # type: ignore
            assert (
                self.upscale.embed_dim == self.encoder.backbone.num_features  # type: ignore
            ), f"{self.upscale.embed_dim=} != {self.encoder.backbone.num_features=}"  # type: ignore
        elif self.upscale.__class__.__name__ == "JAFARWrapper":
            assert (
                self.upscale.v_dim == self.encoder.backbone.num_features  # type: ignore
            ), f"{self.upscale.v_dim=} != {self.encoder.backbone.num_features=}"  # type: ignore
        elif self.upscale.__class__.__name__ == "ConvScalerFixedDepth":
            assert (
                self.upscale.embed_dim == self.encoder.backbone.num_features  # type: ignore
            ), f"{self.upscale.embed_dim=} != {self.encoder.backbone.num_features=}"  # type: ignore

    def _initialize_mask_head(self):
        return nn.Sequential(
            nn.Linear(
                self.encoder.backbone.num_features,  # type: ignore
                self.encoder.backbone.num_features,  # type: ignore
            ),
            nn.GELU(),
            nn.Linear(
                self.encoder.backbone.num_features,  # type: ignore
                self.encoder.backbone.num_features,  # type: ignore
            ),
            nn.GELU(),
            nn.Linear(
                self.encoder.backbone.num_features,  # type: ignore
                self.encoder.backbone.num_features,  # type: ignore
            ),
        )

    def _initialize_class_head(self, num_classes: int):
        return nn.Linear(self.encoder.backbone.num_features, num_classes + 1)  # type: ignore

    def _initialize_q(self, num_q: int):
        return nn.Embedding(num_q, self.encoder.backbone.num_features)  # type: ignore

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[:, : self.num_q, self.num_q :][random_queries] = True

        return attn_mask

    def _predict(
        self, x: torch.Tensor, feat_size: tuple[int, int], img_batch: torch.Tensor
    ):
        q = x[:, : self.num_q, :]
        x = x[:, self.num_q :, :]

        x = x.transpose(1, 2).reshape(x.shape[0], -1, *feat_size)

        kwargs = {"image": img_batch, "output_size": img_batch.shape[-2:]}
        mask_feature = self.upscale(x, **kwargs)
        mask_feature = self.dropout(mask_feature)
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), mask_feature)
        mask_logits = self.mask_logit_proj(mask_logits, mask_feature)

        class_logits = self.class_head(q)
        if self.collapse_to_binary_logits:
            class_logits = self._collapse_to_binary_logits(class_logits)
        return mask_logits, class_logits

    @staticmethod
    def _interpolate_mask(
        mask: torch.Tensor, size: Union[int, tuple[int, int]]
    ) -> torch.Tensor:
        dtype = mask.dtype
        mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        if dtype == torch.bool:
            mask = (mask > 0.5).to(dtype)
        else:
            mask = mask.to(dtype)
        return mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        feat_size: tuple[int, int],
        rotations: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape
        H, W = feat_size
        q = module.q(x).reshape(B, N, module.num_heads, -1).permute(0, 2, 1, 3)  # type: ignore

        if module.pool is not None:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = module.sr(module.pool(x)).reshape(B, C, -1).permute(0, 2, 1)  # type: ignore
            x = module.norm(x)  # type: ignore
            x = module.act(x)  # type: ignore
            kv = (
                module.kv(x)  # type: ignore
                .reshape(B, -1, 2, module.num_heads, module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            if module.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                x = module.sr(x).reshape(B, C, -1).permute(0, 2, 1)  # type: ignore
                x = module.norm(x)  # type: ignore
                if mask is not None:
                    mask = self._interpolate_mask(mask, size=(H * W, x.size(1)))
                kv = (
                    module.kv(x)  # type: ignore
                    .reshape(B, -1, 2, module.num_heads, module.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    module.kv(x)  # type: ignore
                    .reshape(B, -1, 2, module.num_heads, module.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )

        k, v = kv.unbind(0)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)  # type: ignore

        if rotations is not None:
            q, k = apply_rope(q, rotations), apply_rope(k, rotations)

        dropout_p = module.attn_drop.p if self.training else 0.0  # type: ignore

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)  # type: ignore
        else:
            q = q * module.scale  # type: ignore
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)  # type: ignore
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))  # type: ignore
        return x

    def _attn_mask(
        self,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        i: int,
        feat_size: tuple[int, int],
    ):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(mask_logits, feat_size, mode="bilinear")
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[:, : self.num_q, self.num_q :] = interpolated > 0
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[  # type: ignore
                i - self.encoder.num_backbone_blocks + self.num_blocks  # type: ignore
            ],
        )
        return attn_mask

    def _q_generation_step(
        self, x: torch.Tensor, aembs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
        """Add audio information to the backbone visual feature extractor.

        Args:
            x (torch.Tensor): Visual features.
            aembs (torch.Tensor): Audio embeddings.

        Returns:
            tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
                Fused features, feature size, similarity (if calculated).
        """
        # TODO: Remove similarity calculation
        sim = None
        if self.calc_sim:
            sim = self.similarity(aembs, x)  # type: ignore

        q = self._get_learnable_queries(aembs).expand(x.shape[0], -1, -1)
        # TODO: Temporal (channel-wise) aggregation?
        x = torch.cat((q, x), dim=1)

        _, N, _ = x.shape
        assert N == math.isqrt(N) ** 2, f"Expected square number of tokens, got {N}"
        feat_size_ext = (int(math.sqrt(N)), int(math.sqrt(N)))
        return x, feat_size_ext, sim

    def _fusion_step(
        self,
        x: torch.Tensor,
        aembs: Optional[torch.Tensor],
        block_idx: int,
        feat_size: tuple[int, int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        pre_x = None
        if block_idx > self.encoder.num_backbone_blocks - self.num_blocks:  # type: ignore
            pre_x = x[:, : self.num_q, :]
            x = x = x[:, self.num_q :, :]

        x, aembs = self.fuser(x, aembs, *feat_size)  # type: ignore

        if pre_x is not None:
            x = torch.cat((pre_x, x), dim=1)
        return x, aembs

    def forward(self, x: torch.Tensor, aembs: Optional[torch.Tensor] = None):
        img_batch = F.interpolate(x, scale_factor=0.25, mode="bilinear")
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std  # type: ignore
        x = self.encoder.backbone.patch_embed(x)  # type: ignore

        aembs = self.audio_enhance(aembs)

        i = 0
        sim = None
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for s_i, stage in enumerate(self.encoder.backbone.stages):  # type: ignore
            if stage.downsample is not None:
                x = x.reshape(B, feat_size[0], feat_size[1], -1).permute(0, 3, 1, 2)
                x = stage.downsample(x)
            B, H, W, C = x.shape
            feat_size = feat_size_ext = (H, W)
            x = x.reshape(B, -1, C)

            for block in stage.blocks:
                if i in self.fusion_at_blocks:
                    x, _aembs = self._fusion_step(x, aembs, i, feat_size)
                    if self.propagate_aembs:
                        aembs = _aembs

                if i == self.encoder.num_backbone_blocks - self.num_blocks:  # type: ignore
                    x, feat_size_ext, sim = self._q_generation_step(x, aembs)  # type: ignore

                if (
                    self.masked_attn_enabled
                    and i >= self.encoder.num_backbone_blocks - self.num_blocks  # type: ignore
                ):
                    mask_logits, class_logits = self._predict(
                        stage.norm(x), feat_size, img_batch
                    )
                    mask_logits_per_layer.append(mask_logits)
                    class_logits_per_layer.append(class_logits)

                    attn_mask = self._attn_mask(x, mask_logits, i, feat_size)

                x = x + block.drop_path1(
                    self._attn(
                        block.attn, block.norm1(x), attn_mask, feat_size_ext, self.rope
                    )
                )
                x = x + block.drop_path2(block.mlp(block.norm2(x), feat_size_ext))
                i += 1

            x = stage.norm(x)
            if s_i == self.pvt_stage_as_img_batch:
                img_batch = x.reshape(B, -1, feat_size[0], feat_size[1])
                img_batch = self.img_batch_proj(img_batch)

        if self.late_fusion:
            x, aembs = self._fusion_step(
                x,
                aembs,
                self.encoder.num_backbone_blocks,  # type: ignore
                feat_size,
            )

        mask_logits, class_logits = self._predict(x, feat_size, img_batch)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer, sim
