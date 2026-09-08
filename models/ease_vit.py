# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License and from tue-mps/EoMT,
# used under the MIT License.
# ---------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ease import EASE
from models.feat_upscalers.mask_logits_identity_proj import MaskLogitsIdentityProj


class EASEViT(EASE):
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
        dino_stage_as_img_batch: int = -1,
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

        self.late_fusion = late_fusion
        self.fusion_at_blocks = fusion_at_blocks if fusion_at_blocks is not None else []
        if len(self.fusion_at_blocks) > 0:
            assert fuser is not None, "Fuser module must be provided."
            assert max(self.fusion_at_blocks) < len(self.encoder.backbone.blocks), "Fusion block index is out of bounds."  # type: ignore
            assert (
                min(self.fusion_at_blocks) >= 0
            ), "Fusion blocks must be non-negative."
            self.fuser = fuser

        if self.late_fusion:
            assert fuser is not None, "Fuser module must be provided."
            if not hasattr(self, "fuser"):
                self.fuser = fuser  # type: ignore

        self.dino_stage_as_img_batch = dino_stage_as_img_batch

        if final_dropout_rate > 0:
            self.dropout = nn.Dropout2d(0.1)
        else:
            self.dropout = nn.Identity()  # type: ignore

        self.propagate_aembs = propagate_aembs

    def _predict(
        self,
        x: torch.Tensor,
        img_batch: Optional[torch.Tensor] = None,
    ):
        q = x[:, : self.num_q, :]
        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]

        class_logits = self.class_head(q)
        if self.collapse_to_binary_logits:
            class_logits = self._collapse_to_binary_logits(class_logits)

        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        kwargs = {
            "image": img_batch,
            "output_size": img_batch.shape[-2:] if img_batch is not None else None,
        }
        xu = self.upscale(x, **kwargs)
        xu = self.dropout(xu)
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), xu)

        return mask_logits, class_logits

    def _q_generation_step(
        self, x: torch.Tensor, aembs: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add audio information to the backbone visual feature extractor.

        Args:
            x (torch.Tensor): Visual features.
            aembs (torch.Tensor): Audio embeddings.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]:
                Fused features, similarity (if calculated).
        """
        # TODO: Remove similarity calculation
        sim = None
        if self.calc_sim:
            sim = self.similarity(aembs, x)  # type: ignore

        q = self._get_learnable_queries(aembs).expand(x.shape[0], -1, -1)
        # TODO: Temporal (channel-wise) aggregation?
        x = torch.cat((q, x), dim=1)
        return x, sim

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def _fusion_step(
        self, x: torch.Tensor, aembs: Optional[torch.Tensor], block_idx: int
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if aembs is None:
            return x, None

        pre_x = None
        if block_idx > len(self.encoder.backbone.blocks) - self.num_blocks:
            pre_x = x[:, : self.num_q + self.encoder.backbone.num_prefix_tokens, :]
            x = x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        else:
            pre_x = x[:, : self.encoder.backbone.num_prefix_tokens, :]
            x = x = x[:, self.encoder.backbone.num_prefix_tokens :, :]

        x, aembs = self.fuser(  # type: ignore
            x,
            aembs,
            self.encoder.backbone.patch_embed.grid_size[0],  # type: ignore
            self.encoder.backbone.patch_embed.grid_size[1],  # type: ignore
        )

        if pre_x is not None:
            x = torch.cat((pre_x, x), dim=1)
        return x, aembs

    def forward(self, x: torch.Tensor, aembs: Optional[torch.Tensor] = None):
        img_batch = x
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std  # type: ignore

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)  # type: ignore

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        aembs = self.audio_enhance(aembs)

        sim = None
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i in self.fusion_at_blocks:
                x, _aembs = self._fusion_step(x, aembs, i)
                if self.propagate_aembs:
                    aembs = _aembs

            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x, sim = self._q_generation_step(x, aembs)  # type: ignore

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(
                    self.encoder.backbone.norm(x), img_batch
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)

            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

            if i == self.dino_stage_as_img_batch:
                img_batch = self.img_batch_proj(img_batch)

        if self.late_fusion:
            x, _ = self._fusion_step(x, aembs, len(self.encoder.backbone.blocks))

        mask_logits, class_logits = self._predict(
            self.encoder.backbone.norm(x), img_batch
        )
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer, sim
