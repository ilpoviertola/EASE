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

from utils.rope import compute_rope_rotations


class EASE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feat_upscaler: nn.Module,
        num_classes,
        num_q,
        rope_seq_len=1,
        query_generator: Optional[nn.Module] = None,
        num_blocks=4,
        masked_attn_enabled=True,
        similarity: Optional[nn.Module] = None,
        collapse_to_binary_logits: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.upscale = feat_upscaler
        self.similarity = similarity
        self._sanity_check_feat_upscaler()

        self.use_q_gen = query_generator is not None
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.rope_seq_len = rope_seq_len
        self.collapse_to_binary_logits = collapse_to_binary_logits

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = self._initialize_q(num_q)
        if self.use_q_gen:
            self.q_gen = query_generator
            if type(self.q_gen) != nn.Identity:
                assert (
                    self.q_gen.output_dim == self.q.embedding_dim
                ), f"{self.q_gen.output_dim=} != {self.q.embedding_dim=}"

        self.class_head = self._initialize_class_head(num_classes)
        self.mask_head = self._initialize_mask_head()

        if rope_seq_len > 1:
            self.rope = self._initialize_rotations()
        else:
            self.rope = None

    @property
    def calc_sim(self):
        return self.similarity is not None and self.training

    def _collapse_to_binary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        bg = logits[..., 0:1]
        obj = torch.logsumexp(logits[..., 1:-1], dim=-1, keepdim=True)
        return torch.cat([bg, obj, logits[..., -1:]], dim=-1)

    def _sanity_check_feat_upscaler(self):
        if self.upscale.__class__.__name__ == "ConvScaler":
            assert (
                self.upscale.encoder_patch_size
                == self.encoder.backbone.patch_embed.patch_size
            ), f"{self.upscale.encoder_patch_size=} != {self.encoder.backbone.patch_embed.patch_size=}"
            assert (
                self.upscale.embed_dim == self.encoder.backbone.embed_dim
            ), f"{self.upscale.embed_dim=} != {self.encoder.backbone.embed_dim=}"
        elif self.upscale.__class__.__name__ == "JAFARWrapper":
            assert (
                self.upscale.v_dim == self.encoder.backbone.embed_dim
            ), f"{self.upscale.v_dim=} != {self.encoder.backbone.embed_dim=}"
        elif self.upscale.__class__.__name__ == "ConvScalerFixedDepth":
            assert (
                self.upscale.embed_dim == self.encoder.backbone.embed_dim
            ), f"{self.upscale.embed_dim=} != {self.encoder.backbone.embed_dim=}"

    def _initialize_mask_head(self):
        return nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

    def _initialize_class_head(self, num_classes: int):
        return nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

    def _initialize_q(self, num_q: int):
        if num_q <= 0:
            return nn.Identity()
        return nn.Embedding(num_q, self.encoder.backbone.embed_dim)

    def _initialize_rotations(self):
        base_freq = 1.0
        rot = compute_rope_rotations(
            self.rope_seq_len,
            64,
            10000,
            freq_scaling=base_freq,
            device=self.class_head.weight.device,
        )

        return nn.Buffer(rot, persistent=False)

    def _predict(self, x: torch.Tensor, img_batch: Optional[torch.Tensor] = None):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)
        if self.collapse_to_binary_logits:
            class_logits = self._collapse_to_binary_logits(class_logits)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        kwargs = {
            "image": img_batch,
            "output_size": img_batch.shape[-2:] if img_batch is not None else None,
        }
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x, **kwargs)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _get_learnable_queries(self, aembs: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_q_gen:
            assert (
                aembs is not None
            ), "Audio embeddings must be provided for query generation."
            # TODO: Some other aggregation method?
            # TODO: Add positional encoding?
            return self.q_gen(aembs) + self.q.weight[None, :, :]

        return self.q.weight[None, :, :]

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rotations: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError("Attention mechanism not implemented.")

    def forward(self, x: torch.Tensor, aembs: Optional[torch.Tensor] = None):
        raise NotImplementedError("Forward method not implemented.")
