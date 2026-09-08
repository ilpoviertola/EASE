# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from tue-mps/EoMT,
# used under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationMS3(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[Union[list[int], list[float]]] = None,
        attn_mask_annealing_end_steps: Optional[Union[list[int], list[float]]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        lr_mult: float = 0.1,
        llrd_l2_enabled: bool = True,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        cons_coefficient: float = 0.0,
        sim_coefficient: float = 1.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
        per_frame: bool = True,
        imgs_per_audio: int = 1,
        save_masks_on_eval: bool = False,
        disable_masked_attn_for_val: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            lr_mult=lr_mult,
            llrd_l2_enabled=llrd_l2_enabled,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
            per_frame=per_frame,
            imgs_per_audio=imgs_per_audio,
            save_masks_on_eval=save_masks_on_eval,
            disable_masked_attn_for_val=disable_masked_attn_for_val,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            sim_coefficient=sim_coefficient,
            cons_coefficient=cons_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.cosine_consistency_loss: Optional[nn.CosineEmbeddingLoss] = None
        if cons_coefficient > 0.0 and not self.per_frame:
            self.cosine_consistency_loss = nn.CosineEmbeddingLoss(reduction="none")

        self.init_metrics_s4_ms3(
            ignore_idx,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

    def training_step(self, batch, batch_idx):
        imgs, aembs, targets = batch
        _targets = []
        for t in targets:
            _targets += self.group_targets(
                t["masks"], t["labels"], t["mask_to_frame_idx"], t["is_neg_sample"]
            )
        targets = _targets
        return super().training_step(
            batch=(imgs, aembs, targets),
            batch_idx=batch_idx,
        )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, aembs, _targets = batch
        imgs = [im[i] for im in imgs for i in range(len(im))]
        imgs_per_audio = self.imgs_per_audio
        if not self.per_frame:
            aembs = [ae[i : i + 1] for ae in aembs for i in range(len(ae))]
            imgs_per_audio = 1
        targets = []
        for t in _targets:
            targets += self.group_targets(
                t["masks"], t["labels"], t["mask_to_frame_idx"], t["is_neg_sample"]
            )
        is_neg_samples = [t["is_neg_sample"].item() for t in targets]

        img_sizes = [img.shape[-2:] for img in imgs]
        # as we crop the images, we need to replicate the audio embeddings for each crop
        crops, origins, aembs = self.window_imgs_s4_ms3_ss(imgs, aembs, imgs_per_audio)

        # Handle RoPE embeddings for cropped frames
        mask_logits_per_layer, class_logits_per_layer, _ = self.forward_with_crops(
            crops, origins, aembs, aembs.shape[0]
        )

        targets = self.to_per_pixel_targets_s4_ms3_ss(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_s4_ms3_ss(mask_logits, class_logits)
            logits = self.revert_window_logits_s4_ms3_ss(
                crop_logits, origins, img_sizes
            )

            self.update_metrics_s4_ms3_ss(logits, targets, i, is_neg_samples)

            if batch_idx == 0:
                self.plot_s4_ms3_ss(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

            if self.save_masks_on_eval and i == len(mask_logits_per_layer) - 1:
                uids = []
                for t in _targets:
                    uids += t["uids"]
                self.save_masks(imgs, targets, logits, uids)

    def on_validation_epoch_start(self) -> None:
        self._on_eval_epoch_start_s4_ms3_ss("val")

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_s4_ms3_ss("val")

    def on_validation_end(self):
        self._on_eval_end_s4_ms3_ss("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end_s4_ms3_ss("test")

    def on_test_end(self):
        self._on_eval_end_s4_ms3_ss("test")
