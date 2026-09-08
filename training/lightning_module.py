# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from tue-mps/EoMT,
# used under the MIT License.
# ---------------------------------------------------------------

import math
from typing import Any, Optional, Union
from pathlib import Path
import lightning
from lightning.fabric.utilities import rank_zero_info, rank_zero_warn
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate, binary_cross_entropy_with_logits
from einops import rearrange

from datasets.constants import MASK_SIZE
from utils.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule
from utils.avs_semantic_with_negatives_metric import (
    AVSSMetricWithNegatives as AVSSMetric,
)
from utils.save_mask_util import colored_masks

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[Union[list[int], list[float]]],
        attn_mask_annealing_end_steps: Optional[Union[list[int], list[float]]],
        lr: float,
        llrd: float,
        lr_mult: float,
        llrd_l2_enabled: bool,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        per_frame: bool,
        imgs_per_audio: int,
        ckpt_path=None,
        load_ckpt_class_head=True,
        save_masks_on_eval=False,
        disable_masked_attn_for_val=True,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.llrd_l2_enabled = llrd_l2_enabled
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.per_frame = per_frame
        self.imgs_per_audio = imgs_per_audio
        self.save_masks_on_eval = save_masks_on_eval
        self.disable_masked_attn_for_val = disable_masked_attn_for_val

        self.strict_loading = False

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}

            if not load_ckpt_class_head:
                ckpt = {
                    k: v
                    for k, v in ckpt.items()
                    if "class_head" not in k and "class_predictor" not in k
                }

            incompatible_keys = self.load_state_dict(ckpt, strict=False)

            if incompatible_keys.missing_keys:
                if not load_ckpt_class_head:
                    missing_keys = [
                        key
                        for key in incompatible_keys.missing_keys
                        if "class_head" not in key and "class_predictor" not in key
                    ]
                else:
                    missing_keys = incompatible_keys.missing_keys

                if missing_keys:
                    raise ValueError(f"Missing keys: {missing_keys}")

            if incompatible_keys.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys: {incompatible_keys.unexpected_keys}"
                )

        self.log = torch.compiler.disable(self.log)  # type: ignore
        self.disabled_masked_attn_for_val = False

    def on_train_epoch_start(self) -> None:
        self.network.train()

    def on_validation_epoch_start(self) -> None:
        self.network.eval()

    def on_test_epoch_start(self) -> None:
        self.network.eval()

    def on_fit_start(self):
        if self.attn_mask_annealing_enabled:
            self.attn_mask_annealing_start_steps = self.set_mask_annealing_steps(
                self.attn_mask_annealing_start_steps
            )
            self.attn_mask_annealing_end_steps = self.set_mask_annealing_steps(
                self.attn_mask_annealing_end_steps
            )

    def set_mask_annealing_steps(
        self, steps: Optional[Union[list[int], list[float]]]
    ) -> Union[list[int], None]:
        if not steps:
            return steps
        if isinstance(steps[0], float):
            assert self.trainer, f"{self.trainer=}"
            total_steps = self.trainer.estimated_stepping_batches
            steps = [math.floor(s * total_steps) for s in steps]
        return steps

    def configure_optimizers(self):
        def _get_block_i(name_list, i, encoder):
            if encoder.__class__.__name__ == "PvT":
                stage = int(name_list[i - 1])
                return encoder.get_depth_to_stage(stage) + int(name_list[i + 1])
            return int(name_list[i + 1])

        encoder_param_names = {
            n for n, _ in self.network.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = self.network.encoder.num_backbone_blocks
        block_i = backbone_blocks

        l2_blocks = torch.arange(
            backbone_blocks - self.network.num_blocks, backbone_blocks
        ).tolist()

        for name, param in reversed(list(self.named_parameters())):
            lr = self.lr * math.sqrt(
                self.trainer.world_size * self.trainer.accumulate_grad_batches
            )
            if name.replace("network.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")

                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = _get_block_i(name_list, i, self.network.encoder)
                        is_block = True

                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)

                elif (is_block or block_i == 0) and self.lr_mult != 1.0:
                    lr *= self.lr_mult

                if "backbone.norm" in name:
                    lr = self.lr

                if (
                    is_block
                    and (block_i in l2_blocks)
                    and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0))
                ):
                    lr = self.lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def save_masks(
        self,
        imgs: list[torch.Tensor],
        targets: list[torch.Tensor],
        logits: list[torch.Tensor],
        uids: list[str],
        save_pred_only: bool = True,
    ):
        assert (
            len(imgs) == len(targets) == len(logits) == len(uids)
        ), f"{len(imgs)=} != {len(targets)=} != {len(logits)=} != {len(uids)=}"
        assert self.trainer, f"{self.trainer=}"
        assert self.trainer.logger, f"{self.trainer.logger=}"
        if type(self.trainer.logger.experiment.dir) is not str:  # type: ignore
            return
        save_dir = Path(self.trainer.logger.experiment.dir) / "masks"  # type: ignore
        save_dir.mkdir(exist_ok=True)
        for img, target, logit, uid in zip(imgs, targets, logits, uids):
            uid, frame_num = uid.rsplit("/", 1)
            if save_pred_only:
                sd_sub = Path(save_dir / uid)
                sd_sub.mkdir(parents=True, exist_ok=True)
                preds = torch.argmax(logit, dim=0)[None, ...]
                colored_pred = colored_masks(preds.cpu().numpy())[0]
                im = Image.fromarray(colored_pred).resize((360, 240))
                im.save(sd_sub / f"{uid}_{frame_num}_pred.png")
            else:
                plot: Image.Image = self.plot_s4_ms3_ss(
                    img,
                    target,
                    logit,
                    log_prefix=None,
                    block_idx=None,
                    batch_idx=None,
                    to_logger=False,
                )
                plot.save(save_dir / f"{uid}_{frame_num}.png")

    def _select_logits(self, mask_logits_per_block, class_logits_per_block):
        return mask_logits_per_block, class_logits_per_block

    @staticmethod
    @torch.compiler.disable
    def group_targets(
        masks: torch.Tensor,
        class_labels: torch.Tensor,
        indices: torch.Tensor,
        is_neg_sample: torch.Tensor,
    ):
        # Group indices by their values
        groups = {}
        for i, idx in enumerate(indices):
            idx_val = idx.item()
            if idx_val not in groups:
                groups[idx_val] = []
            groups[idx_val].append(i)

        # Extract masks for each group
        grouped = []
        for key in sorted(groups.keys()):
            mask_indices = torch.tensor(groups[key], device=masks.device)
            selected_masks = torch.index_select(masks, 0, mask_indices)
            selected_labels = torch.index_select(class_labels, 0, mask_indices)
            grouped.append(
                {
                    "masks": selected_masks,
                    "labels": selected_labels,
                    "is_neg_sample": torch.tensor(
                        [is_neg_sample[key]], device=masks.device
                    ),
                }
            )
        return grouped

    def forward_with_crops(self, crops, origins, aembs, bs):
        if not (hasattr(self.network, "rope") and self.network.rope is not None):
            return self(crops, aembs)

        # Extract original frame positions from origins
        original_positions = [origin[0] for origin in origins]
        # Create rope indices that match the original temporal positions
        rope_indices = torch.tensor(original_positions, device=crops.device)

        # Adjust the network's rope embeddings to match the cropped sequence
        if len(rope_indices) != self.network.rope_seq_len * bs:
            # Temporarily adjust rope embeddings for this forward pass
            original_rope = self.network.rope
            # Select rope embeddings based on original frame positions
            adjusted_rope = self.network.rope.repeat(
                1, bs // self.network.rope_seq_len, 1, 1, 1
            )[:, rope_indices]
            self.network.rope = adjusted_rope
            try:
                return self(crops, aembs)
            finally:
                # Restore original rope embeddings
                self.network.rope = original_rope
        else:
            return self(crops, aembs)

    def forward(self, imgs, aembs=None, labels=None):
        x = imgs / 255.0

        return self.network(x, aembs)

    def training_step(self, batch, batch_idx):
        losses_all_blocks = {}
        imgs, aembs, targets = batch

        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B * T, C, H, W)
        if not self.per_frame:
            aembs = aembs.view(B * T, 1, -1)

        mask_logits_per_block, class_logits_per_block, similarity = self(imgs, aembs)
        if self.use_consistency_loss:
            losses_all_blocks = self.compute_consistency_loss(
                mask_logits_per_block, B, T, losses_all_blocks
            )

        if not self.per_frame:
            mask_logits_per_block, class_logits_per_block = self._select_logits(
                mask_logits_per_block, class_logits_per_block
            )

        if similarity is not None:
            similarity_labels = torch.cat([~t["is_neg_sample"] for t in targets]).to(
                torch.float
            )
            losses_all_blocks["similarity"] = binary_cross_entropy_with_logits(
                similarity, similarity_labels
            )
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "test")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def init_metrics_s4_ms3(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [AVSSMetric(t=1, num_cls=self.num_classes) for _ in range(num_blocks)]
        )

    def init_metrics_ss(self, ignore_idx, num_blocks):
        num_cls = self.num_classes if not self.network.collapse_to_binary_logits else 2
        self.metrics = nn.ModuleList(
            [AVSSMetric(t=1, num_cls=num_cls) for _ in range(num_blocks)]
        )

    @torch.compiler.disable
    def update_metrics_s4_ms3_ss(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
        is_neg_sample: Optional[list[torch.Tensor]] = None,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(
                preds[i][None, ...],
                targets[i][None, ...],
                is_neg_sample=(
                    torch.tensor(
                        [is_neg_sample[i]], dtype=torch.bool, device=preds[i].device
                    )
                    if is_neg_sample
                    else None
                ),
            )  # type: ignore

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _log_avsbench_metrics(
        self,
        results: dict[str, torch.Tensor],
        log_prefix: str,
        log_per_class: bool,
        block_postfix: str,
        is_ss: bool,
    ) -> dict[str, torch.Tensor]:
        iou_per_class = results["miou_pc"]
        f_measure_per_class = results["f_measure_pc"]
        cls_count = results["cls_count"].squeeze(-1)

        present_mask = cls_count > 0
        if not present_mask.all():
            missing_classes = [
                idx for idx, present in enumerate(present_mask) if not present
            ]
            rank_zero_warn(
                f"The following classes were not present in the evaluation set: {missing_classes}"
            )

        if log_per_class:
            for class_idx, (iou, f_measure, present) in enumerate(
                zip(iou_per_class, f_measure_per_class, present_mask)
            ):
                if present:
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx + 1}{block_postfix}",
                        float(iou),
                    )
                    self.log(
                        f"metrics/{log_prefix}_f_measure_class_{class_idx + 1}{block_postfix}",
                        float(f_measure),
                    )

        # Only average over present classes
        if present_mask.any():
            if is_ss:
                iou_all = float(iou_per_class[present_mask].mean())
                f_measure_all = float(f_measure_per_class[present_mask].mean())
            else:
                iou_all = float(iou_per_class[present_mask][-1])
                f_measure_all = float(f_measure_per_class[present_mask][-1])
        else:
            iou_all = 0.0
            f_measure_all = 0.0

        self.log(
            f"metrics/{log_prefix}_iou_all{block_postfix}",
            iou_all,
        )
        self.log(
            f"metrics/{log_prefix}_f_measure{block_postfix}",
            f_measure_all,
        )

        return {
            "iou_per_class": iou_per_class,
            "f_measure_per_class": f_measure_per_class,
            "cls_count": cls_count,
        }

    def _log_avsbench_robust_metrics(
        self,
        results: dict[str, torch.Tensor],
        log_prefix: str,
        log_per_class: bool,
        block_postfix: str,
        is_ss: bool,
        iou_per_class: torch.Tensor,
        f_measure_per_class: torch.Tensor,
        cls_count: torch.Tensor,
    ):
        n_cls_count = results["cls_neg_count"].squeeze(-1)
        n_present_mask = n_cls_count > 0
        if not n_present_mask.all():
            missing_classes = [
                idx for idx, present in enumerate(n_present_mask) if not present
            ]
            rank_zero_warn(
                f"The following classes had no negative samples in the evaluation set: {missing_classes}"
            )
        n_iou_per_class = results["miou_neg_pc"]
        n_f_measure_per_class = results["f_measure_neg_pc"]
        g_fpr = results["g_fpr"]

        if log_per_class:
            for class_idx, (iou, f_measure, present) in enumerate(
                zip(n_iou_per_class, n_f_measure_per_class, n_present_mask)
            ):
                if present:
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx + 1}{block_postfix}",
                        float(iou),
                    )
                    self.log(
                        f"metrics/{log_prefix}_f_measure_class_{class_idx + 1}{block_postfix}",
                        float(f_measure),
                    )

        if n_present_mask.any():
            if is_ss:
                n_iou_all = float(n_iou_per_class[n_present_mask].mean())
                n_f_measure_all = float(n_f_measure_per_class[n_present_mask].mean())
                g_fpr_all = float(g_fpr[n_present_mask].mean())
            else:
                n_iou_all = 1.0 - float(n_iou_per_class[1])
                n_f_measure_all = 1.0 - float(n_f_measure_per_class[1])
                g_fpr_all = float(g_fpr[1])
        else:
            n_iou_all = 0.0
            n_f_measure_all = 0.0
            g_fpr_all = 0.0

        self.log(
            f"metrics/{log_prefix}_iou_neg_all{block_postfix}",
            n_iou_all,
        )
        self.log(
            f"metrics/{log_prefix}_f_measure_neg{block_postfix}",
            n_f_measure_all,
        )
        self.log(
            f"metrics/{log_prefix}_g_fpr{block_postfix}",
            g_fpr_all,
        )

        present_mask = cls_count > 0
        if present_mask.any():
            if is_ss:
                iou_all = float(iou_per_class[present_mask].mean())
                f_measure_all = float(f_measure_per_class[present_mask].mean())
            else:
                iou_all = float(iou_per_class[1])
                f_measure_all = float(f_measure_per_class[1])
        else:
            iou_all = 0.0
            f_measure_all = 0.0
        self.log(
            f"metrics/{log_prefix}_g_iou_all{block_postfix}",
            2 * iou_all * n_iou_all / (iou_all + n_iou_all + 1e-8),
        )
        self.log(
            f"metrics/{log_prefix}_g_f_measure{block_postfix}",
            2
            * f_measure_all
            * n_f_measure_all
            / (f_measure_all + n_f_measure_all + 1e-8),
        )

    def _on_eval_epoch_start_s4_ms3_ss(self, log_prefix):
        if self.network.masked_attn_enabled and self.disable_masked_attn_for_val:
            rank_zero_info("Disabling masked attention for evaluation.")
            self.network.masked_attn_enabled = False
            self.metrics = self.metrics[:1]
            self.disabled_masked_attn_for_val = True

    def _on_eval_epoch_end_s4_ms3_ss(
        self, log_prefix, log_per_class=False, is_ss=False
    ):
        for i, metric in enumerate(self.metrics):
            block_postfix = self.block_postfix(i)
            results = metric.compute()
            metric.reset()

            metrics = self._log_avsbench_metrics(
                results, log_prefix, log_per_class, block_postfix, is_ss
            )
            if "miou_neg_pc" in results:
                self._log_avsbench_robust_metrics(
                    results, log_prefix, log_per_class, block_postfix, is_ss, **metrics
                )

        if self.disabled_masked_attn_for_val:
            rank_zero_info("Re-enabling masked attention after evaluation.")
            self.network.masked_attn_enabled = True
            self.disabled_masked_attn_for_val = False

    def _on_eval_end_s4_ms3_ss(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
            )
            rank_zero_info(
                f"{bold_green}F-measure: {self.trainer.callback_metrics[f'metrics/{log_prefix}_f_measure'] * 100:.1f}{reset}"
            )
            if f"metrics/{log_prefix}_iou_neg_all" in self.trainer.callback_metrics:
                rank_zero_info(
                    f"{bold_green}mIoU (neg): {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_neg_all'] * 100:.1f}{reset}"
                )
                rank_zero_info(
                    f"{bold_green}F-measure (neg): {self.trainer.callback_metrics[f'metrics/{log_prefix}_f_measure_neg'] * 100:.1f}{reset}"
                )
                rank_zero_info(
                    f"{bold_green}G-FPR: {self.trainer.callback_metrics[f'metrics/{log_prefix}_g_fpr']:.3f}{reset}"
                )
                rank_zero_info(
                    f"{bold_green}G-mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_g_iou_all'] * 100:.1f}{reset}"
                )
                rank_zero_info(
                    f"{bold_green}G-F-measure: {self.trainer.callback_metrics[f'metrics/{log_prefix}_g_f_measure'] * 100:.1f}{reset}"
                )

    @torch.compiler.disable
    def plot_s4_ms3_ss(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
        to_logger=True,
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        img = F.interpolate(
            img[None, ...].to(torch.float32), MASK_SIZE, mode="bilinear"
        )[0].to(torch.uint8)
        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        if to_logger:
            block_postfix = self.block_postfix(block_idx)
            name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
            self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})
        else:
            return Image.open(buf)

    @torch.compiler.disable
    def scale_img_size_s4_ms3_ss(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_s4_ms3_ss(self, imgs, aembs, imgs_per_audio=1):
        crops, origins, replicated_aembs = [], [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_s4_ms3_ss(img.shape[-2:])
            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))
                replicated_aembs.append(aembs[i // imgs_per_audio])

        return torch.stack(crops), origins, torch.stack(replicated_aembs)

    def revert_window_logits_s4_ms3_ss(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_s4_ms3_ss(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                MASK_SIZE,
                mode="bilinear",
            )[0]
            for (sums, counts) in zip(logit_sums, logit_counts)
        ]

    @staticmethod
    def to_per_pixel_logits_s4_ms3_ss(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_s4_ms3_ss(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    @property
    def use_consistency_loss(self) -> bool:
        return getattr(self, "cosine_consistency_loss", None) is not None

    def compute_consistency_loss(self, mask_logits_per_block, bs, t, losses_all_blocks):
        assert self.use_consistency_loss, "Consistency loss is not enabled."
        for i, mask_logits in enumerate(mask_logits_per_block):
            pred_masks = rearrange(
                mask_logits,
                "(b f) q h w -> b f (q h w)",
                f=t,
            )
            cosine_loss = torch.zeros(bs, device=pred_masks.device)
            total_loss_num = 0
            for frame in range(t - 1):
                total_loss_num += 1
                temp_cosine_loss = self.cosine_consistency_loss(  # type: ignore
                    pred_masks[:, frame, :].squeeze(dim=1),
                    pred_masks[:, frame + 1, :].squeeze(dim=1),
                    torch.ones(bs, device=pred_masks.device),
                )
                exp_weight = torch.exp(-temp_cosine_loss)
                cosine_loss += temp_cosine_loss * exp_weight
            del pred_masks
            del temp_cosine_loss
            del exp_weight
            cosine_loss = cosine_loss.sum() / (bs * total_loss_num)
            block_postfix = self.block_postfix(i)
            losses = {f"loss_consistency{block_postfix}": cosine_loss}
            losses_all_blocks |= losses
        return losses_all_blocks
