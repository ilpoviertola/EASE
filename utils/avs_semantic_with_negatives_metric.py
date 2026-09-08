# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from math import prod

import torch
from torch import Tensor
from .avs_semantic_metric import AVSSMetric


class AVSSMetricWithNegatives(AVSSMetric):
    def __init__(self, beta2: float = 0.3, t: int = 1, num_cls: int = 71, **kwargs):
        super().__init__(beta2=beta2, t=t, num_cls=num_cls, **kwargs)
        self.add_state(
            "miou_neg_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )
        self.add_state(
            "fscore_neg_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )
        self.add_state(
            "cls_neg_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )
        self.add_state(
            "fpr_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )

    def _calc_fpr(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Calculate false positive rate."""
        preds = torch.argmax(preds, dim=1)  # (B, H, W)
        fpr = torch.zeros(self.num_cls, device=preds.device, dtype=torch.float32)

        for cls in range(self.num_cls):
            # Pixels where GT is NOT this class (negatives)
            neg_mask = targets != cls
            # Of those, how many are predicted as this class? (False Positives)
            fp = ((preds == cls) & neg_mask).sum()
            # Total negatives for this class
            tn_fp = neg_mask.sum()
            fpr[cls] = fp.float() / (tn_fp.float() + 1e-8)  # avoid div by zero

        return fpr

    def update(self, preds: Tensor, targets: Tensor, **kwargs):
        """Update metric state with predictions and targets.

        Args:
            preds (Tensor): A int tensor of shape ``(B*T, ...)`` or float tensor of shape ``(B*T, C, ..)``.
                If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
                probabilities/logits into an int tensor.
            targets (Tensor): An int tensor of shape ``(B*T, ...)``.
        """
        is_neg_sample = kwargs.get(
            "is_neg_sample",
            torch.zeros(preds.shape[0], dtype=torch.bool, device=preds.device),
        )
        assert preds.dim() == targets.dim() + 1, f"{preds.dim()} != {targets.dim() + 1}"
        assert preds.size(1) == self.num_cls, f"{preds.size(1)} != {self.num_cls}"

        preds_pos = preds[~is_neg_sample]
        targets_pos = targets[~is_neg_sample]
        if preds_pos.shape[0] > 0:
            ious, f_scores, cls_count = self._metric_func(preds_pos, targets_pos)
            self.miou_pc += ious
            self.fscore_pc += f_scores
            self.cls_pc += cls_count

        preds_neg = preds[is_neg_sample]
        targets_neg = targets[is_neg_sample]
        if preds_neg.shape[0] > 0:
            ious, f_scores, cls_count = self._metric_func(preds_neg, targets_neg)
            self.miou_neg_pc += ious
            self.fscore_neg_pc += f_scores
            self.cls_neg_pc += cls_count
            # For negative samples the prediction should be blank (all zeros)
            # but AVSBR calculates negative F-score and IoU using the original mask as target
            # (thus F-score and IoU get better while approaching zero)
            # Here we calculate the global FPR where the target is all zeros
            # (since every activated pixel is false positive in negative samples)
            fpr = self._calc_fpr(preds_neg, torch.zeros_like(targets_neg))
            self.fpr_pc += fpr

    def compute(self) -> dict[str, Tensor]:
        """Compute the final mean IoU and F-score."""
        results = super().compute()
        if self.cls_neg_pc.sum() > 0:
            _miou_neg_pc = self.miou_neg_pc / self.cls_neg_pc
            _miou_neg_pc[torch.isnan(_miou_neg_pc)] = 0

            results["miou_neg_pc"] = _miou_neg_pc
            _f_measure_neg_pc = self.fscore_neg_pc / self.cls_neg_pc
            _f_measure_neg_pc[torch.isnan(_f_measure_neg_pc)] = 0
            results["f_measure_neg_pc"] = _f_measure_neg_pc
            results["cls_neg_count"] = self.cls_neg_pc
            results["g_fpr"] = self.fpr_pc / self.cls_neg_pc
            results["g_fpr"][torch.isnan(results["g_fpr"])] = 0
        return results
