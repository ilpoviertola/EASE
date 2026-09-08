# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import torch
from torch import Tensor
from torchmetrics import Metric


class MaskIoU(Metric):
    def __init__(self, eps: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("total_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with predictions and targets.

        Args:
            preds (Tensor): Predictions of shape [N, H, W] (logits or probabilities)
            target (Tensor): Targets of shape [N, H, W] (binary masks)
        """
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)

        assert (
            len(preds.shape) == 3 and preds.shape == target.shape
        ), f"Expected 3D tensors with same shape, got preds: {preds.shape}, target: {target.shape}"

        N = preds.size(0)
        num_pixels = preds.size(-1) * preds.size(-2)

        # Identify samples with no objects (all black targets)
        no_obj_flag = target.sum(2).sum(1) == 0

        # Apply sigmoid and threshold to get binary predictions
        temp_pred = torch.sigmoid(preds)
        pred_binary = (temp_pred > 0.5).int()

        # Calculate intersection and union
        inter = (pred_binary * target).sum(2).sum(1)
        union = torch.max(pred_binary, target).sum(2).sum(1)

        # For samples with no objects, calculate intersection as correct background pixels
        inter_no_obj = ((1 - target) * (1 - pred_binary)).sum(2).sum(1)
        inter[no_obj_flag] = inter_no_obj[no_obj_flag]
        union[no_obj_flag] = num_pixels

        # Calculate IoU for each sample and accumulate
        sample_ious = inter / (union + self.eps)
        batch_iou = torch.sum(sample_ious)

        self.total_iou += batch_iou
        self.total_samples += N

    def compute(self) -> Tensor:
        """Compute the final mean IoU score."""
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.total_iou.device)
        return self.total_iou / self.total_samples
