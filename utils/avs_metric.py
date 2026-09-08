# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import torch
from torch import Tensor
from torchmetrics import Metric


class AVSMetric(Metric):
    def __init__(
        self, beta2: float = 0.3, pr_num: int = 255, eps: float = 1e-7, **kwargs
    ):
        super().__init__(**kwargs)
        # F-measure
        self.pr_num = pr_num
        self.beta2 = beta2
        self.add_state("avg_f", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_f_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        # MIoU
        self.eps = eps
        self.add_state("total_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_iou_samples", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def _eval_pr(self, y_pred: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate precision and recall across thresholds."""
        device = y_pred.device
        prec = torch.zeros(self.pr_num, device=device)
        recall = torch.zeros(self.pr_num, device=device)
        thlist = torch.linspace(0, 1 - 1e-10, self.pr_num, device=device)

        for i in range(self.pr_num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i] = tp / (y_temp.sum() + 1e-20)
            recall[i] = tp / (y.sum() + 1e-20)

        return prec, recall

    def _calc_fmeasure(self, preds: Tensor, target: Tensor):
        N = preds.shape[0]
        score = torch.zeros(self.pr_num, device=preds.device)
        valid_imgs = 0

        for img_id in range(N):
            # TODO: If we introduce AVSBench-Robust do we change the metric or just remove this check?
            # Skip examples with totally black GTs
            if torch.sum(target[img_id]) == 0.0:
                continue

            prec, recall = self._eval_pr(preds[img_id], target[img_id])
            f_score = (1 + self.beta2) * prec * recall / (self.beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # Handle NaN values
            score += f_score
            valid_imgs += 1

        if valid_imgs > 0:
            score = score / valid_imgs
            self.avg_f += score.max()
            self.total_f_samples += 1

    def _calc_miou(self, preds: Tensor, target: Tensor):
        """Calculate the mean IoU for the given predictions and targets."""
        N = preds.size(0)
        num_pixels = preds.size(-1) * preds.size(-2)

        # Identify samples with no objects (all black targets)
        no_obj_flag = target.sum(2).sum(1) == 0

        # Calculate intersection and union
        inter = (preds * target).sum(2).sum(1)
        union = torch.max(preds, target).sum(2).sum(1)

        # For samples with no objects, calculate intersection as correct background pixels
        inter_no_obj = ((1 - target) * (1 - preds)).sum(2).sum(1)
        inter[no_obj_flag] = inter_no_obj[no_obj_flag]
        union[no_obj_flag] = num_pixels

        # Calculate IoU for each sample and accumulate
        sample_ious = inter / (union + self.eps)
        batch_iou = torch.sum(sample_ious)

        self.total_iou += batch_iou
        self.total_iou_samples += N

    def update(self, preds: Tensor, target: Tensor, **kwargs) -> None:
        """Update metric state with predictions and targets.

        Args:
            preds (Tensor): A int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
                If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
                probabilities/logits into an int tensor.
            target (Tensor): An int tensor of shape ``(N, ...)``.
        """
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)
        assert (
            len(preds.shape) == 3 and preds.shape == target.shape
        ), f"Expected 3D tensors with same shape, got preds: {preds.shape}, target: {target.shape}"

        self._calc_fmeasure(preds, target)
        self._calc_miou(preds, target)

    def compute(self) -> dict[str, Tensor]:
        """Compute the final mean F-measure and IoU scores."""
        results = {}
        if self.total_f_samples > 0:
            results["f_measure"] = self.avg_f / self.total_f_samples
        else:
            results["f_measure"] = torch.tensor(0.0, device=self.avg_f.device)

        if self.total_iou_samples > 0:
            results["miou"] = self.total_iou / self.total_iou_samples
        else:
            results["miou"] = torch.tensor(0.0, device=self.total_iou.device)
        return results
