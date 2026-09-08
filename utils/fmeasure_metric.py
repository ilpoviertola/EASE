# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import torch
from torch import Tensor
from torchmetrics import Metric


class FMeasure(Metric):
    def __init__(self, beta2: float = 0.3, pr_num: int = 255, **kwargs):
        super().__init__(**kwargs)
        self.pr_num = pr_num
        self.beta2 = beta2
        self.add_state("avg_f", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("img_num", default=torch.tensor(0), dist_reduce_fx="sum")

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

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with predictions and targets.

        Args:
            preds (Tensor): A int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
                If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
                probabilities/logits into an int tensor.
            target (Tensor): An int tensor of shape ``(N, ...)``.
        """
        N = preds.shape[0]
        score = torch.zeros(self.pr_num, device=preds.device)
        valid_imgs = 0
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)

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
            self.img_num += 1

    def compute(self) -> Tensor:
        """Compute the final F-measure score."""
        if self.img_num == 0:
            return torch.tensor(0.0, device=self.avg_f.device)
        return self.avg_f / self.img_num
