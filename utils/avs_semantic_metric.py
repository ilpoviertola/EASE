# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import Callable

import torch
from torch import Tensor
from torchmetrics import Metric


class AVSSMetric(Metric):
    def __init__(self, beta2: float = 0.3, t: int = 1, num_cls: int = 71, **kwargs):
        super().__init__(**kwargs)
        self.beta2 = beta2
        self.t = t
        self.num_cls = num_cls
        self.add_state(
            "miou_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )
        self.add_state(
            "fscore_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )
        self.add_state(
            "cls_pc", default=torch.zeros((self.num_cls)), dist_reduce_fx="sum"
        )

        self._metric_func: Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]] = (
            self._calc_miou_fscore if num_cls > 2 else self._calc_miou_fscore_binary
        )

    def _calc_miou_fscore(self, preds: Tensor, targets: Tensor):
        """Calculate mean IoU and F-score."""
        mini, maxi, bins = 1, self.num_cls, self.num_cls
        targets = targets.float()
        preds = torch.argmax(preds, dim=1) + 1
        targets = targets.float() + 1
        preds = preds.float() * (targets > 0).float()
        intersection = preds * (preds == targets).float()
        cls_count = torch.zeros(self.num_cls, device=preds.device)
        ious = torch.zeros(self.num_cls, device=preds.device, dtype=torch.float32)
        f_scores = torch.zeros(self.num_cls, device=preds.device, dtype=torch.float32)

        vid_miou_list = []
        for i in range(targets.size(0)):
            area_inter = torch.histc(
                intersection[i], bins=bins, min=mini, max=maxi
            )  # TP
            area_pred = torch.histc(preds[i], bins=bins, min=mini, max=maxi)  # TP + FP
            area_lab = torch.histc(targets[i], bins=bins, min=mini, max=maxi)  # TP + FN
            area_union = area_pred + area_lab - area_inter
            assert (
                torch.sum(area_inter > area_union).item() == 0
            ), "Intersection area should be smaller than Union area"
            iou = (
                1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
            )
            ious += iou
            cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

            precision = area_inter / area_pred
            recall = area_inter / area_lab
            fscore = (
                (1 + self.beta2)
                * precision
                * recall
                / (self.beta2 * precision + recall)
            )
            fscore[torch.isnan(fscore)] = 0.0
            f_scores += fscore

            vid_miou_list.append(torch.sum(iou) / (torch.sum(iou != 0).float()))

        return ious, f_scores, cls_count

    def _calc_miou_fscore_binary(self, preds: Tensor, targets: Tensor):
        """
        Calculate mean IoU and F-score for binary predictions
        to match AVSBench metric calculation scheme for S4 and MS3.

        Args:
            preds (Tensor): A probability or logit tensor of shape
                ``(B*T, 2, H, W)``, ``(B*T, 1, H, W)`` or ``(B*T, H, W)``.
            targets (Tensor): An int tensor of shape ``(B*T, H, W)``.
        Returns:
            tuple[Tensor, Tensor, Tensor]: ious, f_scores, cls_count
        """

        def _eval_pr(y_pred, y, num):
            prec = torch.zeros(num, device=preds.device)
            recall = torch.zeros(num, device=preds.device)
            thlist = torch.linspace(0, 1 - 1e-10, num, device=preds.device)
            for i in range(num):
                y_temp = (y_pred >= thlist[i]).float()
                tp = (y_temp * y).sum()
                prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
            return prec, recall

        cls_count = torch.zeros(self.num_cls, device=preds.device)
        ious = torch.zeros(self.num_cls, device=preds.device, dtype=torch.float32)
        f_scores = torch.zeros(self.num_cls, device=preds.device, dtype=torch.float32)

        if preds.dim() == 4 and preds.size(1) == 2:
            pred_binary = torch.argmax(preds, dim=1)
            preds = preds[:, 1, :, :] - preds[:, 0, :, :]
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
        elif preds.dim() == 4 and preds.size(1) == 1:
            preds = preds[:, 0, :, :]
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            pred_binary = (preds > 0.5).long()
        elif preds.dim() == 3:
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            pred_binary = (preds > 0.5).long()
        else:
            raise ValueError(f"Invalid preds shape: {preds.shape}")

        assert (
            pred_binary.shape == targets.shape
        ), f"{pred_binary.shape} != {targets.shape}"
        N, H, W = targets.shape
        num_pixels = H * W
        no_obj_flag = targets.sum(dim=2).sum(dim=1) == 0

        # mIoU calculation
        inter = (pred_binary * targets).sum(2).sum(1)
        union = torch.max(pred_binary, targets).sum(2).sum(1)

        inter_no_obj = ((1 - targets) * (1 - pred_binary)).sum(2).sum(1)
        inter[no_obj_flag] = inter_no_obj[no_obj_flag]
        union[no_obj_flag] = num_pixels

        iou = torch.sum(inter / (union + 1e-7)) / N
        ious[0] = 0.0  # background class (not used)
        ious[1] = iou
        cls_count[0] = 1
        cls_count[1] = 1

        # F-score calculation
        avg_f, img_num = 0.0, 0
        score = torch.zeros(255, device=preds.device)  # 255 is AVSBench max threshold
        targets = targets.float()
        for img_id in range(N):
            # examples with totally black GTs are out of consideration
            if torch.mean(targets[img_id]) == 0.0:
                continue
            prec, recall = _eval_pr(preds[img_id], targets[img_id], 255)
            f_score = (1 + self.beta2) * prec * recall / (self.beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for Nan
            avg_f += f_score
            img_num += 1
            score = avg_f / img_num  # type: ignore

        f_scores[0] = 0.0  # background class (not used)
        f_scores[1] = score.max()
        cls_count[0] = 1
        cls_count[1] = 1

        return ious, f_scores, cls_count

    def update(self, preds: Tensor, targets: Tensor, **kwargs):
        """Update metric state with predictions and targets.

        Args:
            preds (Tensor): A int tensor of shape ``(B*T, C, ..)``.
                We apply ``torch.argmax`` along the ``C`` dimension to automatically convert
                probabilities/logits into an int tensor.
            targets (Tensor): An int tensor of shape ``(B*T, ...)``.
        Returns:
            tuple[Tensor, Tensor, Tensor]: ious, f_scores, cls_count
        """
        assert preds.dim() == targets.dim() + 1, f"{preds.dim()} != {targets.dim() + 1}"
        assert preds.size(1) == self.num_cls, f"{preds.size(1)} != {self.num_cls}"
        ious, f_scores, cls_count = self._metric_func(preds, targets)
        self.miou_pc += ious
        self.fscore_pc += f_scores
        self.cls_pc += cls_count

    def compute(self) -> dict[str, Tensor]:
        """Compute the final mean IoU and F-score."""
        results = {}
        if self.cls_pc.sum() > 0:
            _miou_pc = self.miou_pc / self.cls_pc
            _miou_pc[torch.isnan(_miou_pc)] = 0
            results["miou_pc"] = _miou_pc
            _f_measure_pc = self.fscore_pc / self.cls_pc
            _f_measure_pc[torch.isnan(_f_measure_pc)] = 0
            results["f_measure_pc"] = _f_measure_pc
            results["cls_count"] = self.cls_pc
        else:
            results["miou_pc"] = torch.zeros(self.num_cls, device=self.miou_pc.device)
            results["f_measure_pc"] = torch.zeros(
                self.num_cls, device=self.fscore_pc.device
            )
            results["cls_count"] = torch.zeros(self.num_cls, device=self.cls_pc.device)

        return results
