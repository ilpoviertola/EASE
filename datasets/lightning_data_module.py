# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from tue-mps/EoMT,
# used under the MIT License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import lightning


class LightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        num_classes: int,
        check_empty_targets: bool,
        vggish_emb_fn: str = "vggish_emb.pt",
        read_per_frame: bool = True,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        neg_audio_p: float = 0.1,
        use_robust_for_testing_if_available: bool = True,
    ) -> None:
        super().__init__()

        self.path = path
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        self.img_size = img_size
        self.num_classes = num_classes
        self.read_per_frame = read_per_frame
        self.neg_audio_p = neg_audio_p
        self.vggish_emb_fn = vggish_emb_fn
        self.prefer_robust = use_robust_for_testing_if_available

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def train_collate(batch):
        imgs, aembs, targets = [], [], []

        for img, aemb, target in batch:
            imgs.append(img)
            aembs.append(aemb)
            targets.append(target)

        return torch.stack(imgs), torch.stack(aembs), targets

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))
