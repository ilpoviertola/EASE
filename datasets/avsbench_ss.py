# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import typing as tp
from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader
from lightning_fabric.utilities import rank_zero_info

from .avsbr_dataset import AVSBRDataset
from .dataset import Dataset
from .transforms import ImageTransformsAVSB, ImageTransformsEval
from .lightning_data_module import LightningDataModule


class AVSBenchSS(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (512, 512),
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        num_classes: int = 71,
        read_per_frame: bool = True,
        neg_audio_p: float = 0.0,
        vggish_emb_fn: str = "vggish_emb.pt",
        use_robust_for_testing_if_available: bool = True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            num_classes=num_classes,
            check_empty_targets=True,
            read_per_frame=read_per_frame,
            neg_audio_p=neg_audio_p,
            vggish_emb_fn=vggish_emb_fn,
            use_robust_for_testing_if_available=use_robust_for_testing_if_available,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.transforms = ImageTransformsAVSB(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        self.transforms_eval = ImageTransformsEval(img_size=img_size)

    @staticmethod
    def target_parser(
        target: tp.Dict[str, tp.Any], **kwargs
    ) -> tp.Tuple[tp.List[Tensor], tp.List[int], tp.List[int]]:
        _masks, _labels, _mask_to_frame_idx = [], [], []
        label2idx = target["label2idx"]
        for idx, mask in enumerate(target["masks"]):
            mask = mask.squeeze(0)
            for label_id in mask.unique():
                label_id = label_id.item()
                if label_id not in label2idx.values():
                    continue
                _masks.append(mask == label_id)
                _labels.append(label_id)
                _mask_to_frame_idx.append(idx)

        return _masks, _labels, _mask_to_frame_idx

    def setup(self, stage: tp.Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "data_suffix": "jpg",
            "task_type": "ss",
            "target_parser": self.target_parser,
            "audio_t": 10,
            "read_per_frame": self.read_per_frame,
            "vggish_emb_fn": self.vggish_emb_fn,
        }
        if self.neg_audio_p > 0.0:
            ds = test_ds = AVSBRDataset
            dataset_kwargs["neg_audio_p"] = self.neg_audio_p
        else:
            ds = Dataset
            if hasattr(self, "negative_metapath") and self.prefer_robust:
                test_ds = AVSBRDataset
                dataset_kwargs["negative_metapath"] = getattr(self, "negative_metapath")
            elif (Path(self.path) / "neg_metadata.csv").exists() and self.prefer_robust:
                test_ds = AVSBRDataset
            else:
                test_ds = Dataset
        rank_zero_info(f"Using dataset class {ds.__name__} for training/validation.")
        rank_zero_info(f"Using dataset class {test_ds.__name__} for testing.")

        self.train_dataset = ds(
            datapath=self.path,
            split="train",
            transforms=self.transforms,
            **dataset_kwargs,
        )
        self.val_dataset = ds(
            datapath=self.path,
            split="val",
            transforms=self.transforms_eval,
            **dataset_kwargs,
        )
        self.test_dataset = test_ds(
            datapath=self.path,
            split="test",
            transforms=self.transforms_eval,
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
