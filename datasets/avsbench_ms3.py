# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import typing as tp
from pathlib import Path

from torch import Tensor
from lightning_fabric.utilities import rank_zero_info

from .avsbr_dataset import AVSBRDataset
from .dataset import Dataset
from .transforms import ImageTransformsAVSB, ImageTransformsEval
from .avsbench_s4 import AVSBenchS4
from .lightning_data_module import LightningDataModule


class AVSBenchMS3(AVSBenchS4):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (512, 512),
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        num_classes: int = 23,
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
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
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
        target: tp.Dict[str, tp.Any], **kwargs: tp.Any
    ) -> tp.Tuple[tp.List[Tensor], tp.List[int], tp.List[int]]:
        _masks, _labels, _mask_to_frame_idx = [], [], []
        for idx, mask in enumerate(target["masks"]):
            binary_mask = (mask != 0).any(dim=0)
            if binary_mask.sum() == 0:  # no sounding object
                _masks.append(~binary_mask)  # everything is background
                _labels.append(0)
                _mask_to_frame_idx.append(idx)
                continue
            _masks.append(binary_mask)
            _labels.append(1)
            _mask_to_frame_idx.append(idx)
            _masks.append(~binary_mask)
            _labels.append(0)
            _mask_to_frame_idx.append(idx)
        return _masks, _labels, _mask_to_frame_idx

    def setup(self, stage: tp.Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "data_suffix": "jpg",
            "task_type": "ms3",
            "target_parser": self.target_parser,
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
