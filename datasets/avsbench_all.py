# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import typing as tp
from pathlib import Path

from torch.utils.data import ConcatDataset
from lightning_fabric.utilities import rank_zero_info

from .dataset import Dataset
from .avsbench_ss import AVSBenchSS
from .avsbr_dataset import AVSBRDataset
from .lightning_data_module import LightningDataModule


class AVSBenchAll(AVSBenchSS):
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
        assert read_per_frame, "AVSBenchAll requires read_per_frame=True"
        super().__init__(
            path=path,
            num_workers=num_workers,
            batch_size=batch_size,
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            num_classes=num_classes,
            read_per_frame=read_per_frame,
            neg_audio_p=neg_audio_p,
            vggish_emb_fn=vggish_emb_fn,
            use_robust_for_testing_if_available=use_robust_for_testing_if_available,
        )

    def _resolve_dataset(
        self, dataset_kwargs: tp.Dict[str, tp.Any]
    ) -> tp.Tuple[tp.Type[Dataset], tp.Type[Dataset], tp.Dict[str, tp.Any]]:
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

        return ds, test_ds, dataset_kwargs

    def setup(self, stage: tp.Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "data_suffix": "jpg",
            "task_type": None,  # to be set per sub-dataset
            "audio_t": -1,  # to be set per sub-dataset
            "target_parser": self.target_parser,
            "read_per_frame": self.read_per_frame,
            "vggish_emb_fn": self.vggish_emb_fn,
            "ds_to_be_concatenated": True,
        }

        ds, test_ds, dataset_kwargs = self._resolve_dataset(dataset_kwargs)
        rank_zero_info(f"Using dataset class {ds.__name__} for training/validation.")
        rank_zero_info(f"Using dataset class {test_ds.__name__} for testing.")

        train_datasets, val_datasets, test_datasets = [], [], []
        for task, audio_t in zip(["v1s", "v1m", "v2"], [5, 5, 10]):
            dataset_kwargs["task_type"] = task
            dataset_kwargs["audio_t"] = audio_t

            train_dataset = ds(
                datapath=self.path,
                split="train",
                transforms=self.transforms,
                **dataset_kwargs,
            )
            val_dataset = ds(
                datapath=self.path,
                split="val",
                transforms=self.transforms_eval,
                **dataset_kwargs,
            )
            test_dataset = test_ds(
                datapath=self.path,
                split="test",
                transforms=self.transforms_eval,
                **dataset_kwargs,
            )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)
        self.test_dataset = ConcatDataset(test_datasets)

        return self
