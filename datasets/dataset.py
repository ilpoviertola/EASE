# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import csv
import json
import typing as tp
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TDataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from lightning_fabric.utilities import rank_zero_warn, rank_zero_info

from .transforms import ImageTransforms, VideoTransforms, ImageTransformsEval
from .constants import AVSB_S4_LABELS2IDX, AVSB_MS3_LABELS2IDX


TASK_TYPES = tp.Literal["v1s", "v2", "v1m", "s4", "ms3", "ss"]
TASK_TYPE_MAP = {"s4": "v1s", "ms3": "v1m", "ss": "v2"}
SPLITS = tp.Literal["train", "val", "test"]
MASK_SUFFIX = "png"
bold_green = "\033[1;32m"
bold_red = "\033[1;31m"
reset = "\033[0m"


class Dataset(TDataset):
    def __init__(
        self,
        datapath: str,
        target_parser: tp.Callable,
        data_suffix: str,
        task_type: TASK_TYPES,
        split: SPLITS,
        vggish_emb_fn: str = "vggish_emb.pt",
        audio_t: int = 5,
        read_per_frame: bool = True,
        metapath: tp.Optional[str] = None,
        label2idx: tp.Optional[tp.Union[tp.Dict[str, int], str]] = None,
        transforms: tp.Optional[
            tp.Union[ImageTransforms, VideoTransforms, ImageTransformsEval]
        ] = None,
        ds_to_be_concatenated: bool = False,
    ):
        super().__init__()
        self.ds_to_be_concatenated = ds_to_be_concatenated
        self.vggish_emb_fn = vggish_emb_fn

        task_type = task_type.lower()
        task_type = TASK_TYPE_MAP.get(task_type, task_type)
        if task_type not in TASK_TYPE_MAP.values():
            raise ValueError(f"Invalid task type: {task_type}")

        self.datapath = Path(datapath) / task_type
        assert self.datapath.exists(), f"Data path does not exist: {self.datapath}"
        split = split.lower()
        self.split = split
        if data_suffix.startswith("."):
            data_suffix = data_suffix[1:]
        self.data_suffix = data_suffix.lower()
        self.task = task_type
        self.transforms = transforms
        self.label2idx = self._get_label2idx(label2idx)
        self.target_parser = target_parser
        self.audio_t = audio_t
        self.read_per_frame = read_per_frame
        self.metadata_reader = (
            self._read_per_frame if read_per_frame else self._read_per_sample
        )
        self.metadata = self._get_metadata(metapath)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self, idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any]]:
        loaded, attempts = False, 0
        while not loaded and attempts < 5:
            try:
                meta = self.metadata[idx]
                img, aemb, target, loaded = self._load_datapoint(meta)
            except (FileNotFoundError, ValueError) as e:
                rank_zero_warn(f"{bold_red}{str(e)}{reset}")
                loaded = False
                idx = np.random.randint(0, len(self))
            finally:
                attempts += 1

        if not loaded:
            raise Exception(
                f"Failed to load datapoint after {attempts} attempts. "
                "This might indicate a problem with the dataset."
            )

        return img, aemb, target

    def _load_datapoint(
        self, meta: tp.Dict[str, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any], bool]:
        vggish_emb_path = Path(meta["vggish_emb_path"])
        if not vggish_emb_path.exists():
            raise FileNotFoundError(
                f"VGGish embedding file does not exist: {vggish_emb_path}"
            )

        aemb = torch.load(vggish_emb_path)
        if aemb.shape[0] != self.audio_t:
            raise ValueError(
                f"Audio embedding length {aemb.shape[0]} does not match expected {self.audio_t}."
            )
        if self.read_per_frame:
            assert (
                len(meta["image_path"]) == 1
            ), f"{self.read_per_frame=} but {len(meta['image_path'])=}"
            idx = int(Path(meta["image_path"][0]).stem)
            aemb = aemb[idx : idx + 1]

        imgs = [self._read_image(Path(dp)) for dp in meta["image_path"]]

        masks = [
            self._read_image(Path(mp), is_mask=True, img_shape=imgs[0].shape[-2:])
            for mp in meta["mask_path"]
        ]

        to_parser = {
            "masks": masks,
            "labels": meta["label"],
            "label2idx": self.label2idx,
        }
        masks, labels, mask_to_frame_idx = self.target_parser(to_parser)
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "mask_to_frame_idx": torch.tensor(mask_to_frame_idx),
            "is_neg_sample": torch.tensor([False] * self.audio_t),
            "uids": [f"{meta['uid']}/{Path(mp).stem}" for mp in meta["mask_path"]],
        }
        img = torch.stack(imgs)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, aemb, target, True

    def _read_image(
        self,
        path: Path,
        is_mask: bool = False,
        img_shape: tp.Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Read an image or mask from the given path and return it as a tensor."""
        if not path.exists():
            raise FileNotFoundError(
                f"{'Mask' if is_mask else 'Frame'} does not exist: {path}"
            )

        if is_mask:
            if img_shape is None:
                raise ValueError("Image shape must be provided for mask images.")
            mask = tv_tensors.Mask(Image.open(path), dtype=torch.long)
            if mask.shape[-2:] != img_shape:
                mask = F.resize(
                    mask,
                    list(img_shape),
                    interpolation=F.InterpolationMode.NEAREST,
                )
            return mask

        img = tv_tensors.Image(Image.open(path).convert("RGB"))
        return img

    def _get_metadata(
        self,
        metapath: tp.Optional[str],
        skip_header: bool = True,
    ) -> tp.List[tp.Dict[str, str]]:
        metapath = (
            self.datapath.parent / "metadata.csv"
            if metapath is None
            else Path(metapath)
        )
        assert metapath.exists(), f"Metadata path does not exist: {metapath}"
        metadata = []
        with open(metapath, "r") as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader, None)
            for row in reader:
                if row[-1] == self.task and row[-2] == self.split:
                    metadata += self._read_metadata_row(*row)
        rank_zero_info(
            f"{bold_green}Loaded metadata for {len(metadata)} entries from {self.task} ({self.split}) from {metapath}{reset}"
        )
        return metadata

    def _read_metadata_row(
        self,
        vid: str,
        uid: str,
        s_min: str,
        s_sec: str,
        a_obj: str,
        split: str,
        label: str,
    ) -> tp.List[tp.Dict[str, str]]:
        # read all mask files for the video
        dir_type = "semantic"  # if self.task == "v2" else "rgb"
        return self.metadata_reader(
            uid, a_obj, dir_type, self.datapath, self.data_suffix, self.vggish_emb_fn
        )

    @staticmethod
    def _read_per_sample(
        uid: str,
        a_obj: str,
        dir_type: str,
        datapath: Path,
        data_suffix: str,
        vggish_emb_fn: str,
    ) -> tp.List[tp.Dict[str, tp.Union[str, list[str]]]]:
        """Read per-sample metadata and return corresponding RGB and mask frames.

        Args:
            uid (str): Unique identifier for the video.
            a_obj (str): Audio object label.
            dir_type (str): Type of directory, either "semantic" or "rgb".
            datapath (Path): Path to the dataset directory.
            data_suffix (str): Suffix of the data files (e.g., "jpg", "png").

        Returns:
            tp.List[tp.Dict[str, str]]: List of dictionaries containing per-sample metadata.
        """
        row = {}
        row["uid"] = uid
        row["label"] = a_obj
        mask_paths = [
            f.as_posix()
            for f in datapath.glob(f"{uid}/labels_{dir_type}/*.{MASK_SUFFIX}")
        ]
        row["mask_path"] = sorted(mask_paths, key=lambda x: int(Path(x).stem))
        img_paths = [
            f.as_posix() for f in datapath.glob(f"{uid}/frames/*.{data_suffix}")
        ]
        row["image_path"] = sorted(img_paths, key=lambda x: int(Path(x).stem))
        row["vggish_emb_path"] = datapath / uid / vggish_emb_fn
        return [row]

    @staticmethod
    def _read_per_frame(
        uid: str,
        a_obj: str,
        dir_type: str,
        datapath: Path,
        data_suffix: str,
        vggish_emb_fn: str,
    ) -> tp.List[tp.Dict[str, tp.Union[str, list[str]]]]:
        """Read per-mask frame metadata and return corresponding RGB frame.

        Args:
            uid (str): Unique identifier for the video.
            a_obj (str): Audio object label.
            dir_type (str): Type of directory, either "semantic" or "rgb".
            datapath (Path): Path to the dataset directory.
            data_suffix (str): Suffix of the data files (e.g., "jpg", "png").

        Returns:
            tp.List[tp.Dict[str, str]]: List of dictionaries containing per-frame metadata.
        """
        masks_files = list(datapath.glob(f"{uid}/labels_{dir_type}/*.{MASK_SUFFIX}"))
        rows = []
        for mask_file in masks_files:
            row = {}
            row["uid"] = uid
            row["label"] = a_obj
            row["mask_path"] = [mask_file.as_posix()]
            row["image_path"] = [
                (datapath / f"{uid}/frames/{mask_file.stem}.{data_suffix}").as_posix()
            ]
            row["vggish_emb_path"] = datapath / uid / vggish_emb_fn
            rows.append(row)
        return rows

    def _get_label2idx(
        self,
        label2idx: tp.Optional[tp.Union[tp.Dict[str, int], str]],
    ) -> tp.Dict[str, int]:
        if isinstance(label2idx, dict):
            l2i = label2idx
        elif isinstance(label2idx, str):
            assert Path(
                label2idx
            ).exists(), f"Label2idx path does not exist: {label2idx}"
            with open(label2idx, "r") as f:
                l2i = json.load(f)
        else:
            if self.task == "v1s" and not self.ds_to_be_concatenated:
                l2i = AVSB_S4_LABELS2IDX
            elif self.task == "v1m" and not self.ds_to_be_concatenated:
                l2i = AVSB_MS3_LABELS2IDX
            else:
                l2i_file = self.datapath.parent / "label2idx.json"
                assert l2i_file.exists(), f"Label2idx file does not exist: {l2i_file}"
                with open(l2i_file, "r") as f:
                    l2i = json.load(f)

        if min(l2i.values()) > 0:
            # Ensure labels start from 0
            min_value = min(l2i.values())
            l2i = {k: v - min_value for k, v in l2i.items()}
        return l2i
