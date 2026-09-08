# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import csv
import ast
import json
import typing as tp
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TDataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.nn.functional import pad
from lightning_fabric.utilities import rank_zero_warn, rank_zero_info

from .transforms import ImageTransforms, VideoTransforms, ImageTransformsEval
from .constants import VPO_LABELS2IDX, VPO_IDX2LABELS, VPO_OOD_CLASSES


TASK_TYPES = tp.Literal["VPO-SS", "VPO-MS", "VPO-MSMI"]
SPLITS = tp.Literal["train", "test"]
MASK_SUFFIX = "png"
bold_green = "\033[1;32m"
bold_red = "\033[1;31m"
reset = "\033[0m"


class VPODataset(TDataset):
    def __init__(
        self,
        datapath: str,
        target_parser: tp.Callable,
        data_suffix: str,
        task_type: TASK_TYPES,
        split: SPLITS,
        vggish_emb_fn: str = "vggish_emb.pt",
        audio_t: int = 3,
        read_per_frame: bool = True,
        metapath: tp.Optional[str] = None,
        label2idx: tp.Optional[tp.Union[tp.Dict[str, int], str]] = None,
        transforms: tp.Optional[
            tp.Union[ImageTransforms, VideoTransforms, ImageTransformsEval]
        ] = None,
        ds_to_be_concatenated: bool = False,
        ood: bool = False,
    ):
        super().__init__()

        self.ds_to_be_concatenated = ds_to_be_concatenated
        self.vggish_emb_fn = vggish_emb_fn
        self.ood = ood

        task_type = task_type.upper()
        allowed_task_types = [t.upper() for t in tp.get_args(TASK_TYPES)]
        if task_type not in allowed_task_types:
            raise ValueError(
                f"Invalid task type: {task_type}. Allowed: {allowed_task_types}"
            )

        self.datapath = Path(datapath) / f"{task_type}"
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

        imgs = [self._read_image(Path(dp)) for dp in meta["image_path"]]

        masks = [
            self._read_image(Path(mp), is_mask=True, img_shape=imgs[0].shape[-2:])
            for mp in meta["mask_path"]
        ]

        # Pad aemb, imgs, and masks to length of 5
        if not self.read_per_frame:
            assert len(imgs) == 1, "Expected a single frame."
            assert len(masks) == 1, "Expected a single mask."
            aemb = aemb.mean(dim=0, keepdim=True)  # (1, emb_dim)
            aemb = pad(aemb, (0, 0, 0, 4))  # type: ignore[attr-defined]
            imgs += [torch.zeros_like(imgs[0]) for _ in range(4)]
            masks += [torch.zeros_like(masks[0]) for _ in range(4)]

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

        if not self.read_per_frame:
            target["is_neg_sample"] = torch.tensor([False] * 5)

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
            self.datapath.parent / "vpo_data_clean.csv"
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
                if self.ood:
                    # Include only OOD classes
                    ood_labels = ast.literal_eval(row[3])
                    if (
                        any(
                            VPO_IDX2LABELS[label] in VPO_OOD_CLASSES
                            for label in ood_labels
                        )
                        and row[-1] == self.task
                        and row[-2] == self.split
                    ):
                        metadata += self._read_metadata_row(*row)
                else:
                    if row[-1] == self.task and row[-2] == self.split:
                        metadata += self._read_metadata_row(*row)
        rank_zero_info(
            f"{bold_green}Loaded metadata for {len(metadata)} entries from {self.task} ({self.split}) from {metapath}{reset}"
        )
        return metadata

    def _read_metadata_row(
        self,
        uid: str,
        frame_path: str,
        mask_path: str,
        category: str,
        split: str,
        type: str,
    ) -> tp.List[tp.Dict[str, tp.Union[str, list[str]]]]:
        # read all mask files for the video
        dir_type = "labels_semantic"
        return self._read_per_frame(
            uid,
            category,
            frame_path,
            dir_type,
            mask_path,
            self.datapath,
            self.vggish_emb_fn,
        )

    @staticmethod
    def _read_per_frame(
        uid: str,
        a_obj: str,
        frame_path: str,
        dir_type: str,
        mask_path: str,
        datapath: Path,
        vggish_emb_fn: str,
    ) -> tp.List[tp.Dict[str, tp.Union[str, list[str]]]]:
        """Read per-mask frame metadata and return corresponding RGB frame.

        Args:
            uid (str): Unique identifier for the video.
            a_obj (str): Audio object label.
            frame_path (str): Path to the frame image.
            dir_type (str): Directory type (e.g., "semantic").
            mask_path (str): Path to the mask image.
            datapath (Path): Path to the dataset directory.
            vggish_emb_fn (str): Filename of the VGGish embedding file.

        Returns:
            tp.List[tp.Dict[str, str]]: List of dictionaries containing per-frame metadata.
        """
        row = {}
        uid = uid.rsplit("/", 1)[-1]
        row["uid"] = uid
        row["label"] = [VPO_IDX2LABELS[a_obj] for a_obj in ast.literal_eval(a_obj)]
        row["mask_path"] = [(datapath / f"{uid}/{dir_type}/{mask_path}").as_posix()]
        row["image_path"] = [(datapath / f"{uid}/frames/{frame_path}").as_posix()]
        row["vggish_emb_path"] = datapath / uid / vggish_emb_fn
        return [row]

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
            l2i = VPO_LABELS2IDX

        if min(l2i.values()) > 0:
            # Ensure labels start from 0
            min_value = min(l2i.values())
            l2i = {k: v - min_value for k, v in l2i.items()}
        return l2i
