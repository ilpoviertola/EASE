# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import csv
import random
import typing as tp
from pathlib import Path

import torch
from torchvision import tv_tensors
from lightning_fabric.utilities import rank_zero_info

from .dataset import (
    Dataset,
    TASK_TYPES,
    SPLITS,
    bold_green,
    reset,
)
from .transforms import ImageTransforms, VideoTransforms, ImageTransformsEval
from .constants import (
    AVSB_LABELS2CATEGORY,
    OFFSCREEN_AUDIO_PROB,
    SILENT_AUDIO_PROB,
    NOISE_AUDIO_PROB,
)


class AVSBRDataset(Dataset):
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
        neg_audio_p: float = 0.2,
        metapath: tp.Optional[str] = None,
        negative_metapath: tp.Optional[str] = None,
        label2idx: tp.Optional[tp.Union[tp.Dict[str, int], str]] = None,
        transforms: tp.Optional[
            tp.Union[ImageTransforms, VideoTransforms, ImageTransformsEval]
        ] = None,
    ):
        super().__init__(
            datapath=datapath,
            target_parser=target_parser,
            data_suffix=data_suffix,
            task_type=task_type,
            split=split,
            vggish_emb_fn=vggish_emb_fn,
            audio_t=audio_t,
            read_per_frame=read_per_frame,
            metapath=metapath,
            label2idx=label2idx,
            transforms=transforms,
        )

        self.neg_audio_p = neg_audio_p
        self.weights = [OFFSCREEN_AUDIO_PROB, SILENT_AUDIO_PROB, NOISE_AUDIO_PROB]
        self.elements = ["offscreen", "silent", "noise"]

        if neg_audio_p > 0.0:
            self.uid_to_neg_meta = self._get_neg_metadata(negative_metapath)
            assert (
                self.datapath / "negative_samples"
            ).exists(), "Negative samples directory does not exist."
            self.negative_samples_path = self.datapath / "negative_samples"

    def _get_neg_metadata(
        self, neg_metapath: tp.Optional[str], skip_header: bool = True
    ) -> tp.Dict[str, tp.List[str]]:
        if self.split == "train":
            return {}

        metapath = (
            self.datapath.parent / "neg_metadata.csv"
            if neg_metapath is None
            else Path(neg_metapath)
        )
        if not metapath.exists():
            raise FileNotFoundError(
                f"Negative metadata file does not exist: {metapath}. "
                "Please generate it using utils/generate_negative_sample_meta.py"
            )

        uid_to_neg_meta = {}
        with metapath.open("r") as f:
            reader = csv.DictReader(f)
            if skip_header:
                next(reader, None)  # skip the header
            for row in reader:
                if row["label"] == self.task and row["split"] == self.split:
                    uid = row.pop("uid")
                    uid_to_neg_meta[uid] = row

        rank_zero_info(
            f"{bold_green}Loaded negative metadata for {len(uid_to_neg_meta)} entries from {metapath}{reset}"
        )
        return uid_to_neg_meta

    def _use_neg_audio(self, deterministic: bool = False) -> bool:
        if deterministic:
            return self.neg_audio_p > 0.0
        return random.random() < self.neg_audio_p

    def _get_neg_audio(
        self, meta: tp.Dict[str, str]
    ) -> tp.Tuple[tp.Optional[str], tp.Optional[str]]:
        if self.split == "train":
            return self._get_neg_audio_train(meta["label"])
        else:
            return self._get_neg_audio_eval(meta)

    def _get_neg_audio_eval(
        self, meta: tp.Dict[str, str]
    ) -> tp.Tuple[tp.Optional[str], tp.Optional[str]]:
        if meta["uid"] not in self.uid_to_neg_meta:
            return None, None

        neg_meta = self.uid_to_neg_meta[meta["uid"]]
        vggish_emb_path = Path(neg_meta["neg_vggish_emb_dir"]) / self.vggish_emb_fn  # type: ignore
        label = neg_meta["neg_a_obj"]  # type: ignore
        return vggish_emb_path, label

    def _get_neg_audio_train(
        self, current_cls: str
    ) -> tp.Tuple[tp.Optional[str], tp.Optional[str]]:
        choice = random.choices(self.elements, weights=self.weights, k=1)[0]
        if choice == "offscreen":
            return self._get_offscreen_audio(current_cls)
        elif choice == "silent":
            return self._get_silent_audio()
        elif choice == "noise":
            return self._get_noise_audio()
        else:
            raise ValueError(f"Unknown choice for negative audio: {choice}")

    def _get_noise_audio(self) -> tp.Tuple[str, str]:
        d = random.choice(list(self.negative_samples_path.glob("random_audio_*")))
        return (d / self.vggish_emb_fn).as_posix(), "background"

    def _get_silent_audio(self) -> tp.Tuple[str, str]:
        return (
            self.negative_samples_path / "silent_audio" / self.vggish_emb_fn
        ).as_posix(), "background"

    def _get_offscreen_audio(
        self, current_cls: str
    ) -> tp.Tuple[tp.Optional[str], tp.Optional[str]]:
        # current_cls can have multiple labels divided by "_"
        current_cls = current_cls.split("_")  # type: ignore
        # get categories for all of the labels
        current_ctg = set(
            [
                AVSB_LABELS2CATEGORY[label]
                for label in current_cls
                if label not in ["background", "off-the-screen"]
            ]
        )
        sample = random.sample(self.metadata, 1)[0]
        # get categories for the sample
        sample_ctg = set(
            [
                AVSB_LABELS2CATEGORY[label]
                for label in sample["label"].split("_")
                if label not in ["background", "off-the-screen"]
            ]
        )
        retries = 0
        while sample_ctg.intersection(current_ctg):
            sample = random.sample(self.metadata, 1)[0]
            sample_ctg = set(
                [
                    AVSB_LABELS2CATEGORY[label]
                    for label in sample["label"].split("_")
                    if label not in ["background", "off-the-screen"]
                ]
            )
            retries += 1
            if retries >= 20:
                return None, None
        return sample["vggish_emb_path"], sample["label"]

    def _load_datapoint(
        self, meta: tp.Dict[str, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any], bool]:
        vggish_emb_path = Path(meta["vggish_emb_path"])
        if not vggish_emb_path.exists():
            raise FileNotFoundError(
                f"VGGish embedding file does not exist: {vggish_emb_path}"
            )

        using_neg_audio = False
        if self._use_neg_audio(deterministic=(self.split != "train")):
            _vggish_emb_path, _label = self._get_neg_audio(meta)
            if _vggish_emb_path is not None and _label is not None:
                vggish_emb_path = _vggish_emb_path
                meta["label"] = _label
                using_neg_audio = True

        aemb = torch.load(vggish_emb_path)
        if aemb.shape[0] != self.audio_t:
            raise ValueError(
                f"Audio embedding length {aemb.shape[0]} does not match expected {self.audio_t}."
            )

        imgs = [self._read_image(Path(dp)) for dp in meta["image_path"]]
        if using_neg_audio and self.split == "train":
            # During training with negative audio, use blank masks
            # During evaluation use the original masks to get proper metrics (AVSBench-Robust)
            masks = [
                tv_tensors.Mask(torch.zeros((1, imgs[i].shape[-2], imgs[i].shape[-1])))
                for i in range(len(meta["mask_path"]))
            ]
        else:
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
            "is_neg_sample": torch.tensor([using_neg_audio] * self.audio_t),
            "uids": [f"{meta['uid']}/{Path(mp).stem}" for mp in meta["mask_path"]],
        }
        img = torch.stack(imgs)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, aemb, target, True
