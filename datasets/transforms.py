# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook,
# Inc. and its affiliates, used under the Apache 2.0 License and
# from tue-mps/EoMT, used under the MIT License.
# ---------------------------------------------------------------

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor
from torch import nn, Tensor
from typing import Any, Union

from .constants import MASK_SIZE


class ImageTransforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)
        return img, target


class ImageTransformsAVSB(ImageTransforms):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        use_crop_and_resize: bool = False,
    ):
        super().__init__(
            img_size,
            color_jitter_enabled,
            scale_range,
            max_brightness_delta,
            max_contrast_factor,
            saturation_factor,
            max_hue_delta,
        )
        self.use_crop_and_resize = use_crop_and_resize
        if not use_crop_and_resize:
            self.random_crop = None
        self.resize = T.Resize(img_size)
        self.resize_mask = T.Resize(
            MASK_SIZE, interpolation=T.InterpolationMode.NEAREST
        )

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)

        if torch.rand(()) < 0.5 and self.use_crop_and_resize:
            img, target = self.pad(img, target)
            img, target = self.random_crop(img, target)  # type: ignore
            target["masks"] = self.resize_mask(target["masks"])
        else:
            img = self.resize(img)
            target["masks"] = self.resize_mask(target["masks"])
        return img, target


class ImageTransformsEval(nn.Module):
    def __init__(self, img_size: tuple[int, int]):
        super().__init__()
        self.img_size = img_size
        self.resize = T.Resize(img_size)
        self.resize_mask = T.Resize(
            MASK_SIZE, interpolation=T.InterpolationMode.NEAREST_EXACT
        )

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img = self.resize(img)
        target["masks"] = self.resize_mask(target["masks"])
        return img, target


class VideoTransforms(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("VideoTransforms is not implemented yet.")
