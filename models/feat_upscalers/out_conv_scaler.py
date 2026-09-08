import torch
from torch import nn
import torch.nn.functional as F


class OutConvScaler(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.out_norm = nn.GroupNorm(32, dim)
        self.out_act = nn.ReLU(True)

    def forward(self, features: torch.Tensor, image: torch.Tensor, **kwargs):
        mask_feature = F.interpolate(
            features, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        mask_feature = image + mask_feature
        mask_feature = self.out_proj(mask_feature)
        mask_feature = self.out_norm(mask_feature)
        mask_feature = self.out_act(mask_feature)
        return mask_feature
