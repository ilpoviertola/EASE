import torch
from torch import nn


class CosineSimilarity(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        audio_dim: int = 256,
        visual_dim: int = 512,
    ):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        if audio_dim != dim:
            self.audio_proj = nn.Linear(audio_dim, dim)
        else:
            self.audio_proj = nn.Identity()  # type: ignore

        self.visual_pool = nn.AdaptiveAvgPool1d(1)
        if visual_dim != dim:
            self.visual_proj = nn.Linear(visual_dim, dim)
        else:
            self.visual_proj = nn.Identity()  # type: ignore

    def forward(
        self, audio_feat: torch.Tensor, visual_feat: torch.Tensor
    ) -> torch.Tensor:
        audio_feat = self.audio_proj(audio_feat)[:, 0, ...]
        visual_feat = self.visual_proj(visual_feat)

        visual_feat = self.visual_pool(visual_feat.permute(0, 2, 1))[..., 0]

        assert (
            audio_feat.shape == visual_feat.shape
        ), f"{audio_feat.shape}, {visual_feat.shape}"
        similarity = self.cosine_similarity(audio_feat, visual_feat)

        return similarity
