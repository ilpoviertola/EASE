import torch.nn as nn


class LinearAudioFeatureEnhancer(nn.Module):
    def __init__(self, aud_dim=128, embed_dim=320):
        super(LinearAudioFeatureEnhancer, self).__init__()
        self.linear = nn.Linear(aud_dim, embed_dim)

    def forward(self, aud_fea):
        """
        Args:
            aud_fea: [B, H*W, in_features] -> audio features
        Returns:
            enhanced_aud_fea: [B, H*W, out_features] -> enhanced audio features
        """
        enhanced_aud_fea = self.linear(aud_fea)
        return enhanced_aud_fea
