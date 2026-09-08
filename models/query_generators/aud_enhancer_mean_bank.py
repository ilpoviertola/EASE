# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# --------------------------------------------------------------

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_


class TokenFeatureEnhancer(nn.Module):
    def __init__(self, embed_dim, k, fea_bank_path):
        super(TokenFeatureEnhancer, self).__init__()
        self.embed_dim = embed_dim
        self.selected_cls_num = k
        self.fea_bank = torch.load(fea_bank_path)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def apply_weights(self, x, weights, dim):
        return x * torch.softmax(weights, dim=dim)

    def forward(self, target_token, fea_bank=None):
        """Enhances the target token using feature banks.

        Args:
            target_token (torch.Tensor): [B, embed_dim]
            fea_bank (torch.Tensor): [71, bank_size, embed_dim]

        Returns:
            torch.Tensor: Enhanced target tokens [B, selected_cls_num, embed_dim]
        """
        if fea_bank is None:
            fea_bank = self.fea_bank.to(target_token.device)

        if target_token.dim() == 3:
            assert (
                target_token.shape[1] == 1
            ), "If target_token has 3 dimensions, the second dimension must be 1."
            target_token = target_token.squeeze(1)

        B, _ = target_token.shape
        _, intra_fea_num, dim = fea_bank.shape

        fea_mean = fea_bank.mean(dim=1)

        distances = torch.cdist(target_token, fea_mean, p=2)

        _, class_indices = torch.topk(
            distances, k=self.selected_cls_num, dim=-1, largest=False
        )

        expanded_class_indices = class_indices.unsqueeze(-1).expand(-1, -1, dim)

        fea_mean_expand = fea_mean.unsqueeze(0).expand(B, -1, dim)
        nearest_class_means = torch.gather(fea_mean_expand, 1, expanded_class_indices)

        return nearest_class_means
