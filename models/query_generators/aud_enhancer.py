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

        # Define common MLP structures
        def build_mlp():
            return nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

        def build_offset():
            return nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())

        self.edge_fc_s1 = build_mlp()
        self.offset_s1 = build_offset()

        self.edge_fc_s2 = build_mlp()
        self.offset_s2 = build_offset()

        self.fusion_s1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fusion_s2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.apply(self._init_weights)

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

        edge_features = self.edge_fc_s1(nearest_class_means - target_token.unsqueeze(1))
        weights1 = self.fusion_s1(edge_features)
        edge_features = self.apply_weights(edge_features, weights1, dim=1)
        offset_weight = self.offset_s1(target_token.unsqueeze(1) + edge_features)
        one_stage_enhanced_tokens = (1 + offset_weight) * target_token.unsqueeze(1)

        expanded_class_indices = (
            class_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, intra_fea_num, dim)
        )
        corresponding_features = torch.gather(
            fea_bank.unsqueeze(0).expand(B, -1, -1, -1), 1, expanded_class_indices
        )

        edge_features2 = self.edge_fc_s2(
            corresponding_features - one_stage_enhanced_tokens.unsqueeze(2)
        )
        weights2 = self.fusion_s2(edge_features2)
        edge_features2 = self.apply_weights(edge_features2, weights2, dim=2)
        offset_weight2 = self.offset_s2(
            one_stage_enhanced_tokens.unsqueeze(2) + edge_features2
        )
        second_stage_enhanced_tokens = (
            1 + offset_weight2
        ) * one_stage_enhanced_tokens.unsqueeze(2)

        second_stage_enhanced_tokens = second_stage_enhanced_tokens.sum(dim=2)

        return second_stage_enhanced_tokens
