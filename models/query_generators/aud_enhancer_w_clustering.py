# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from the YenanLiu/DDESeg library
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers.weight_init import trunc_normal_


class GumbelSoftmaxClustering(nn.Module):
    def __init__(self, num_clusters, dim, temperature=1.0):
        super(GumbelSoftmaxClustering, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, dim)
        )  # Learnable cluster centers

    def forward(self, aud_fea):
        """
        Args:
            aud_fea: [B, H*W, dim] -> audio features
        Returns:
            cluster_assignments: [B, H*W, num_clusters] -> soft cluster assignments
            cluster_centers: [B, H*W, dim] -> recomputed cluster centers
        """
        # Compute similarity (dot product) between aud_fea and cluster centers
        logits = torch.einsum("bnd,kd->bnk", aud_fea, self.cluster_centers)

        # Apply Gumbel Softmax for cluster assignment
        gumbel_samples = F.gumbel_softmax(
            logits, tau=self.temperature, hard=False, dim=-1
        )
        cluster_centers = torch.einsum("bnk,bnd->bkd", gumbel_samples, aud_fea)
        return cluster_centers, gumbel_samples


class ClusteringTokenFeatureEnhancer(nn.Module):
    def __init__(self, embed_dim: int, k: int, aud_dim: int = 128):
        super().__init__()
        if aud_dim != embed_dim:
            self.audio_proj = nn.Linear(aud_dim, embed_dim)
        else:
            self.audio_proj = nn.Identity()  # type: ignore

        self.embed_dim = embed_dim
        self.selected_cls_num = k
        self.top_k = 5 if self.selected_cls_num >= 5 else self.selected_cls_num

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

        self.gumbel_clustering = GumbelSoftmaxClustering(k, embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def apply_weights(self, x, weights, dim):
        return x * torch.softmax(weights, dim=dim)

    def forward(self, target_token):
        """Enhances the target token using feature banks.

        Args:
            target_token (torch.Tensor): [B, embed_dim] or [B, 1, embed_dim]

        Returns:
            torch.Tensor: Enhanced target tokens [B, selected_cls_num, embed_dim]
        """
        target_token = self.audio_proj(target_token)
        B, _, dim = target_token.shape

        C_a, _ = self.gumbel_clustering(target_token)
        dist = torch.cdist(target_token, C_a, p=2).squeeze(1)
        _, class_indices = torch.topk(dist, k=self.top_k, dim=-1, largest=False)
        expanded_class_indices = class_indices.unsqueeze(-1).expand(-1, -1, dim)

        fea_mean = self.gumbel_clustering.cluster_centers
        fea_mean_expand = fea_mean.unsqueeze(0).expand(B, -1, dim)
        nearest_class_means = torch.gather(fea_mean_expand, 1, expanded_class_indices)

        edge_features = self.edge_fc_s1(nearest_class_means - target_token)
        weights1 = self.fusion_s1(edge_features)
        edge_features = self.apply_weights(edge_features, weights1, dim=1)
        offset_weight = self.offset_s1(target_token + edge_features)
        one_stage_enhanced_tokens = (1 + offset_weight) * target_token

        expanded_class_indices = (
            class_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, dim)
        )
        corresponding_features = torch.gather(
            C_a.unsqueeze(2), 1, expanded_class_indices
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
