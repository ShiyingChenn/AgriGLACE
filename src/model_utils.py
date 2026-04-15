import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedAddFusion(nn.Module):
    """
    Fused feature:
        fused = normalize(global + w * local)

    where:
        w = max_local_weight * sigmoid(raw_weight)

    This design injects local disease information into the global feature
    with a bounded learnable weight, so that local cues act as a supplement
    rather than overwhelming the original CLIP global semantics.
    """

    def __init__(self, feat_dim, max_local_weight=0.3, init_local_weight=0.1):
        super().__init__()
        if max_local_weight <= 0:
            raise ValueError("max_local_weight must be positive.")
        if init_local_weight <= 0 or init_local_weight >= max_local_weight:
            raise ValueError("init_local_weight must be in (0, max_local_weight).")

        self.feat_dim = feat_dim
        self.max_local_weight = float(max_local_weight)

        init_ratio = init_local_weight / max_local_weight
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        raw_init = torch.log(torch.tensor(init_ratio / (1.0 - init_ratio), dtype=torch.float32))
        self.raw_local_weight = nn.Parameter(raw_init)

    def get_weight(self):
        return self.max_local_weight * torch.sigmoid(self.raw_local_weight)

    def forward(self, global_feats, local_feats):
        global_feats = F.normalize(global_feats, dim=-1)
        local_feats = F.normalize(local_feats, dim=-1)

        weight = self.get_weight()
        fused = global_feats + weight * local_feats
        fused = F.normalize(fused, dim=-1)
        return fused


class GlobalLocalSimilarityScorer(nn.Module):
    """
    Global-guided patch scorer.

    We use global-patch dissimilarity as a class-agnostic local saliency signal:
    patches that deviate more from the global representation are more likely to
    contain localized disease patterns.
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, global_feats, patch_tokens):
        """
        Args:
            global_feats: [B, D]
            patch_tokens: [B, N, D]

        Returns:
            glsim_scores: [B, N]
        """
        global_feats = F.normalize(global_feats, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

        cosine = torch.einsum("bd,bnd->bn", global_feats, patch_tokens)  # [B, N]
        glsim_scores = 1.0 - cosine
        return glsim_scores


class PatchRanker(nn.Module):
    """
    PatchRanking-style lightweight patch importance predictor.

    It takes patch tokens together with their residuals w.r.t. the global feature
    and predicts a class-agnostic importance score for each patch.
    """

    def __init__(self, feat_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_feats, patch_tokens):
        """
        Args:
            global_feats: [B, D]
            patch_tokens: [B, N, D]

        Returns:
            rank_scores: [B, N]
        """
        global_feats = F.normalize(global_feats, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

        global_expand = global_feats.unsqueeze(1).expand_as(patch_tokens)  # [B, N, D]
        residual = patch_tokens - global_expand                            # [B, N, D]
        cosine = torch.sum(global_expand * patch_tokens, dim=-1, keepdim=True)  # [B, N, 1]

        feats = torch.cat([patch_tokens, residual, cosine], dim=-1)  # [B, N, 2D+1]
        rank_scores = self.mlp(feats).squeeze(-1)                    # [B, N]
        return rank_scores


class GLSimPatchRankLocalAdapter(nn.Module):
    """
    Global-guided Local Enhancement (GLE) module.

    Image-side local enhancement pipeline:
        1) global-guided patch scoring via GLSim-style dissimilarity
        2) learnable PatchRanking-style importance prediction
        3) weighted score fusion and top-r patch selection
        4) soft pooling and local adapter refinement

    This module is class-agnostic and does not rely on class text features.
    """

    def __init__(
        self,
        feat_dim,
        rank_hidden_dim=256,
        rank_dropout=0.1,
        local_hidden_dim=512,
        local_dropout=0.1,
        top_r=8,
        pool_tau=1.0,
        glsim_weight=0.5,
        rank_weight=0.5,
        eps=1e-12
    ):
        super().__init__()
        if top_r <= 0:
            raise ValueError("top_r must be positive.")
        if pool_tau <= 0:
            raise ValueError("pool_tau must be positive.")
        if glsim_weight < 0 or rank_weight < 0:
            raise ValueError("glsim_weight and rank_weight must be non-negative.")
        if glsim_weight == 0 and rank_weight == 0:
            raise ValueError("At least one of glsim_weight / rank_weight must be > 0.")

        self.top_r = top_r
        self.pool_tau = pool_tau
        self.glsim_weight = float(glsim_weight)
        self.rank_weight = float(rank_weight)
        self.eps = eps

        self.glsim_scorer = GlobalLocalSimilarityScorer(eps=eps)
        self.ranker = PatchRanker(
            feat_dim=feat_dim,
            hidden_dim=rank_hidden_dim,
            dropout=rank_dropout
        )

        self.local_adapter = nn.Sequential(
            nn.Linear(feat_dim, local_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(local_dropout),
            nn.Linear(local_hidden_dim, feat_dim)
        )

    def _minmax_norm(self, x):
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x = (x - x_min) / (x_max - x_min + self.eps)
        return x

    def forward(self, global_feats, patch_tokens):
        """
        Args:
            global_feats: [B, D]
            patch_tokens: [B, N, D]

        Returns:
            local_feat: [B, D]
            aux: dict
        """
        global_feats = F.normalize(global_feats, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

        glsim_scores = self.glsim_scorer(global_feats, patch_tokens)   # [B, N]
        rank_scores = self.ranker(global_feats, patch_tokens)          # [B, N]

        glsim_scores_norm = self._minmax_norm(glsim_scores)
        rank_scores_norm = self._minmax_norm(rank_scores)

        combined_scores = (
            self.glsim_weight * glsim_scores_norm +
            self.rank_weight * rank_scores_norm
        )  # [B, N]

        _, num_tokens, feat_dim = patch_tokens.shape
        k = min(self.top_r, num_tokens)

        top_scores, top_indices = combined_scores.topk(
            k=k, dim=1, largest=True, sorted=True
        )  # [B, k], [B, k]

        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        selected_tokens = torch.gather(patch_tokens, dim=1, index=gather_idx)  # [B, k, D]

        pool_weights = F.softmax(top_scores / self.pool_tau, dim=-1)  # [B, k]
        pooled_local = (selected_tokens * pool_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        local_feat = pooled_local + self.local_adapter(pooled_local)
        local_feat = F.normalize(local_feat, dim=-1)

        aux = {
            "glsim_scores": glsim_scores,
            "rank_scores": rank_scores,
            "glsim_scores_norm": glsim_scores_norm,
            "rank_scores_norm": rank_scores_norm,
            "combined_scores": combined_scores,
            "top_scores": top_scores,
            "top_indices": top_indices,
            "selected_tokens": selected_tokens,
            "pool_weights": pool_weights
        }
        return local_feat, aux