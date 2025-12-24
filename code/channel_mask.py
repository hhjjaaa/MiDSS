import torch
import torch.nn as nn
from typing import Tuple


def _split_paired_batch(feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入是 [2B, C, H, W]，前一半对应 A（原图），后一半对应 A*（增强图）。
    返回 (feat_A, feat_A_aug)，形状均为 [B, C, H, W]。
    """
    if feat.dim() != 4 or feat.size(0) % 2 != 0:
        raise ValueError(f"feat must be [2B,C,H,W], got {feat.shape}")
    B2, C, H, W = feat.shape
    B = B2 // 2
    return feat[:B], feat[B:]


@torch.no_grad()
def _covariance_map(feat: torch.Tensor) -> torch.Tensor:
    """Eq.(4): Σ_i = (1/HW) * W_i W_i^T. feat: [B,C,H,W] -> [B,C,C]"""
    if feat.dim() != 4:
        raise ValueError(f"feat must be [B,C,H,W], got {feat.shape}")
    B, C, H, W = feat.shape
    x = feat.reshape(B, C, H * W)                         # [B, C, HW]
    sigma = torch.bmm(x, x.transpose(1, 2)) / (H * W)     # [B, C, C]
    return sigma


@torch.no_grad()
def compute_V_and_S_from_concat(feat_concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入：已在 batch 维拼接的特征 [2B,C,H,W]，前 B 对应 A，后 B 对应 A*。
    输出：V [C,C]，S [C]。
    """
    feat, feat_aug = _split_paired_batch(feat_concat)

    sigma = _covariance_map(feat)         # [B,C,C]
    sigma_aug = _covariance_map(feat_aug) # [B,C,C]

    mu = 0.5 * (sigma + sigma_aug)
    per_sample_var = 0.5 * ((sigma - mu) ** 2 + (sigma_aug - mu) ** 2)  # [B,C,C]
    V = per_sample_var.mean(dim=0)  # [C,C]
    S = V.mean(dim=1)               # [C]
    return V, S


class AdaptiveChannelMask(nn.Module):
    """
    用已标注 batch 的 (A, A*) 特征更新通道敏感度 S（不对标注 batch 做 mask）；
    在无标签 batch 上按 S 排序屏蔽低敏感通道，迫使模型挖掘弱通道。

    典型使用：
      1) 标注 batch 前向得到特征 concat_feats=[A;A*]，调用 update_from_labeled(concat_feats) 更新 running_S。
      2) 无标签 batch 前向时，将特征通过 forward() 获得通道屏蔽后的输出。
    """

    def __init__(
        self,
        mask_percent: float = 0.2,
        rank_percent: float = 0.2,
        mask_value: str = "zero",  # "zero" or "mean"
        ema_momentum: float = 0.9,
        enabled: bool = True,
    ):
        super().__init__()
        if not (0.0 < mask_percent < 1.0):
            raise ValueError("mask_percent must be in (0,1)")
        if not (0.0 < rank_percent <= 1.0):
            raise ValueError("rank_percent must be in (0,1]")
        if mask_value not in ("zero", "mean"):
            raise ValueError('mask_value must be "zero" or "mean"')
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0,1)")

        self.mask_percent = mask_percent
        self.rank_percent = rank_percent
        self.mask_value = mask_value
        self.ema_momentum = ema_momentum
        self.enabled = enabled

        # running_S 保存跨 iteration 的敏感度（EMA）
        self.register_buffer("running_S", torch.empty(0), persistent=True)

    @torch.no_grad()
    def update_from_labeled(self, concat_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅在有标签 batch 上调用：输入 concat_feat=[A;A*] (形状 [2B,C,H,W])，更新 running_S。
        返回 (V,S) 便于可视化/调试。
        """
        V, S = compute_V_and_S_from_concat(concat_feat)
        if self.running_S.numel() == 0:
            self.running_S = S.detach().clone()
        else:
            if self.running_S.shape != S.shape:
                raise ValueError(f"running_S shape {self.running_S.shape} != new S {S.shape}")
            self.running_S = self.ema_momentum * self.running_S + (1 - self.ema_momentum) * S.detach()
        return V, S

    def _sample_mask_indices(self, C: int, device: torch.device) -> torch.Tensor:
        """
        对 running_S 排序（升序：低敏感 -> 高敏感），从前 rank_percent*C 个候选中随机取 mask_percent*C 个做遮蔽。
        """
        if self.running_S.numel() == 0:
            raise RuntimeError("running_S is empty. Call update_from_labeled(...) first.")
        if self.running_S.numel() != C:
            raise RuntimeError(f"running_S has C={self.running_S.numel()}, but feature has C={C}")

        sorted_idx = torch.argsort(self.running_S)  # low -> high
        M = max(1, int(round(self.rank_percent * C)))   # 候选池
        K = max(1, int(round(self.mask_percent * C)))   # 最终 mask 数

        candidate = sorted_idx[:M]  # mask 低敏感通道
        K = min(K, candidate.numel())
        perm = torch.randperm(candidate.numel(), device=device)
        mask_idx = candidate[perm[:K]]
        return mask_idx

    def sample_mask_indices(self, C: int, device: torch.device) -> torch.Tensor:
        """Public wrapper to sample mask indices from running_S."""
        return self._sample_mask_indices(C, device)

    def apply_mask(self, x: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        """Apply channel masking to x using provided indices."""
        if x.dim() != 4:
            raise ValueError(f"AdaptiveChannelMask expects [B,C,H,W], got {x.shape}")
        B, C, H, W = x.shape
        if mask_idx.numel() == 0:
            return x
        out = x.clone()
        if self.mask_value == "zero":
            out[:, mask_idx, :, :] = 0.0
        else:
            fill = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
            out[:, mask_idx, :, :] = fill.expand(B, mask_idx.numel(), H, W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        在无标签 batch 上调用：输入特征 x=[B,C,H,W]，按照 running_S 屏蔽低敏感通道。
        标注 batch 不调用 forward，只用 update_from_labeled 更新 running_S。
        """
        if (not self.enabled) or (not self.training):
            return x
        if x.dim() != 4:
            raise ValueError(f"AdaptiveChannelMask expects [B,C,H,W], got {x.shape}")
        B, C, H, W = x.shape

        if self.running_S.numel() == 0:
            return x

        mask_idx = self._sample_mask_indices(C, x.device)
        return self.apply_mask(x, mask_idx)
