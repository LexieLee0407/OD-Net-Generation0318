"""损失函数。"""

from __future__ import annotations

import torch


def _masked_mean_squared_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对给定 mask 位置计算均方误差。"""
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff = pred[mask] - target[mask]
    return torch.mean(diff * diff)


def grouped_matrix_mse(pred: torch.Tensor, target: torch.Tensor, mask_diag: torch.Tensor, mask_pos_off: torch.Tensor, mask_zero_off: torch.Tensor) -> torch.Tensor:
    """三组区域等权平均的矩阵损失。"""
    diag_loss = _masked_mean_squared_error(pred, target, mask_diag)
    pos_off_loss = _masked_mean_squared_error(pred, target, mask_pos_off)
    zero_off_loss = _masked_mean_squared_error(pred, target, mask_zero_off)
    return (diag_loss + pos_off_loss + zero_off_loss) / 3.0

