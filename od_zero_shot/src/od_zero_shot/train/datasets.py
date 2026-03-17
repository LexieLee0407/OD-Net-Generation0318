"""训练阶段的数据适配器。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def to_torch_sample(sample: dict[str, Any], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """将 numpy sample 转成 torch 张量。"""
    device_obj = torch.device(device)
    return {
        "x_node": torch.as_tensor(sample["x_node"], dtype=torch.float32, device=device_obj),
        "edge_index": torch.as_tensor(sample["edge_index"], dtype=torch.long, device=device_obj),
        "edge_attr": torch.as_tensor(sample["edge_attr"], dtype=torch.float32, device=device_obj),
        "lap_pe": torch.as_tensor(sample["lap_pe"], dtype=torch.float32, device=device_obj),
        "struct_feat": torch.as_tensor(sample["struct_feat"], dtype=torch.float32, device=device_obj),
        "pair_geo": torch.as_tensor(sample["pair_geo"], dtype=torch.float32, device=device_obj),
        "pair_baseline": torch.as_tensor(sample["pair_baseline"], dtype=torch.float32, device=device_obj),
        "distance_matrix": torch.as_tensor(sample["distance_matrix"], dtype=torch.float32, device=device_obj),
        "y_od": torch.as_tensor(sample["y_od"], dtype=torch.float32, device=device_obj),
        "mask_diag": torch.as_tensor(sample["mask_diag"], dtype=torch.bool, device=device_obj),
        "mask_pos_off": torch.as_tensor(sample["mask_pos_off"], dtype=torch.bool, device=device_obj),
        "mask_zero_off": torch.as_tensor(sample["mask_zero_off"], dtype=torch.bool, device=device_obj),
        "row_sum": torch.as_tensor(sample["row_sum"], dtype=torch.float32, device=device_obj),
        "col_sum": torch.as_tensor(sample["col_sum"], dtype=torch.float32, device=device_obj),
    }


class PairDataset(Dataset):
    """将样本展平为 pair 级别监督。"""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        pairs = []
        targets = []
        for sample in samples:
            pairs.append(sample["pair_baseline"].reshape(-1, sample["pair_baseline"].shape[-1]))
            targets.append(sample["y_od"].reshape(-1))
        self.pairs = torch.as_tensor(np.concatenate(pairs, axis=0), dtype=torch.float32)
        self.targets = torch.as_tensor(np.concatenate(targets, axis=0), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[index], self.targets[index]


class MatrixDataset(Dataset):
    """矩阵级样本数据集。"""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

