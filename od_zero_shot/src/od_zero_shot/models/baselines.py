"""最小基线模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from od_zero_shot.utils.common import load_json, save_json


def build_pair_features_torch(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    """从张量样本构造 `[log(pop_i), log(pop_j), log(d), dx, dy, selfloop]`。"""

    if "pair_baseline" in sample:
        return sample["pair_baseline"]
    population = sample["population"]
    pair_geo = sample["pair_geo"]
    batch_size, num_nodes = population.shape
    log_pop = torch.log1p(population)
    pop_i = log_pop.unsqueeze(2).expand(batch_size, num_nodes, num_nodes)
    pop_j = log_pop.unsqueeze(1).expand(batch_size, num_nodes, num_nodes)
    return torch.cat([pop_i.unsqueeze(-1), pop_j.unsqueeze(-1), pair_geo], dim=-1)


@dataclass(slots=True)
class GravityModel:
    """最小 Gravity 回归基线。"""

    coefficients: np.ndarray | None = None

    def fit(self, samples: list[dict]) -> None:
        feature_rows = []
        targets = []
        for sample in samples:
            # Gravity 基线固定使用人口、距离与 selfloop_flag 四维特征。
            feature_rows.append(sample["pair_baseline"][..., [0, 1, 2, 5]].reshape(-1, 4))
            targets.append(sample["y_od"].reshape(-1))
        x_mat = np.concatenate(feature_rows, axis=0).astype(np.float64)
        y_vec = np.concatenate(targets, axis=0).astype(np.float64)
        design = np.concatenate([np.ones((x_mat.shape[0], 1), dtype=np.float64), x_mat], axis=1)
        self.coefficients = np.linalg.pinv(design.T @ design) @ design.T @ y_vec

    def predict_matrix(self, sample: dict) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Gravity 模型尚未训练。")
        features = sample["pair_baseline"][..., [0, 1, 2, 5]].reshape(-1, 4).astype(np.float64)
        design = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
        pred = design @ self.coefficients
        side = sample["y_od"].shape[0]
        return pred.reshape(side, side).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.coefficients is None:
            raise RuntimeError("Gravity 模型尚未训练。")
        save_json(path, {"coefficients": self.coefficients.tolist()})

    @classmethod
    def load(cls, path: str | Path) -> "GravityModel":
        payload = load_json(path)
        return cls(coefficients=np.asarray(payload["coefficients"], dtype=np.float32))


def fit_gravity_from_sample_dicts(samples: list[dict]) -> GravityModel:
    model = GravityModel()
    model.fit(samples)
    return model


def gravity_predict_sample(model: GravityModel, sample: dict) -> np.ndarray:
    return model.predict_matrix(sample)


class PairMLP(nn.Module):
    """有向 pair 的多层感知机基线。"""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        output = self.net(pair_features)
        return output.squeeze(-1)
