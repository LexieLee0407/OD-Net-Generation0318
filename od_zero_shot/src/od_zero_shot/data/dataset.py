"""样本数据集抽象。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from od_zero_shot.data.sample_builder import GraphSample, load_manifest_paths, load_sample


def sample_to_tensor_dict(sample: GraphSample | dict[str, Any], device: str = "cpu") -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("需要先安装 torch 才能把样本转成张量。") from exc

    payload = sample.to_numpy_dict() if isinstance(sample, GraphSample) else sample
    device_obj = torch.device(device)
    return {
        "x_node": torch.as_tensor(payload["x_node"], dtype=torch.float32, device=device_obj),
        "edge_index": torch.as_tensor(payload["edge_index"], dtype=torch.long, device=device_obj),
        "edge_attr": torch.as_tensor(payload["edge_attr"], dtype=torch.float32, device=device_obj),
        "lap_pe": torch.as_tensor(payload["lap_pe"], dtype=torch.float32, device=device_obj),
        "se_feature": torch.as_tensor(payload["se_feature"], dtype=torch.float32, device=device_obj),
        "pair_geo": torch.as_tensor(payload["pair_geo"], dtype=torch.float32, device=device_obj),
        "pair_baseline": torch.as_tensor(payload["pair_baseline"], dtype=torch.float32, device=device_obj),
        "distance_matrix": torch.as_tensor(payload["distance_matrix"], dtype=torch.float32, device=device_obj),
        "population": torch.as_tensor(payload["population"], dtype=torch.float32, device=device_obj),
        "coords": torch.as_tensor(payload["coords"], dtype=torch.float32, device=device_obj),
        "y_od": torch.as_tensor(payload["y_od"], dtype=torch.float32, device=device_obj),
        "mask_diag": torch.as_tensor(payload["mask_diag"], dtype=torch.bool, device=device_obj),
        "mask_pos_off": torch.as_tensor(payload["mask_pos_off"], dtype=torch.bool, device=device_obj),
        "mask_zero_off": torch.as_tensor(payload["mask_zero_off"], dtype=torch.bool, device=device_obj),
        "row_sum": torch.as_tensor(payload["row_sum"], dtype=torch.float32, device=device_obj),
        "col_sum": torch.as_tensor(payload["col_sum"], dtype=torch.float32, device=device_obj),
    }


class ODSampleDataset:
    def __init__(self, sample_paths: list[str | Path]) -> None:
        self.sample_paths = [Path(path) for path in sample_paths]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return sample_to_tensor_dict(load_sample(self.sample_paths[index]))
