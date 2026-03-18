"""通用工具函数。"""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(arg1: Any, arg2: Any) -> None:
    """保存 JSON。

    为兼容已有代码，同时支持 `save_json(data, path)` 与 `save_json(path, data)`。
    """
    if isinstance(arg1, (str, Path)) and not isinstance(arg2, (str, Path)):
        path_obj = Path(arg1)
        data = arg2
    else:
        data = arg1
        path_obj = Path(arg2)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """读取 JSON。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data: Any, path: str | Path) -> None:
    """保存 pickle。"""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: str | Path) -> Any:
    """读取 pickle。"""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def set_seed(seed: int) -> None:
    """设置实验随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def set_global_seed(seed: int) -> None:
    """兼容别名。"""
    set_seed(seed)


def choose_device(device_name: str) -> str:
    """将 `auto` 解析为可用设备。"""
    if device_name != "auto":
        return device_name
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def to_serializable(obj: Any) -> Any:
    """递归转换 numpy 类型。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    return obj
