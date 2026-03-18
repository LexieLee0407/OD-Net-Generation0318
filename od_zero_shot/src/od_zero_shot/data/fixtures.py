"""内置夹具数据。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from od_zero_shot.data.raw import RawMobilityData
from od_zero_shot.utils.common import load_json, save_json


MINI5_CENTROID = {
    "36061000100": [-74.0461708, 40.6900008],
    "36055011703": [-77.4008057, 43.1028109],
    "36069050101": [-77.4582906, 43.0059497],
    "36055014003": [-77.68674509999998, 43.216090200000004],
    "36075021602": [-76.5295124, 43.4510055],
}

MINI5_POPULATION = {
    "36061000100": 0,
    "36055011703": 4251,
    "36069050101": 3298,
    "36055014003": 4641,
    "36075021602": 2821,
}

MINI5_OD2FLOW = {
    ("36055011703", "36061000100"): 14.0,
    ("36055011703", "36069050101"): 381.0,
    ("36055011703", "36055014003"): 29.0,
    ("36055011703", "36075021602"): 14.0,
    ("36055011703", "36055011703"): 6787.0,
    ("36055014003", "36069050101"): 34.0,
    ("36055014003", "36055014003"): 4008.0,
    ("36055014003", "36055011703"): 34.0,
    ("36069050101", "36069050101"): 2999.0,
    ("36069050101", "36055011703"): 35.0,
    ("36075021602", "36075021602"): 3324.0,
}


def load_mini5_fixture(fixtures_root: str | Path) -> tuple[dict[str, list[float]], dict[str, int], dict[tuple[str, str], float]]:
    """从磁盘读取 5 节点夹具。"""
    root = Path(fixtures_root) / "mini5"
    centroid = load_json(root / "centroid.json")
    population = load_json(root / "population.json")
    od_raw = load_json(root / "od2flow.json")
    od2flow = {}
    for key, value in od_raw.items():
        origin, destination = key.split("|")
        od2flow[(origin, destination)] = float(value)
    return centroid, population, od2flow


def write_fixture_files(fixtures_root: str | Path) -> None:
    """把内置夹具写回磁盘。"""
    root = Path(fixtures_root) / "mini5"
    root.mkdir(parents=True, exist_ok=True)
    save_json(MINI5_CENTROID, root / "centroid.json")
    save_json(MINI5_POPULATION, root / "population.json")
    save_json({f"{o}|{d}": v for (o, d), v in MINI5_OD2FLOW.items()}, root / "od2flow.json")


def build_synthetic_toy100_raw(seed: int = 20260317) -> tuple[dict[str, list[float]], dict[str, int], dict[tuple[str, str], float]]:
    """生成确定性的 100 节点合成数据。"""
    rng = np.random.default_rng(seed)
    centroid: dict[str, list[float]] = {}
    population: dict[str, int] = {}
    od2flow: dict[tuple[str, str], float] = {}
    node_ids: list[str] = []
    coords: list[tuple[float, float]] = []
    pops: list[int] = []
    for row in range(10):
        for col in range(10):
            county_code = "001" if row < 5 else "029"
            tract_code = f"{row * 10 + col + 1:06d}"
            node_id = f"36{county_code}{tract_code}"
            lon = -78.0 + col * 0.18 + float(rng.normal(0.0, 0.005))
            lat = 42.0 + row * 0.16 + float(rng.normal(0.0, 0.005))
            pop = int(900 + 4500 * abs(np.sin((row + 1) * (col + 2) / 17.0)) + rng.integers(0, 400))
            centroid[node_id] = [lon, lat]
            population[node_id] = pop
            node_ids.append(node_id)
            coords.append((lon, lat))
            pops.append(pop)
    coords_np = np.asarray(coords, dtype=np.float64)
    pops_np = np.asarray(pops, dtype=np.float64)
    for i, origin in enumerate(node_ids):
        for j, destination in enumerate(node_ids):
            dx = coords_np[i, 0] - coords_np[j, 0]
            dy = coords_np[i, 1] - coords_np[j, 1]
            distance = np.sqrt(dx * dx + dy * dy) + 1e-6
            county_bonus = 0.7 if origin[2:5] == destination[2:5] else 0.0
            self_bonus = 4.5 if i == j else 0.0
            interaction = (
                0.55 * np.log1p(pops_np[i])
                + 0.45 * np.log1p(pops_np[j])
                - 1.75 * np.log1p(distance * 80.0)
                + county_bonus
                + self_bonus
            )
            flow = np.expm1(max(interaction, 0.0))
            if i != j and flow < 6.0:
                continue
            od2flow[(origin, destination)] = float(np.round(flow))
    return centroid, population, od2flow


def load_five_node_fixture() -> RawMobilityData:
    """返回用户提供的 5 节点精确夹具。"""

    return RawMobilityData(
        centroid={key: list(value) for key, value in MINI5_CENTROID.items()},
        population={key: int(value) for key, value in MINI5_POPULATION.items()},
        od2flow={key: float(value) for key, value in MINI5_OD2FLOW.items()},
    )


def generate_synthetic_toy100() -> RawMobilityData:
    """返回当前工程使用的 100 节点 deterministic toy 数据。"""

    centroid: dict[str, list[float]] = {}
    population: dict[str, int] = {}
    od2flow: dict[tuple[str, str], float] = {}
    county_codes = [47, 61, 81, 85]
    node_ids: list[str] = []
    coords: dict[str, tuple[float, float]] = {}
    for row in range(10):
        for col in range(10):
            tract_idx = row * 10 + col + 1
            county = county_codes[(row + col) % len(county_codes)]
            node_id = f"{36:02d}{county:03d}{tract_idx:06d}"
            lon = -79.5 + col * 0.18 + (row % 2) * 0.02
            lat = 40.5 + row * 0.22 + (col % 3) * 0.015
            pop = 600 + row * 250 + col * 180 + ((row * col) % 7) * 50
            centroid[node_id] = [float(lon), float(lat)]
            population[node_id] = int(pop)
            node_ids.append(node_id)
            coords[node_id] = (float(lon), float(lat))

    for origin in node_ids:
        lon_i, lat_i = coords[origin]
        pop_i = population[origin]
        for destination in node_ids:
            lon_j, lat_j = coords[destination]
            pop_j = population[destination]
            dx = abs(lon_i - lon_j)
            dy = abs(lat_i - lat_j)
            dist = (dx**2 + dy**2) ** 0.5
            same_county = 1.0 if origin[2:5] == destination[2:5] else 0.0
            self_loop = 1.0 if origin == destination else 0.0
            base = (pop_i ** 0.45) * (pop_j ** 0.42)
            decay = 1.0 / (1.0 + 6.0 * dist)
            county_bonus = 1.25 if same_county else 0.85
            self_bonus = 28.0 if self_loop else 1.0
            value = base * decay * county_bonus * self_bonus / 80.0
            if not self_loop and value < 10.0:
                continue
            od2flow[(origin, destination)] = float(round(value, 3))
    return RawMobilityData(centroid=centroid, population=population, od2flow=od2flow)


def load_fixture(name: str) -> RawMobilityData:
    """按名称返回夹具。"""

    if name in {"mini5", "five_node"}:
        return load_five_node_fixture()
    if name == "synthetic_toy100":
        return generate_synthetic_toy100()
    raise ValueError(f"未知夹具名: {name}")
