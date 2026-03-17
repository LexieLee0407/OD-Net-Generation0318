"""原始数据读取与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from od_zero_shot.data.geo import county_code_from_fips, parse_fips
from od_zero_shot.utils.common import load_pickle


@dataclass
class RawMobilityData:
    """原始图结构。"""

    centroid: dict[str, list[float]]
    population: dict[str, int]
    od2flow: dict[tuple[str, str], float]

    @property
    def centroids(self) -> dict[str, list[float]]:
        return self.centroid

    @property
    def populations(self) -> dict[str, int]:
        return self.population

    @property
    def flows(self) -> dict[tuple[str, str], float]:
        return self.od2flow

    @property
    def node_ids(self) -> list[str]:
        return sorted(self.centroid.keys())

    def summary(self) -> dict[str, int]:
        counties = {county_code_from_fips(node_id) for node_id in self.centroid}
        zero_pop = sum(1 for value in self.population.values() if value <= 0)
        return {
            "num_nodes": len(self.centroid),
            "num_flows": len(self.od2flow),
            "num_counties": len(counties),
            "num_zero_population_nodes": zero_pop,
        }


def load_raw_pickles(raw_root: str | Path) -> RawMobilityData:
    """读取原始三件套。"""
    root = Path(raw_root)
    centroid = load_pickle(root / "centroid.pkl")
    population = load_pickle(root / "population.pkl")
    od2flow = load_pickle(root / "od2flow.pkl")
    return RawMobilityData(centroid=centroid, population=population, od2flow=od2flow)


def validate_raw_data(raw_data: RawMobilityData) -> dict[str, int]:
    """验证一致性并返回摘要。"""
    centroid_keys = set(raw_data.centroid.keys())
    population_keys = set(raw_data.population.keys())
    missing_centroid = population_keys - centroid_keys
    missing_population = centroid_keys - population_keys
    invalid_edges = 0
    for node_id in centroid_keys | population_keys:
        parse_fips(node_id)
    for origin, destination in raw_data.od2flow.keys():
        parse_fips(origin)
        parse_fips(destination)
        if origin not in centroid_keys or destination not in centroid_keys:
            invalid_edges += 1
    return {
        "num_centroid_nodes": len(centroid_keys),
        "num_population_nodes": len(population_keys),
        "num_intersection_nodes": len(centroid_keys & population_keys),
        "missing_centroid_nodes": len(missing_centroid),
        "missing_population_nodes": len(missing_population),
        "num_edges": len(raw_data.od2flow),
        "invalid_edges": invalid_edges,
    }


def intersect_node_ids(raw_data: RawMobilityData) -> list[str]:
    """返回可用节点交集。"""
    return sorted(set(raw_data.centroid.keys()) & set(raw_data.population.keys()))
