"""样本构造与序列化。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from od_zero_shot.data.fixtures import build_synthetic_toy100_raw, load_mini5_fixture, write_fixture_files
from od_zero_shot.data.geo import build_knn_graph, county_code_from_fips, distance_matrix, laplacian_positional_encoding, normalize_xy, structural_features
from od_zero_shot.data.raw import RawMobilityData, intersect_node_ids, load_raw_pickles, validate_raw_data
from od_zero_shot.utils.common import ensure_dir, load_pickle, save_pickle
from od_zero_shot.utils.config import ProjectConfig


@dataclass
class SampleBundle:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    fixtures: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def sample_artifact_path(project_root: str | Path) -> Path:
    return Path(project_root) / "artifacts" / "datasets" / "samples.pkl"


def _extract_subgraph(raw_data: RawMobilityData, node_ids: list[str], config: ProjectConfig, split_name: str, seed_id: str) -> dict[str, Any]:
    coords = np.asarray([raw_data.centroid[node_id] for node_id in node_ids], dtype=np.float32)
    populations = np.asarray([raw_data.population[node_id] for node_id in node_ids], dtype=np.float32)
    order = sorted(range(len(node_ids)), key=lambda idx: (coords[idx, 0], coords[idx, 1]))
    ordered_ids = [node_ids[idx] for idx in order]
    ordered_coords = coords[order]
    ordered_pop = populations[order]
    ordered_dist = distance_matrix(ordered_coords)
    ordered_xy = normalize_xy(ordered_coords)
    log_pop = np.log1p(ordered_pop).reshape(-1, 1).astype(np.float32)
    x_node = np.concatenate([ordered_xy, log_pop], axis=1).astype(np.float32)

    num_nodes = len(ordered_ids)
    y_od = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for row, origin in enumerate(ordered_ids):
        for col, destination in enumerate(ordered_ids):
            y_od[row, col] = math.log1p(float(raw_data.od2flow.get((origin, destination), 0.0)))

    edge_index, edge_attr = build_knn_graph(ordered_coords, config.dataset.knn_k)
    lap_pe = laplacian_positional_encoding(edge_index, num_nodes, config.model.lap_pe_dim)
    struct_feat = structural_features(edge_index, num_nodes, rw_steps=config.model.rw_steps)
    dx = ordered_coords[:, 0].reshape(-1, 1) - ordered_coords[:, 0].reshape(1, -1)
    dy = ordered_coords[:, 1].reshape(-1, 1) - ordered_coords[:, 1].reshape(1, -1)
    pair_geo = np.stack(
        [np.log1p(ordered_dist), dx.astype(np.float32), dy.astype(np.float32), np.eye(num_nodes, dtype=np.float32)],
        axis=-1,
    ).astype(np.float32)

    flow_matrix = np.expm1(y_od)
    mask_diag = np.eye(num_nodes, dtype=bool)
    mask_pos_off = (flow_matrix > 0.0) & (~mask_diag)
    mask_zero_off = (flow_matrix <= 0.0) & (~mask_diag)
    row_sum = flow_matrix.sum(axis=1).astype(np.float32)
    col_sum = flow_matrix.sum(axis=0).astype(np.float32)
    pair_baseline = np.stack(
        [
            np.repeat(log_pop, num_nodes, axis=1),
            np.repeat(log_pop.T, num_nodes, axis=0),
            np.log1p(ordered_dist),
            dx.astype(np.float32),
            dy.astype(np.float32),
            np.eye(num_nodes, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "sample_id": f"{split_name}:{seed_id}",
        "split": split_name,
        "seed_id": seed_id,
        "node_ids": ordered_ids,
        "coords": ordered_coords.astype(np.float32),
        "population": ordered_pop.astype(np.float32),
        "x_node": x_node,
        "edge_index": edge_index.astype(np.int64),
        "edge_attr": edge_attr.astype(np.float32),
        "lap_pe": lap_pe.astype(np.float32),
        "struct_feat": struct_feat.astype(np.float32),
        "pair_geo": pair_geo.astype(np.float32),
        "pair_baseline": pair_baseline.astype(np.float32),
        "distance_matrix": ordered_dist.astype(np.float32),
        "y_od": y_od.astype(np.float32),
        "mask_diag": mask_diag,
        "mask_pos_off": mask_pos_off,
        "mask_zero_off": mask_zero_off,
        "row_sum": row_sum,
        "col_sum": col_sum,
    }


def _candidate_pool(raw_data: RawMobilityData, counties: set[str]) -> list[str]:
    return [node_id for node_id in intersect_node_ids(raw_data) if county_code_from_fips(node_id) in counties]


def _nearest_nodes(seed_id: str, candidate_ids: list[str], centroid: dict[str, list[float]], sample_size: int) -> list[str]:
    seed_xy = np.asarray(centroid[seed_id], dtype=np.float32)
    pairs = []
    for node_id in candidate_ids:
        xy = np.asarray(centroid[node_id], dtype=np.float32)
        score = float(np.sqrt(np.sum((seed_xy - xy) ** 2)))
        pairs.append((score, node_id))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return [node_id for _, node_id in pairs[:sample_size]]


def _build_split_samples(raw_data: RawMobilityData, config: ProjectConfig, split_name: str, counties: set[str], max_samples: int | None = None) -> list[dict[str, Any]]:
    candidate_ids = sorted(_candidate_pool(raw_data, counties))
    if not candidate_ids or len(candidate_ids) < config.dataset.sample_size:
        return []
    selected_seed_ids = candidate_ids if max_samples is None else candidate_ids[:max_samples]
    samples = []
    for seed_id in selected_seed_ids:
        local_nodes = _nearest_nodes(seed_id, candidate_ids, raw_data.centroid, config.dataset.sample_size)
        if len(local_nodes) < config.dataset.sample_size:
            continue
        samples.append(_extract_subgraph(raw_data, local_nodes, config, split_name=split_name, seed_id=seed_id))
    return samples


def build_fixture_samples(project_root: str | Path, config: ProjectConfig) -> dict[str, dict[str, Any]]:
    fixtures_root = Path(project_root) / "data" / "fixtures"
    write_fixture_files(fixtures_root)
    mini_centroid, mini_population, mini_od2flow = load_mini5_fixture(fixtures_root)
    mini_raw = RawMobilityData(centroid=mini_centroid, population=mini_population, od2flow=mini_od2flow)
    mini_sample = _extract_subgraph(mini_raw, sorted(mini_centroid.keys()), config, split_name="fixture", seed_id="mini5")
    toy_centroid, toy_population, toy_od2flow = build_synthetic_toy100_raw()
    toy_raw = RawMobilityData(centroid=toy_centroid, population=toy_population, od2flow=toy_od2flow)
    toy_sample = _extract_subgraph(toy_raw, sorted(toy_centroid.keys()), config, split_name="fixture", seed_id="synthetic_toy100")
    return {"mini5": mini_sample, "synthetic_toy100": toy_sample}


def build_sample_bundle(project_root: str | Path, config: ProjectConfig, raw_data: RawMobilityData | None = None) -> SampleBundle:
    fixtures = build_fixture_samples(project_root, config)
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    raw_summary: dict[str, Any] = {"raw_data_available": False}
    if raw_data is not None:
        raw_summary = validate_raw_data(raw_data)
        raw_summary["raw_data_available"] = True
        all_counties = {county_code_from_fips(node_id) for node_id in intersect_node_ids(raw_data)}
        test_counties = set(config.dataset.heldout_counties)
        val_counties = set(config.dataset.val_counties)
        train_counties = all_counties - test_counties - val_counties
        train_samples = _build_split_samples(raw_data, config, "train", train_counties, max_samples=config.dataset.num_train_samples)
        val_samples = _build_split_samples(raw_data, config, "val", val_counties)
        test_samples = _build_split_samples(raw_data, config, "test", test_counties)
    return SampleBundle(
        train=train_samples,
        val=val_samples,
        test=test_samples,
        fixtures=fixtures,
        summary={
            "raw": raw_summary,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
            "fixtures": list(fixtures.keys()),
        },
    )


def save_sample_bundle(project_root: str | Path, bundle: SampleBundle) -> Path:
    path = sample_artifact_path(project_root)
    ensure_dir(path.parent)
    save_pickle(bundle, path)
    return path


def load_or_build_sample_bundle(project_root: str | Path, config: ProjectConfig) -> SampleBundle:
    path = sample_artifact_path(project_root)
    if path.exists():
        return load_pickle(path)
    raw_root = Path(project_root) / config.dataset.raw_root
    raw_data = load_raw_pickles(raw_root) if raw_root.exists() else None
    bundle = build_sample_bundle(project_root, config, raw_data=raw_data)
    save_sample_bundle(project_root, bundle)
    return bundle
