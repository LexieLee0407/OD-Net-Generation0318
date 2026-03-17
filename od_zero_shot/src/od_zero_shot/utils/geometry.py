"""纯几何计算与图结构构造。"""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def parse_fips(fips: str) -> tuple[str, str, str]:
    if not isinstance(fips, str) or len(fips) != 11 or not fips.isdigit():
        raise ValueError(f"非法 FIPS: {fips}")
    return fips[:2], fips[2:5], fips[5:]


def county_code_from_fips(fips: str) -> str:
    return parse_fips(fips)[1]


def order_indices_xy(coords_lon_lat: np.ndarray) -> np.ndarray:
    return np.lexsort((coords_lon_lat[:, 1], coords_lon_lat[:, 0]))


def normalize_coords(coords_lon_lat: np.ndarray) -> np.ndarray:
    mins = coords_lon_lat.min(axis=0, keepdims=True)
    maxs = coords_lon_lat.max(axis=0, keepdims=True)
    spans = np.where((maxs - mins) < 1e-6, 1.0, maxs - mins)
    return (coords_lon_lat - mins) / spans


def haversine_matrix(coords_lon_lat: np.ndarray) -> np.ndarray:
    lon = np.radians(coords_lon_lat[:, 0])[:, None]
    lat = np.radians(coords_lon_lat[:, 1])[:, None]
    dlon = lon.T - lon
    dlat = lat.T - lat
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c


def coordinate_delta_matrices(coords_lon_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = coords_lon_lat[:, 0][:, None] - coords_lon_lat[:, 0][None, :]
    dy = coords_lon_lat[:, 1][:, None] - coords_lon_lat[:, 1][None, :]
    return dx.astype(np.float32), dy.astype(np.float32)


def build_knn_graph(distance_matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = distance_matrix.shape[0]
    if k >= num_nodes:
        raise ValueError(f"k={k} 必须严格小于节点数 {num_nodes}")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
    for node_idx in range(num_nodes):
        nearest = np.argsort(distance_matrix[node_idx])[1 : k + 1]
        adjacency[node_idx, nearest] = True
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)
    edge_index = np.vstack(np.nonzero(adjacency)).astype(np.int64)
    return edge_index, adjacency


def degree_feature(adjacency: np.ndarray) -> np.ndarray:
    return adjacency.sum(axis=1, keepdims=True).astype(np.float32)


def rw_diagonal_feature(adjacency: np.ndarray, steps: int = 2) -> np.ndarray:
    degree = adjacency.sum(axis=1, keepdims=True).astype(np.float64)
    degree = np.where(degree == 0, 1.0, degree)
    transition = adjacency.astype(np.float64) / degree
    power = np.linalg.matrix_power(transition, steps)
    return np.diag(power).reshape(-1, 1).astype(np.float32)


def laplacian_positional_encoding(adjacency: np.ndarray, dim: int) -> np.ndarray:
    num_nodes = adjacency.shape[0]
    adj = adjacency.astype(np.float64)
    degree = adj.sum(axis=1)
    safe_degree = np.where(degree <= 0, 1.0, degree)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(safe_degree))
    lap = np.eye(num_nodes) - d_inv_sqrt @ adj @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(lap)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    useful = eigvecs[:, 1 : dim + 1]
    if useful.shape[1] < dim:
        useful = np.pad(useful, ((0, 0), (0, dim - useful.shape[1])))
    return useful.astype(np.float32)


def bucketize_by_edges(values: np.ndarray, bin_edges) -> np.ndarray:
    return np.digitize(values, np.asarray(list(bin_edges), dtype=np.float64), right=False)


def log1p_safe(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(values, a_min=0.0, a_max=None))


def inverse_log1p(values: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(values), a_min=0.0, a_max=None)


def stable_sample(items: list[str], max_count: int) -> list[str]:
    return items[: max(0, min(len(items), max_count))]
