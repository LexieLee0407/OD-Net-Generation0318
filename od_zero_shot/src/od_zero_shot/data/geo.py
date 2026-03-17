"""几何与结构特征计算。"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh


EARTH_RADIUS_KM = 6371.0


def parse_fips(fips: str) -> tuple[str, str, str]:
    """解析 11 位 FIPS 编码。"""
    if len(fips) != 11 or not fips.isdigit():
        raise ValueError(f"非法 FIPS 编码: {fips}")
    return fips[:2], fips[2:5], fips[5:]


def county_code_from_fips(fips: str) -> str:
    """提取县代码。"""
    return parse_fips(fips)[1]


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """球面距离，单位公里。"""
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(max(a, 0.0)))
    return EARTH_RADIUS_KM * c


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """计算全体节点的距离矩阵。"""
    num_nodes = coords.shape[0]
    dist = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            value = haversine_distance(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dist[i, j] = value
            dist[j, i] = value
    return dist


def normalize_xy(coords: np.ndarray) -> np.ndarray:
    """将经纬度线性归一化到 [-1, 1]。"""
    minimum = coords.min(axis=0, keepdims=True)
    maximum = coords.max(axis=0, keepdims=True)
    denom = np.where((maximum - minimum) > 1e-6, maximum - minimum, 1.0)
    scaled = (coords - minimum) / denom
    return (scaled * 2.0 - 1.0).astype(np.float32)


def build_knn_graph(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """构造无向 kNN 图。"""
    dist = distance_matrix(coords)
    num_nodes = dist.shape[0]
    edge_pairs: set[tuple[int, int]] = set()
    for node_index in range(num_nodes):
        nearest = np.argsort(dist[node_index] + np.eye(num_nodes, dtype=np.float32)[node_index] * 1e9)[:k]
        for neighbor in nearest:
            edge_pairs.add((node_index, int(neighbor)))
            edge_pairs.add((int(neighbor), node_index))
    ordered_pairs = sorted(edge_pairs)
    edge_index = np.asarray(ordered_pairs, dtype=np.int64).T
    edge_attr = []
    for src, dst in ordered_pairs:
        d_ij = float(dist[src, dst])
        dx = float(coords[dst, 0] - coords[src, 0])
        dy = float(coords[dst, 1] - coords[src, 1])
        edge_attr.append([math.log1p(d_ij), dx, dy])
    return edge_index, np.asarray(edge_attr, dtype=np.float32)


def laplacian_positional_encoding(edge_index: np.ndarray, num_nodes: int, dim: int) -> np.ndarray:
    """计算 Laplacian PE。"""
    if num_nodes == 1:
        return np.zeros((1, dim), dtype=np.float32)
    values = np.ones(edge_index.shape[1], dtype=np.float32)
    adjacency = csr_matrix((values, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    adjacency = adjacency.maximum(adjacency.T)
    lap = sparse_laplacian(adjacency, normed=True)
    eig_count = min(dim + 1, max(2, num_nodes - 1))
    try:
        _, eigenvectors = eigsh(lap, k=eig_count, which="SM")
        eigenvectors = eigenvectors[:, 1:eig_count]
    except Exception:
        eigenvectors = np.zeros((num_nodes, max(1, eig_count - 1)), dtype=np.float32)
    if eigenvectors.shape[1] < dim:
        pad = np.zeros((num_nodes, dim - eigenvectors.shape[1]), dtype=np.float32)
        eigenvectors = np.concatenate([eigenvectors, pad], axis=1)
    return eigenvectors[:, :dim].astype(np.float32)


def structural_features(edge_index: np.ndarray, num_nodes: int, rw_steps: int = 2) -> np.ndarray:
    """计算 degree 与随机游走对角项。"""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(axis=1, keepdims=True)
    degree_safe = np.where(degree > 0.0, degree, 1.0)
    transition = adjacency / degree_safe
    walk = transition.copy()
    for _ in range(max(1, rw_steps) - 1):
        walk = walk @ transition
    diagonal = np.diag(walk).reshape(-1, 1)
    return np.concatenate([degree, diagonal], axis=1).astype(np.float32)

