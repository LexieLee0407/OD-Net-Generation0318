"""诊断图可视化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from od_zero_shot.utils.common import ensure_dir


def _save_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true.reshape(-1), y_pred.reshape(-1), s=8, alpha=0.35)
    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("True log1p(flow)")
    ax.set_ylabel("Pred log1p(flow)")
    ax.set_title("True vs Pred Scatter")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_row_col_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    true_flow = np.expm1(y_true)
    pred_flow = np.maximum(np.expm1(y_pred), 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(true_flow.sum(axis=1), label="true")
    axes[0].plot(pred_flow.sum(axis=1), label="pred")
    axes[0].set_title("Row Sum")
    axes[0].legend()
    axes[1].plot(true_flow.sum(axis=0), label="true")
    axes[1].plot(pred_flow.sum(axis=0), label="pred")
    axes[1].set_title("Column Sum")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_topk_plot(y_true: np.ndarray, y_pred: np.ndarray, top_k: int, path: Path) -> None:
    true_adj = np.zeros_like(y_true, dtype=np.float32)
    pred_adj = np.zeros_like(y_pred, dtype=np.float32)
    true_flow = np.expm1(y_true).copy()
    pred_flow = np.expm1(y_pred).copy()
    np.fill_diagonal(true_flow, -np.inf)
    np.fill_diagonal(pred_flow, -np.inf)
    for row in range(true_flow.shape[0]):
        true_adj[row, np.argsort(true_flow[row])[-top_k:]] = 1.0
        pred_adj[row, np.argsort(pred_flow[row])[-top_k:]] = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(true_adj, cmap="Greys")
    axes[0].set_title("True Top-K Outgoing")
    axes[1].imshow(pred_adj, cmap="Greys")
    axes[1].set_title("Pred Top-K Outgoing")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_distance_decay_plot(y_true: np.ndarray, y_pred: np.ndarray, distances: np.ndarray, bins: int, path: Path) -> None:
    valid_mask = np.ones_like(distances, dtype=bool)
    np.fill_diagonal(valid_mask, False)
    dist_values = distances[valid_mask]
    true_flow = np.expm1(y_true)[valid_mask]
    pred_flow = np.maximum(np.expm1(y_pred), 0.0)[valid_mask]
    if dist_values.size == 0:
        return
    edges = np.linspace(float(dist_values.min()), float(dist_values.max()), bins + 1)
    centers = []
    true_curve = []
    pred_curve = []
    for idx in range(bins):
        mask = (dist_values >= edges[idx]) & (dist_values <= edges[idx + 1] if idx == bins - 1 else dist_values < edges[idx + 1])
        if np.sum(mask) == 0:
            continue
        centers.append((edges[idx] + edges[idx + 1]) / 2.0)
        true_curve.append(float(true_flow[mask].mean()))
        pred_curve.append(float(pred_flow[mask].mean()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, true_curve, marker="o", label="true")
    ax.plot(centers, pred_curve, marker="o", label="pred")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Mean flow")
    ax.set_title("Distance Decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_diagnostic_plots(sample: dict, pred_log: np.ndarray, output_dir: str | Path, prefix: str, top_k: int, distance_bins: int) -> None:
    output = ensure_dir(output_dir)
    true_log = sample["y_od"]
    _save_heatmap(true_log, "True OD Heatmap", output / f"{prefix}_true_heatmap.png")
    _save_heatmap(pred_log, "Pred OD Heatmap", output / f"{prefix}_pred_heatmap.png")
    _save_scatter(true_log, pred_log, output / f"{prefix}_scatter.png")
    _save_row_col_plot(true_log, pred_log, output / f"{prefix}_row_col.png")
    _save_topk_plot(true_log, pred_log, top_k=top_k, path=output / f"{prefix}_topk.png")
    _save_distance_decay_plot(true_log, pred_log, sample["distance_matrix"], bins=distance_bins, path=output / f"{prefix}_distance_decay.png")


def plot_heatmap(true_log: np.ndarray, pred_log: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(true_log, cmap="magma")
    axes[0].set_title("True OD Heatmap")
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(pred_log, cmap="magma")
    axes[1].set_title("Pred OD Heatmap")
    fig.colorbar(im2, ax=axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(true_log: np.ndarray, pred_log: np.ndarray, path: str | Path) -> None:
    _save_scatter(true_log, pred_log, Path(path))


def plot_row_col_sum(true_log: np.ndarray, pred_log: np.ndarray, path: str | Path) -> None:
    _save_row_col_plot(true_log, pred_log, Path(path))


def plot_top_k_edges(true_log: np.ndarray, pred_log: np.ndarray, top_k: int, path: str | Path) -> None:
    _save_topk_plot(true_log, pred_log, top_k=top_k, path=Path(path))


def plot_distance_decay(curves: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curves["true_curve"], marker="o", label="true")
    ax.plot(curves["pred_curve"], marker="o", label="pred")
    ax.set_xlabel("Distance Bin")
    ax.set_ylabel("Mean Flow")
    ax.set_title("Distance Decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
