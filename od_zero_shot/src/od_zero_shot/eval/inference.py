"""统一推理与评估入口。"""

from __future__ import annotations

from pathlib import Path

import torch

from od_zero_shot.data.dataset import sample_to_tensor_dict
from od_zero_shot.data.fixtures import load_fixture
from od_zero_shot.data.sample_builder import build_single_fixture_sample, load_manifest_paths, load_sample
from od_zero_shot.eval.metrics import compute_all_metrics
from od_zero_shot.eval.plots import plot_distance_decay, plot_heatmap, plot_row_col_sum, plot_scatter, plot_top_k_edges
from od_zero_shot.models.autoencoder import ODAutoEncoder
from od_zero_shot.models.baselines import GravityModel, PairMLP, build_pair_features_torch, gravity_predict_sample
from od_zero_shot.models.diffusion import ConditionalLatentDiffusion
from od_zero_shot.models.graphgps import GraphGPSRegressor
from od_zero_shot.utils.common import choose_device, save_json


def _load_eval_samples(manifest_path: str | None, split: str, fixture_name: str | None):
    if fixture_name is not None:
        return [build_single_fixture_sample(load_fixture(fixture_name), split=split, knn_k=8).to_numpy_dict()]
    if manifest_path is None:
        raise ValueError("评估需要 manifest_path 或 fixture。")
    return [load_sample(path).to_numpy_dict() for path in load_manifest_paths(manifest_path, split)]


def evaluate_model(dataset_cfg, model_cfg, eval_cfg, model_kind: str, checkpoint: str, manifest_path: str | None, split: str, fixture_name: str | None, regressor_checkpoint: str | None = None, ae_checkpoint: str | None = None) -> dict[str, object]:
    samples = _load_eval_samples(manifest_path=manifest_path, split=split, fixture_name=fixture_name)
    figures_dir = Path(eval_cfg.figures_dir) / model_kind
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_collection = []
    device = choose_device("cpu")

    gravity = None
    pair_mlp = None
    regressor = None
    autoencoder = None
    diffusion = None
    if model_kind == "gravity":
        gravity = GravityModel.load(checkpoint)
    elif model_kind == "pair_mlp":
        pair_mlp = PairMLP(hidden_dim=model_cfg.hidden_dim, dropout=model_cfg.dropout).to(device)
        pair_mlp.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        pair_mlp.eval()
    elif model_kind == "regressor":
        regressor = GraphGPSRegressor(hidden_dim=model_cfg.hidden_dim, heads=model_cfg.heads, num_layers=model_cfg.gps_layers, pair_dim=model_cfg.pair_dim, dropout=model_cfg.dropout).to(device)
        regressor.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        regressor.eval()
    elif model_kind == "diffusion":
        if regressor_checkpoint is None or ae_checkpoint is None:
            raise ValueError("评估 diffusion 需要同时提供 regressor_checkpoint 与 ae_checkpoint。")
        regressor = GraphGPSRegressor(hidden_dim=model_cfg.hidden_dim, heads=model_cfg.heads, num_layers=model_cfg.gps_layers, pair_dim=model_cfg.pair_dim, dropout=model_cfg.dropout).to(device)
        regressor.load_state_dict(torch.load(regressor_checkpoint, map_location=device)["state_dict"])
        regressor.eval()
        autoencoder = ODAutoEncoder(latent_channels=model_cfg.latent_channels).to(device)
        autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device)["state_dict"])
        autoencoder.eval()
        diffusion = ConditionalLatentDiffusion(latent_channels=model_cfg.latent_channels, pair_dim=model_cfg.pair_dim, diffusion_steps=model_cfg.diffusion_steps, conditional=True).to(device)
        diffusion.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        diffusion.eval()
    else:
        raise ValueError(f"未知模型类型: {model_kind}")

    for idx, sample in enumerate(samples):
        true_log = sample["y_od"]
        if model_kind == "gravity":
            pred_log = gravity_predict_sample(gravity, sample)
        elif model_kind == "pair_mlp":
            tensor_sample = sample_to_tensor_dict(sample, device=device)
            with torch.no_grad():
                pred_log = pair_mlp(build_pair_features_torch({"pair_baseline": tensor_sample["pair_baseline"].unsqueeze(0)})).squeeze(0).cpu().numpy()
        elif model_kind == "regressor":
            tensor_sample = sample_to_tensor_dict(sample, device=device)
            batch = {key: value.unsqueeze(0) for key, value in tensor_sample.items()}
            with torch.no_grad():
                pred_log = regressor(batch)["y_pred"].squeeze(0).cpu().numpy()
        else:
            tensor_sample = sample_to_tensor_dict(sample, device=device)
            batch = {key: value.unsqueeze(0) for key, value in tensor_sample.items()}
            with torch.no_grad():
                pair_condition = regressor(batch)["pair_condition_map"]
                latent = diffusion.sample(num_samples=1, device=device, pair_condition=pair_condition)
                pred_log = autoencoder.decode(latent).squeeze(0).squeeze(0).cpu().numpy()

        metrics = compute_all_metrics(sample=sample, pred_log=pred_log, threshold=eval_cfg.threshold, top_k=eval_cfg.top_k, distance_bins=eval_cfg.distance_bins)
        metrics["sample_id"] = str(sample["sample_id"])
        metrics_collection.append(metrics)
        plot_heatmap(true_log=true_log, pred_log=pred_log, path=figures_dir / f"{idx:03d}_heatmap.png")
        plot_scatter(true_log=true_log, pred_log=pred_log, path=figures_dir / f"{idx:03d}_scatter.png")
        plot_row_col_sum(true_log=true_log, pred_log=pred_log, path=figures_dir / f"{idx:03d}_row_col_sum.png")
        plot_top_k_edges(true_log=true_log, pred_log=pred_log, top_k=eval_cfg.top_k, path=figures_dir / f"{idx:03d}_topk.png")
        plot_distance_decay(curves=metrics["distance_decay_curve"], path=figures_dir / f"{idx:03d}_distance_decay.png")

    output = {"model_kind": model_kind, "split": split, "metrics": metrics_collection}
    save_json(eval_cfg.metrics_path, output)
    return output
