"""阶段化训练与评估入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from od_zero_shot.data.samples import SampleBundle, build_sample_bundle, load_or_build_sample_bundle, save_sample_bundle
from od_zero_shot.eval.metrics import compute_all_metrics
from od_zero_shot.eval.plots import save_diagnostic_plots
from od_zero_shot.models.autoencoder import ODAutoEncoder
from od_zero_shot.models.baselines import GravityModel, PairMLP
from od_zero_shot.models.diffusion import GaussianDiffusion, TinyLatentUNet
from od_zero_shot.models.graphgps import GraphGPSRegressor
from od_zero_shot.train.datasets import MatrixDataset, PairDataset, to_torch_sample
from od_zero_shot.train.losses import grouped_matrix_mse
from od_zero_shot.utils.common import ensure_dir, save_json, save_pickle, set_seed
from od_zero_shot.utils.config import ProjectConfig, save_config_snapshot


def _checkpoint_dir(project_root: str | Path) -> Path:
    return ensure_dir(Path(project_root) / "artifacts" / "checkpoints")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _select_training_samples(bundle: SampleBundle) -> list[dict[str, Any]]:
    return bundle.train if bundle.train else [bundle.fixtures["synthetic_toy100"]]


def _select_eval_sample(bundle: SampleBundle, split: str) -> dict[str, Any]:
    if split == "mini5":
        return bundle.fixtures["mini5"]
    if split == "synthetic_toy100":
        return bundle.fixtures["synthetic_toy100"]
    if split == "train" and bundle.train:
        return bundle.train[0]
    if split == "val" and bundle.val:
        return bundle.val[0]
    if split == "test" and bundle.test:
        return bundle.test[0]
    return bundle.fixtures["synthetic_toy100"]


def build_samples(project_root: str | Path, config: ProjectConfig) -> dict[str, Any]:
    """构造并缓存样本。"""
    raw_root = Path(project_root) / config.dataset.raw_root
    raw_data = None
    if raw_root.exists():
        from od_zero_shot.data.raw import load_raw_pickles

        raw_data = load_raw_pickles(raw_root)
    bundle = build_sample_bundle(project_root, config, raw_data=raw_data)
    output_path = save_sample_bundle(project_root, bundle)
    save_config_snapshot(config, Path(project_root) / "artifacts" / "config_snapshot.yaml")
    summary = dict(bundle.summary)
    summary["sample_artifact"] = str(output_path)
    return summary


def train_gravity(project_root: str | Path, config: ProjectConfig) -> dict[str, Any]:
    set_seed(config.train.seed)
    bundle = load_or_build_sample_bundle(project_root, config)
    model = GravityModel()
    samples = _select_training_samples(bundle)
    model.fit(samples)
    checkpoint_path = _checkpoint_dir(project_root) / "gravity.pkl"
    save_pickle(model, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "num_samples": len(samples)}


def train_pair_mlp(project_root: str | Path, config: ProjectConfig) -> dict[str, Any]:
    set_seed(config.train.seed)
    bundle = load_or_build_sample_bundle(project_root, config)
    device = _resolve_device(config.train.device)
    samples = _select_training_samples(bundle)
    dataset = PairDataset(samples)
    loader = DataLoader(dataset, batch_size=config.dataset.batch_size * 256, shuffle=True)
    model = PairMLP(input_dim=6, hidden_dim=config.model.hidden_dim, dropout=config.model.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr_pair_mlp, weight_decay=config.train.weight_decay)
    history = []
    for _ in range(config.train.epochs):
        model.train()
        epoch_loss = 0.0
        for pair_features, targets in loader:
            pair_features = pair_features.to(device)
            targets = targets.to(device)
            pred = model(pair_features)
            loss = torch.mean((pred - targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append(epoch_loss / max(len(loader), 1))
    checkpoint_path = _checkpoint_dir(project_root) / "pair_mlp.pt"
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "history": history}


def _build_regressor(config: ProjectConfig) -> GraphGPSRegressor:
    return GraphGPSRegressor(
        node_input_dim=3,
        edge_input_dim=3,
        struct_input_dim=2,
        lap_pe_dim=config.model.lap_pe_dim,
        hidden_dim=config.model.hidden_dim,
        pair_dim=config.model.pair_dim,
        num_layers=config.model.gps_layers,
        heads=config.model.heads,
        dropout=config.model.dropout,
    )


def train_regressor(project_root: str | Path, config: ProjectConfig) -> dict[str, Any]:
    set_seed(config.train.seed)
    bundle = load_or_build_sample_bundle(project_root, config)
    device = _resolve_device(config.train.device)
    model = _build_regressor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr_regressor, weight_decay=config.train.weight_decay)
    samples = _select_training_samples(bundle)
    history = []
    for _ in range(config.train.epochs):
        model.train()
        epoch_loss = 0.0
        for sample in samples:
            torch_sample = to_torch_sample(sample, device=device)
            output = model(torch_sample)
            loss = grouped_matrix_mse(output["pred"], torch_sample["y_od"], torch_sample["mask_diag"], torch_sample["mask_pos_off"], torch_sample["mask_zero_off"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append(epoch_loss / max(len(samples), 1))
    checkpoint_path = _checkpoint_dir(project_root) / "graphgps_regressor.pt"
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "history": history}


def train_autoencoder(project_root: str | Path, config: ProjectConfig) -> dict[str, Any]:
    set_seed(config.train.seed)
    bundle = load_or_build_sample_bundle(project_root, config)
    device = _resolve_device(config.train.device)
    samples = _select_training_samples(bundle)
    dataset = MatrixDataset(samples)
    loader = DataLoader(dataset, batch_size=config.dataset.batch_size, shuffle=True, collate_fn=lambda batch: batch)
    model = ODAutoEncoder(latent_channels=config.model.latent_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr_ae, weight_decay=config.train.weight_decay)
    history = []
    for _ in range(config.train.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch_loss = 0.0
            for sample in batch:
                torch_sample = to_torch_sample(sample, device=device)
                matrix = torch_sample["y_od"].unsqueeze(0).unsqueeze(0)
                recon, _ = model(matrix)
                batch_loss = batch_loss + grouped_matrix_mse(recon.squeeze(0).squeeze(0), torch_sample["y_od"], torch_sample["mask_diag"], torch_sample["mask_pos_off"], torch_sample["mask_zero_off"])
            batch_loss = batch_loss / max(len(batch), 1)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())
        history.append(epoch_loss / max(len(loader), 1))
    checkpoint_path = _checkpoint_dir(project_root) / "od_autoencoder.pt"
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "history": history}


def train_diffusion(project_root: str | Path, config: ProjectConfig, mode: str = "conditional") -> dict[str, Any]:
    set_seed(config.train.seed)
    bundle = load_or_build_sample_bundle(project_root, config)
    device = _resolve_device(config.train.device)
    samples = _select_training_samples(bundle)
    ae = ODAutoEncoder(latent_channels=config.model.latent_channels).to(device)
    ae.load_state_dict(torch.load(_checkpoint_dir(project_root) / "od_autoencoder.pt", map_location=device)["model_state"])
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)
    regressor = _build_regressor(config).to(device)
    regressor_ckpt = _checkpoint_dir(project_root) / "graphgps_regressor.pt"
    if regressor_ckpt.exists():
        regressor.load_state_dict(torch.load(regressor_ckpt, map_location=device)["model_state"])
    regressor.eval()
    for param in regressor.parameters():
        param.requires_grad_(False)
    denoiser = TinyLatentUNet(latent_channels=config.model.latent_channels, cond_channels=config.model.pair_dim, base_channels=config.model.hidden_dim // 2).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config.train.lr_diffusion, weight_decay=config.train.weight_decay)
    diffusion = GaussianDiffusion(config.model.diffusion_steps).to(device)
    history = []
    for _ in range(config.train.epochs):
        denoiser.train()
        epoch_loss = 0.0
        for sample in samples:
            torch_sample = to_torch_sample(sample, device=device)
            with torch.no_grad():
                latent = ae.encode(torch_sample["y_od"].unsqueeze(0).unsqueeze(0))
                cond_map = regressor(torch_sample)["pair_condition"].unsqueeze(0)
                cond_latent = torch.nn.functional.interpolate(cond_map, size=latent.shape[-2:], mode="bilinear", align_corners=False)
                if mode == "unconditional":
                    cond_latent = torch.zeros_like(cond_latent)
            loss = diffusion.training_loss(denoiser, latent, cond_latent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append(epoch_loss / max(len(samples), 1))
    checkpoint_path = _checkpoint_dir(project_root) / f"{mode}_diffusion.pt"
    torch.save({"model_state": denoiser.state_dict(), "history": history}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "history": history}


def evaluate_and_infer(project_root: str | Path, config: ProjectConfig, split: str = "synthetic_toy100") -> dict[str, Any]:
    bundle = load_or_build_sample_bundle(project_root, config)
    sample = _select_eval_sample(bundle, split=split)
    figure_dir = Path(project_root) / config.eval.figures_dir
    results: dict[str, Any] = {}

    gravity_path = _checkpoint_dir(project_root) / "gravity.pkl"
    if gravity_path.exists():
        import numpy as np

        gravity: GravityModel = torch.load if False else None  # 避免静态分析误判
        from od_zero_shot.utils.common import load_pickle

        gravity = load_pickle(gravity_path)
        gravity_pred = gravity.predict_matrix(sample)
        results["gravity"] = compute_all_metrics(sample, gravity_pred, config.eval.threshold, config.eval.top_k, config.eval.distance_bins)
        save_diagnostic_plots(sample, gravity_pred, figure_dir, prefix=f"{split}_gravity", top_k=config.eval.top_k, distance_bins=config.eval.distance_bins)

    pair_path = _checkpoint_dir(project_root) / "pair_mlp.pt"
    if pair_path.exists():
        device = _resolve_device(config.train.device)
        pair_model = PairMLP(input_dim=6, hidden_dim=config.model.hidden_dim, dropout=config.model.dropout).to(device)
        pair_model.load_state_dict(torch.load(pair_path, map_location=device)["model_state"])
        pair_model.eval()
        with torch.no_grad():
            pair_pred = pair_model(torch.as_tensor(sample["pair_baseline"], dtype=torch.float32, device=device)).cpu().numpy()
        results["pair_mlp"] = compute_all_metrics(sample, pair_pred, config.eval.threshold, config.eval.top_k, config.eval.distance_bins)
        save_diagnostic_plots(sample, pair_pred, figure_dir, prefix=f"{split}_pair_mlp", top_k=config.eval.top_k, distance_bins=config.eval.distance_bins)

    regressor_path = _checkpoint_dir(project_root) / "graphgps_regressor.pt"
    if regressor_path.exists():
        device = _resolve_device(config.train.device)
        regressor = _build_regressor(config).to(device)
        regressor.load_state_dict(torch.load(regressor_path, map_location=device)["model_state"])
        regressor.eval()
        with torch.no_grad():
            reg_pred = regressor(to_torch_sample(sample, device=device))["pred"].cpu().numpy()
        results["graphgps_regressor"] = compute_all_metrics(sample, reg_pred, config.eval.threshold, config.eval.top_k, config.eval.distance_bins)
        save_diagnostic_plots(sample, reg_pred, figure_dir, prefix=f"{split}_graphgps_regressor", top_k=config.eval.top_k, distance_bins=config.eval.distance_bins)

    ae_path = _checkpoint_dir(project_root) / "od_autoencoder.pt"
    cond_diff_path = _checkpoint_dir(project_root) / "conditional_diffusion.pt"
    if ae_path.exists() and regressor_path.exists() and cond_diff_path.exists() and sample["y_od"].shape == (100, 100):
        device = _resolve_device(config.train.device)
        ae = ODAutoEncoder(latent_channels=config.model.latent_channels).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device)["model_state"])
        ae.eval()
        regressor = _build_regressor(config).to(device)
        regressor.load_state_dict(torch.load(regressor_path, map_location=device)["model_state"])
        regressor.eval()
        denoiser = TinyLatentUNet(latent_channels=config.model.latent_channels, cond_channels=config.model.pair_dim, base_channels=config.model.hidden_dim // 2).to(device)
        denoiser.load_state_dict(torch.load(cond_diff_path, map_location=device)["model_state"])
        denoiser.eval()
        diffusion = GaussianDiffusion(config.model.diffusion_steps).to(device)
        with torch.no_grad():
            torch_sample = to_torch_sample(sample, device=device)
            cond_map = regressor(torch_sample)["pair_condition"].unsqueeze(0)
            cond_latent = torch.nn.functional.interpolate(cond_map, size=(25, 25), mode="bilinear", align_corners=False)
            latent = diffusion.sample(denoiser, (1, config.model.latent_channels, 25, 25), cond_latent, device)
            diff_pred = ae.decode(latent).squeeze(0).squeeze(0).cpu().numpy()
        results["graphgps_conditional_diffusion"] = compute_all_metrics(sample, diff_pred, config.eval.threshold, config.eval.top_k, config.eval.distance_bins)
        save_diagnostic_plots(sample, diff_pred, figure_dir, prefix=f"{split}_graphgps_conditional_diffusion", top_k=config.eval.top_k, distance_bins=config.eval.distance_bins)

    metrics_path = Path(project_root) / config.eval.metrics_path
    save_json(results, metrics_path)
    return {"metrics_path": str(metrics_path), "results": results}
