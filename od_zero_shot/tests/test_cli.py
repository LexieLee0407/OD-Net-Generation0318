"""CLI 最小集成测试。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


class CLISmokeTest(unittest.TestCase):
    def test_build_samples_with_fixture(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "od_zero_shot.cli",
                "build_samples",
                "--config",
                "od_zero_shot/configs/default.yaml",
                "--fixture",
                "synthetic_toy100",
            ],
            cwd=PROJECT_ROOT.parent,
            env={"PYTHONPATH": str(SRC_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("train", result.stdout)

    def test_minimal_train_eval_smoke_on_toy100(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            config_path = tmp_root / "smoke.yaml"
            built_root = tmp_root / "datasets"
            checkpoint_dir = tmp_root / "checkpoints"
            figures_dir = tmp_root / "figures"
            metrics_path = tmp_root / "metrics" / "metrics.json"
            payload = {
                "dataset": {
                    "raw_root": "data/ny_state",
                    "sample_size": 100,
                    "knn_k": 8,
                    "ordering": "xy",
                    "neighbor_metric": "haversine",
                    "split_mode": "county",
                    "built_root": str(built_root),
                    "heldout_counties": ["061"],
                    "val_counties": ["047"],
                    "num_train_samples": 1,
                    "num_val_samples": 0,
                    "num_test_samples": 0,
                    "max_node_overlap": 0.3,
                    "batch_size": 1,
                },
                "model": {
                    "gps_layers": 2,
                    "hidden_dim": 32,
                    "heads": 4,
                    "dropout": 0.1,
                    "lap_pe_dim": 8,
                    "rw_steps": 2,
                    "pair_dim": 16,
                    "latent_channels": 8,
                    "diffusion_steps": 5,
                },
                "train": {
                    "optimizer": "AdamW",
                    "lr_gravity": 0.001,
                    "lr_pair_mlp": 0.0002,
                    "lr_regressor": 0.0002,
                    "lr_ae": 0.001,
                    "lr_diffusion": 0.0002,
                    "weight_decay": 0.0001,
                    "epochs": {"gravity": 1, "pair_mlp": 1, "regressor": 1, "ae": 1, "diffusion": 1},
                    "device": "cpu",
                    "seed": 20260317,
                },
                "eval": {
                    "threshold": 0.0,
                    "top_k": 3,
                    "distance_bins": [0.0, 5.0, 20.0, 80.0, 320.0],
                    "figures_dir": str(figures_dir),
                    "metrics_path": str(metrics_path),
                    "device": "cpu",
                },
            }
            config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
            env = {"PYTHONPATH": str(SRC_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}

            def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-m", "od_zero_shot.cli", *args],
                    cwd=PROJECT_ROOT.parent,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )

            run_cli("build_samples", "--config", str(config_path), "--fixture", "synthetic_toy100")
            run_cli("train_regressor", "--config", str(config_path), "--fixture", "synthetic_toy100", "--checkpoint-dir", str(checkpoint_dir))
            run_cli("train_ae", "--config", str(config_path), "--fixture", "synthetic_toy100", "--checkpoint-dir", str(checkpoint_dir))
            run_cli(
                "train_diffusion",
                "--config",
                str(config_path),
                "--fixture",
                "synthetic_toy100",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--ae-checkpoint",
                str(checkpoint_dir / "od_autoencoder.pt"),
            )
            run_cli(
                "train_diffusion",
                "--config",
                str(config_path),
                "--fixture",
                "synthetic_toy100",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--conditional",
                "--regressor-checkpoint",
                str(checkpoint_dir / "graphgps_regressor.pt"),
                "--ae-checkpoint",
                str(checkpoint_dir / "od_autoencoder.pt"),
            )
            reg_eval = run_cli(
                "evaluate_infer",
                "--config",
                str(config_path),
                "--fixture",
                "synthetic_toy100",
                "--split",
                "train",
                "--model-kind",
                "regressor",
                "--checkpoint",
                str(checkpoint_dir / "graphgps_regressor.pt"),
            )
            diff_eval = run_cli(
                "evaluate_infer",
                "--config",
                str(config_path),
                "--fixture",
                "synthetic_toy100",
                "--split",
                "train",
                "--model-kind",
                "conditional_diffusion",
                "--checkpoint",
                str(checkpoint_dir / "conditional_diffusion.pt"),
                "--regressor-checkpoint",
                str(checkpoint_dir / "graphgps_regressor.pt"),
                "--ae-checkpoint",
                str(checkpoint_dir / "od_autoencoder.pt"),
            )
            self.assertIn("metrics_path", reg_eval.stdout)
            self.assertIn("metrics_path", diff_eval.stdout)
            self.assertTrue((tmp_root / "metrics" / "train_regressor_metrics.json").exists())
            self.assertTrue((tmp_root / "metrics" / "train_conditional_diffusion_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
