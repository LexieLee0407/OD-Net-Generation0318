"""模型形状 smoke 测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

if torch is not None:
    from od_zero_shot.data.samples import build_fixture_samples
    from od_zero_shot.models.autoencoder import ODAutoEncoder
    from od_zero_shot.models.diffusion import GaussianDiffusion, TinyLatentUNet
    from od_zero_shot.models.graphgps import GraphGPSRegressor
    from od_zero_shot.train.datasets import to_torch_sample
    from od_zero_shot.utils.config import ProjectConfig


@unittest.skipIf(torch is None, f"torch 不可用: {TORCH_IMPORT_ERROR}")
class ModelShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ProjectConfig()
        cls.sample = build_fixture_samples(PROJECT_ROOT, cls.config)["synthetic_toy100"]

    def test_graphgps_forward_shape(self) -> None:
        model = GraphGPSRegressor(num_layers=2)
        output = model(to_torch_sample(self.sample))
        self.assertEqual(tuple(output["pred"].shape), (100, 100))
        self.assertEqual(tuple(output["pair_condition"].shape), (self.config.model.pair_dim, 100, 100))

    def test_autoencoder_shape(self) -> None:
        model = ODAutoEncoder(latent_channels=16)
        matrix = torch.as_tensor(self.sample["y_od"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        recon, latent = model(matrix)
        self.assertEqual(tuple(recon.shape), (1, 1, 100, 100))
        self.assertEqual(tuple(latent.shape), (1, 16, 25, 25))

    def test_diffusion_shape(self) -> None:
        autoencoder = ODAutoEncoder(latent_channels=16)
        matrix = torch.as_tensor(self.sample["y_od"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        _, latent = autoencoder(matrix)
        cond = torch.zeros((1, 64, 25, 25), dtype=torch.float32)
        denoiser = TinyLatentUNet(latent_channels=16, cond_channels=64, base_channels=32)
        diffusion = GaussianDiffusion(timesteps=10)
        noisy, noise = diffusion.q_sample(latent, torch.tensor([3], dtype=torch.long))
        pred_noise = denoiser(noisy, torch.tensor([3], dtype=torch.long), cond)
        self.assertEqual(tuple(noisy.shape), tuple(latent.shape))
        self.assertEqual(tuple(noise.shape), tuple(latent.shape))
        self.assertEqual(tuple(pred_noise.shape), tuple(latent.shape))


if __name__ == "__main__":
    unittest.main()
