"""数据层测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from od_zero_shot.data.fixtures import build_synthetic_toy100_raw, load_mini5_fixture
from od_zero_shot.data.raw import RawMobilityData
from od_zero_shot.data.samples import build_fixture_samples, build_sample_bundle
from od_zero_shot.utils.config import ProjectConfig


class DataPipelineTest(unittest.TestCase):
    def test_load_mini5_fixture(self) -> None:
        centroid, population, od2flow = load_mini5_fixture(PROJECT_ROOT / "data" / "fixtures")
        self.assertEqual(len(centroid), 5)
        self.assertEqual(len(population), 5)
        self.assertIn(("36055011703", "36069050101"), od2flow)

    def test_build_fixture_samples(self) -> None:
        config = ProjectConfig()
        fixtures = build_fixture_samples(PROJECT_ROOT, config)
        mini = fixtures["mini5"]
        toy = fixtures["synthetic_toy100"]
        self.assertEqual(mini["y_od"].shape, (5, 5))
        self.assertEqual(toy["y_od"].shape, (100, 100))
        total = mini["mask_diag"].sum() + mini["mask_pos_off"].sum() + mini["mask_zero_off"].sum()
        self.assertEqual(int(total), 25)

    def test_county_split_has_no_overlap(self) -> None:
        config = ProjectConfig()
        config.dataset.heldout_counties = ["029"]
        config.dataset.val_counties = []
        config.dataset.num_train_samples = 4
        centroid, population, od2flow = build_synthetic_toy100_raw()
        raw = RawMobilityData(centroid=centroid, population=population, od2flow=od2flow)
        bundle = build_sample_bundle(PROJECT_ROOT, config, raw_data=raw)
        for sample in bundle.train:
            self.assertEqual({node_id[2:5] for node_id in sample["node_ids"]}, {"001"})
        for sample in bundle.test:
            self.assertEqual({node_id[2:5] for node_id in sample["node_ids"]}, {"029"})

    def test_geometry_graph_does_not_depend_on_od_edges(self) -> None:
        config = ProjectConfig()
        centroid, population, od2flow = build_synthetic_toy100_raw()
        raw_a = RawMobilityData(centroid=centroid, population=population, od2flow=od2flow)
        raw_b = RawMobilityData(centroid=centroid, population=population, od2flow={})
        bundle_a = build_sample_bundle(PROJECT_ROOT, config, raw_data=raw_a)
        bundle_b = build_sample_bundle(PROJECT_ROOT, config, raw_data=raw_b)
        self.assertTrue((bundle_a.fixtures["synthetic_toy100"]["edge_index"] == bundle_b.fixtures["synthetic_toy100"]["edge_index"]).all())


if __name__ == "__main__":
    unittest.main()

