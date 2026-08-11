from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiment_manager.config import ConfigError, load_collection, load_experiment


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = "/iopsstor/scratch/cscs/smehra"


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"SCRATCH": SCRATCH})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_span_condition_resolves_all_stages(self) -> None:
        experiment = load_experiment(ROOT / "studies/1b_masking_ablation/span_020_s2.yaml")
        self.assertEqual(experiment.experiment_name, "1b-masking-ablation__span-020-s2")
        self.assertEqual(experiment.main.environment["INPUT_MASK_RATIO"], "0.2")
        self.assertEqual(experiment.main.environment["INPUT_MASK_SPAN_LENGTH"], "2")
        self.assertEqual(len(experiment.cooldowns), 3)
        self.assertEqual(experiment.resolved["derived"]["persistent_save_interval"], 2384)

    def test_collection_has_unique_conditions(self) -> None:
        name, experiments = load_collection(ROOT / "collections/1b_masking_ablation.yaml")
        self.assertEqual(name, "1b-masking-ablation")
        self.assertEqual(len(experiments), 5)
        self.assertEqual(len({item.experiment_name for item in experiments}), 5)

    def test_unknown_override_is_rejected(self) -> None:
        condition = {
            "schema_version": 1,
            "recipe": str(ROOT / "recipes/1b_meap.yaml"),
            "name": "typo-test",
            "overrides": {"INPUT_MSAK_RATIO": 0.2},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.yaml"
            path.write_text(yaml.safe_dump(condition))
            with self.assertRaisesRegex(ConfigError, "INPUT_MSAK_RATIO"):
                load_experiment(path)

    def test_unaligned_cooldown_source_is_rejected(self) -> None:
        recipe = yaml.safe_load((ROOT / "recipes/1b_meap.yaml").read_text())
        recipe["stages"]["cooldown"]["source_iterations"] = [2385]
        condition = {"schema_version": 1, "recipe": "recipe.yaml", "name": "bad-source"}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            (directory / "condition.yaml").write_text(yaml.safe_dump(condition))
            with self.assertRaisesRegex(ConfigError, "not a persistent checkpoint"):
                load_experiment(directory / "condition.yaml")


if __name__ == "__main__":
    unittest.main()
