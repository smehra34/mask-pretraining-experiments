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
        self.assertEqual(experiment.resolved["derived"]["persistent_save_interval"], 9537)

    def test_each_cooldown_branch_has_its_own_token_budget(self) -> None:
        recipe = yaml.safe_load((ROOT / "recipes/1b_llama.yaml").read_text())
        recipe["execution"]["submission_script"] = str(
            ROOT / "submission/train_1b_llama.sh"
        )
        recipe["stages"]["cooldown"]["branches"] = [
            {"source_iteration": 9537, "tokens": 1000000000},
            {"source_iteration": 47685, "tokens": 10000000000},
            {"source_iteration": 95370, "tokens": 20000000000},
        ]
        condition = {"schema_version": 1, "recipe": "recipe.yaml", "name": "varied-lengths"}
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            (directory / "condition.yaml").write_text(yaml.safe_dump(condition))
            experiment = load_experiment(directory / "condition.yaml")
        self.assertEqual(
            [stage.environment["COOLDOWN_TOKENS"] for stage in experiment.cooldowns],
            ["1000000000", "10000000000", "20000000000"],
        )

    def test_collection_has_unique_conditions(self) -> None:
        name, experiments = load_collection(ROOT / "collections/1b_masking_ablation.yaml")
        self.assertEqual(name, "1b-masking-ablation")
        self.assertEqual(len(experiments), 5)
        self.assertEqual(len({item.experiment_name for item in experiments}), 5)

    def test_smoke_run_is_short_and_isolated(self) -> None:
        experiment = load_experiment(ROOT / "studies/test_run/test_run.yaml")
        self.assertEqual(experiment.experiment_name, "mistral-v03-test_run")
        self.assertEqual(experiment.project_name, "mask_pretraining_test")
        self.assertEqual(experiment.main.environment["TRAIN_TOKENS"], "100000000")
        self.assertEqual(
            [stage.environment["COOLDOWN_TOKENS"] for stage in experiment.cooldowns],
            ["10000000"],
        )
        self.assertEqual(experiment.cooldowns[0].source_iteration, 24)
        self.assertEqual(experiment.main.environment["ROLLING_CHECKPOINTS"], "true")
        self.assertEqual(
            experiment.main.environment["ROLLING_SAVE_EVERY_TOKENS"], "50000000"
        )
        self.assertIn("--time=00:45:00", experiment.sbatch_args)
        self.assertEqual(experiment.resolved["derived"]["main_iterations"], 24)

    def test_long_context_recipe_is_single_complete_extension_stage(self) -> None:
        condition = {
            "schema_version": 1,
            "recipe": str(ROOT / "recipes/1b_llama_long_context.yaml"),
            "study": "long-context",
            "name": "ntp",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.yaml"
            path.write_text(yaml.safe_dump(condition))
            experiment = load_experiment(path)
        self.assertEqual(len(experiment.stages), 1)
        self.assertEqual(experiment.main.mode, "extension")
        self.assertEqual(experiment.main.environment["TRAIN_TOKENS"], "4000000000")
        self.assertEqual(experiment.main.environment["ROTARY_BASE"], "640000")
        self.assertEqual(experiment.resolved["derived"]["tokens_per_iteration"], 4194304)
        self.assertEqual(experiment.resolved["derived"]["main_iterations"], 954)

    def test_unknown_override_is_rejected(self) -> None:
        condition = {
            "schema_version": 1,
            "recipe": str(ROOT / "recipes/1b_llama.yaml"),
            "name": "typo-test",
            "overrides": {"INPUT_MSAK_RATIO": 0.2},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.yaml"
            path.write_text(yaml.safe_dump(condition))
            with self.assertRaisesRegex(ConfigError, "INPUT_MSAK_RATIO"):
                load_experiment(path)

    def test_unaligned_cooldown_source_is_rejected(self) -> None:
        recipe = yaml.safe_load((ROOT / "recipes/1b_llama.yaml").read_text())
        recipe["execution"]["submission_script"] = str(
            ROOT / "submission/train_1b_llama.sh"
        )
        recipe["stages"]["cooldown"]["branches"] = [
            {"source_iteration": 2385, "tokens": 10000000000}
        ]
        condition = {"schema_version": 1, "recipe": "recipe.yaml", "name": "bad-source"}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "recipe.yaml").write_text(yaml.safe_dump(recipe))
            (directory / "condition.yaml").write_text(yaml.safe_dump(condition))
            with self.assertRaisesRegex(ConfigError, "not a persistent checkpoint"):
                load_experiment(directory / "condition.yaml")


if __name__ == "__main__":
    unittest.main()
