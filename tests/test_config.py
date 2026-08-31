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
        experiment = load_experiment(
            ROOT / "studies/1b_objective_screen/meap-span-015-s5.yaml"
        )
        self.assertEqual(
            experiment.experiment_name, "1b-objective-screen__meap-span-015-s5"
        )
        self.assertEqual(experiment.main.environment["INPUT_MASK_RATIO"], "0.15")
        self.assertEqual(experiment.main.environment["INPUT_MASK_SPAN_LENGTH"], "5")
        self.assertEqual(len(experiment.cooldowns), 1)
        self.assertEqual(experiment.resolved["derived"]["persistent_save_interval"], 9537)

    def test_each_cooldown_branch_has_its_own_token_budget(self) -> None:
        recipe = yaml.safe_load((ROOT / "recipes/1b_llama.yaml").read_text())
        recipe["execution"]["submission_script"] = str(
            ROOT / "submission/train_1b_llama.sh"
        )
        recipe["stages"]["cooldown"]["branches"] = [
            {"source_iteration": 9537, "tokens": 1000000000},
            {"source_iteration": 19074, "tokens": 10000000000},
            {"source_iteration": 34333, "tokens": 20000000000},
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
        for size in ("300m", "1b"):
            name, experiments = load_collection(
                ROOT / f"collections/{size}_objective_screen.yaml"
            )
            self.assertEqual(name, f"{size}-objective-screen")
            self.assertEqual(len(experiments), 4)
            self.assertEqual(len({item.experiment_name for item in experiments}), 4)
            objectives = {
                (
                    item.main.environment["MTP_NUM_LAYERS"],
                    item.main.environment["INPUT_MASK_RATIO"],
                    item.main.environment["INPUT_MASK_STRATEGY"],
                    item.main.environment["INPUT_MASK_SPAN_LENGTH"],
                )
                for item in experiments
            }
            self.assertEqual(
                objectives,
                {
                    ("0", "0.0", "random", "1"),
                    ("1", "0.0", "random", "1"),
                    ("0", "0.15", "random", "1"),
                    ("0", "0.15", "span", "5"),
                },
            )

    def test_model_family_sizes_and_budgets_are_frozen(self) -> None:
        small = load_experiment(ROOT / "studies/300m_objective_screen/ntp.yaml")
        large = load_experiment(ROOT / "studies/1b_objective_screen/ntp.yaml")
        self.assertEqual(
            (
                small.main.environment["NUM_LAYERS"],
                small.main.environment["HIDDEN_SIZE"],
                small.main.environment["FFN_HIDDEN_SIZE"],
                small.main.environment["NUM_ATTENTION_HEADS"],
                small.main.environment["NUM_QUERY_GROUPS"],
            ),
            ("24", "1024", "2816", "16", "4"),
        )
        self.assertEqual(
            (
                large.main.environment["NUM_LAYERS"],
                large.main.environment["HIDDEN_SIZE"],
                large.main.environment["FFN_HIDDEN_SIZE"],
                large.main.environment["NUM_ATTENTION_HEADS"],
                large.main.environment["NUM_QUERY_GROUPS"],
            ),
            ("24", "2048", "5632", "32", "8"),
        )
        self.assertEqual(small.main.environment["TRAIN_TOKENS"], "10800000000")
        self.assertEqual(small.cooldowns[0].environment["COOLDOWN_TOKENS"], "1200000000")
        self.assertEqual(large.main.environment["TRAIN_TOKENS"], "36000000000")
        self.assertEqual(large.cooldowns[0].environment["COOLDOWN_TOKENS"], "4000000000")

    def test_muon_sweep_is_short_unmasked_and_varies_optimizer_settings(self) -> None:
        name, experiments = load_collection(ROOT / "collections/1b_muon_sweep.yaml")
        self.assertEqual(name, "1b-muon-sweep")
        self.assertEqual(len(experiments), 7)
        self.assertEqual(
            {experiment.main.environment["WANDB_PROJECT"] for experiment in experiments},
            {"mask_pretraining"},
        )
        self.assertEqual(
            {experiment.main.environment["STUDY_NAME"] for experiment in experiments},
            {"1b-muon-sweep"},
        )
        self.assertIn("--time=10:00:00", experiments[0].sbatch_args)
        self.assertEqual(
            {experiment.main.environment["TRAIN_TOKENS"] for experiment in experiments},
            {"5000000000"},
        )
        self.assertEqual(
            {experiment.main.environment["WARMUP_STEPS"] for experiment in experiments},
            {"500"},
        )
        self.assertEqual(
            {experiment.main.environment["INPUT_MASK_RATIO"] for experiment in experiments},
            {"0.0"},
        )
        self.assertEqual(
            {experiment.main.environment["OPTIMIZER"] for experiment in experiments},
            {"muon"},
        )
        self.assertEqual(
            {experiment.main.environment["PEAK_LR"] for experiment in experiments},
            {"0.0002", "0.0004", "0.0008"},
        )
        self.assertEqual(
            {
                experiment.main.environment["MUON_EXTRA_SCALE_FACTOR"]
                for experiment in experiments
            },
            {"0.2", "0.5", "1.0"},
        )
        self.assertEqual(
            {
                (
                    experiment.main.environment["PEAK_LR"],
                    experiment.main.environment["MUON_EXTRA_SCALE_FACTOR"],
                )
                for experiment in experiments
            },
            {
                ("0.0002", "1.0"),
                ("0.0002", "0.2"),
                ("0.0004", "1.0"),
                ("0.0004", "0.5"),
                ("0.0004", "0.2"),
                ("0.0008", "1.0"),
                ("0.0008", "0.2"),
            },
        )

    def test_main_recipe_uses_selected_muon_configuration(self) -> None:
        experiment = load_experiment(ROOT / "studies/1b_objective_screen/ntp.yaml")
        self.assertEqual(experiment.main.environment["WARMUP_STEPS"], "1000")
        self.assertEqual(experiment.main.environment["PEAK_LR"], "0.0008")
        self.assertEqual(experiment.main.environment["MIN_LR"], "8e-05")
        self.assertEqual(experiment.main.environment["MUON_EXTRA_SCALE_FACTOR"], "0.2")

    def test_objective_screens_tolerate_transient_cluster_stalls(self) -> None:
        for size in ("300m", "1b"):
            experiment = load_experiment(
                ROOT / f"studies/{size}_objective_screen/ntp.yaml"
            )
            self.assertEqual(
                experiment.main.environment["DISTRIBUTED_TIMEOUT_MINUTES"], "60"
            )

    def test_300m_muon_sweep_matches_production_except_calibration_controls(self) -> None:
        name, experiments = load_collection(ROOT / "collections/300m_muon_sweep.yaml")
        production = load_experiment(ROOT / "studies/300m_objective_screen/ntp.yaml")
        self.assertEqual(name, "300m-muon-sweep")
        self.assertEqual(len(experiments), 7)

        frozen_keys = (
            "DATASETS",
            "TOKENIZER_MODEL",
            "MBS",
            "GBS",
            "SEQ_LEN",
            "NUM_LAYERS",
            "HIDDEN_SIZE",
            "FFN_HIDDEN_SIZE",
            "NUM_ATTENTION_HEADS",
            "NUM_QUERY_GROUPS",
            "INIT_METHOD_STD",
            "WEIGHT_DECAY",
            "OPTIMIZER",
            "MUON_MOMENTUM",
            "MUON_NESTEROV",
            "MUON_SCALE_MODE",
            "MUON_NUM_NS_STEPS",
            "MUON_SCALAR_OPTIMIZER",
        )
        for experiment in experiments:
            for key in frozen_keys:
                self.assertEqual(
                    experiment.main.environment[key], production.main.environment[key]
                )
            self.assertEqual(experiment.main.environment["TRAIN_TOKENS"], "1500000000")
            self.assertEqual(experiment.main.environment["WARMUP_STEPS"], "150")
            self.assertEqual(experiment.main.environment["INPUT_MASK_RATIO"], "0.0")
            self.assertEqual(experiment.main.environment["MTP_NUM_LAYERS"], "0")

        self.assertEqual(
            {
                (
                    experiment.main.environment["PEAK_LR"],
                    experiment.main.environment["MUON_EXTRA_SCALE_FACTOR"],
                )
                for experiment in experiments
            },
            {
                ("0.0004", "1.0"),
                ("0.0004", "0.2"),
                ("0.0008", "1.0"),
                ("0.0008", "0.5"),
                ("0.0008", "0.2"),
                ("0.0008", "0.1"),
                ("0.0016", "0.2"),
            },
        )

    def test_smoke_run_is_short_and_isolated(self) -> None:
        experiment = load_experiment(ROOT / "studies/test_run/test_run.yaml")
        self.assertEqual(experiment.experiment_name, "mistral-v03-test_run")
        self.assertEqual(experiment.project_name, "mask_pretraining_test")
        self.assertEqual(experiment.main.environment["TRAIN_TOKENS"], "100000000")
        self.assertEqual(experiment.main.environment["MBS"], "8")
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
