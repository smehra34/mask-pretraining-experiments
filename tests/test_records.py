from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiment_manager.config import load_experiment
from experiment_manager.records import create_run_record, load_run_record, shell_command


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = "/iopsstor/scratch/cscs/smehra"


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"SCRATCH": SCRATCH})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _temporary_experiment(self, directory: Path):
        recipe = yaml.safe_load((ROOT / "recipes/1b_meap.yaml").read_text())
        recipe["execution"]["artifacts_root"] = str(directory / "runs")
        recipe_path = directory / "recipe.yaml"
        recipe_path.write_text(yaml.safe_dump(recipe))
        condition_path = directory / "condition.yaml"
        condition_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "recipe": "recipe.yaml",
                    "study": "test-study",
                    "name": "record-test",
                    "overrides": {"INPUT_MASK_RATIO": 0.1},
                }
            )
        )
        return load_experiment(condition_path)

    def test_command_contains_execution_context_and_exports(self) -> None:
        experiment = load_experiment(ROOT / "studies/1b_masking_ablation/vanilla.yaml")
        command = shell_command(experiment, experiment.main)
        self.assertIn("--time=02:00:00", command)
        self.assertIn(
            "--chdir=/users/smehra/developer/mask-pretraining",
            command,
        )
        export = next(value for value in command if value.startswith("--export="))
        self.assertIn("RUN_MODE=main", export)
        self.assertIn("INPUT_MASK_TOKEN=<SPECIAL_999>", export)

    def test_render_creates_complete_immutable_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            experiment = self._temporary_experiment(Path(directory_value))
            run_dir = create_run_record(experiment)
            self.assertTrue((run_dir / "resolved.yaml").is_file())
            self.assertTrue((run_dir / "metadata.yaml").is_file())
            self.assertTrue((run_dir / "jobs.yaml").is_file())
            self.assertTrue((run_dir / "scripts/submit-main.sh").is_file())
            loaded_dir, resolved = load_run_record(run_dir)
            self.assertEqual(loaded_dir, run_dir)
            self.assertEqual(resolved["condition"]["name"], "record-test")


if __name__ == "__main__":
    unittest.main()
