from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiment_manager.config import load_experiment
from experiment_manager.evaluation import (
    create_evaluation_record,
    load_evaluation_config,
    render_spellbook_script,
    resolve_evaluation,
)
from experiment_manager.records import create_run_record


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = "/iopsstor/scratch/cscs/smehra"
EVAL_CONFIG = ROOT / "evaluations/suites.yaml"


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"SCRATCH": SCRATCH})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _training_run(self, directory: Path) -> tuple[Path, int]:
        recipe = yaml.safe_load((ROOT / "recipes/test_run.yaml").read_text())
        recipe["execution"]["submission_script"] = str(
            ROOT / "submission/train_1b_llama.sh"
        )
        recipe["execution"]["artifacts_root"] = str(directory / "runs")
        recipe["execution"]["checkpoint_root"] = str(
            directory / "checkpoints/{experiment}"
        )
        recipe_path = directory / "recipe.yaml"
        recipe_path.write_text(yaml.safe_dump(recipe))
        condition_path = directory / "condition.yaml"
        condition_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "recipe": "recipe.yaml",
                    "study": "eval-test",
                    "name": "ntp",
                    "overrides": {"INPUT_MASK_RATIO": 0.0},
                }
            )
        )
        experiment = load_experiment(condition_path)
        run_dir = create_run_record(experiment)
        step = 100
        checkpoint_dir = experiment.checkpoint_root / "main"
        (checkpoint_dir / f"iter_{step:07d}").mkdir(parents=True)
        (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(f"{step}\n")
        return run_dir, step

    def test_named_suites_are_cost_tiered_and_code_is_unsafe(self) -> None:
        _, suites, _ = load_evaluation_config(EVAL_CONFIG)
        self.assertEqual(
            set(suites),
            {"smoke", "core", "math", "code"},
        )
        self.assertEqual(suites["smoke"].limit, 20)
        self.assertFalse(suites["math"].unsafe_code)
        self.assertTrue(suites["code"].unsafe_code)

    def test_resolves_latest_native_megatron_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            run_dir, step = self._training_run(Path(directory_value))
            evaluation = resolve_evaluation(
                run_dir,
                stage_key="main",
                checkpoint_step=None,
                suite_name="core",
                config_path=EVAL_CONFIG,
            )

            self.assertEqual(evaluation.checkpoint_step, step)
            self.assertEqual(evaluation.checkpoint_dir.name, "main")
            self.assertEqual(evaluation.training["objective"], "ntp")
            self.assertEqual(
                evaluation.training["tokens_seen"],
                step * 1024 * 4096,
            )
            script = render_spellbook_script(evaluation, log_dir=Path("logs"))
            self.assertIn(f"load={evaluation.checkpoint_dir}", script)
            self.assertIn("--model megatron_lm", script)
            self.assertIn("--cache_requests true", script)
            self.assertIn("WANDB_RUN_GROUP=\"eval-test\"", script)
            self.assertNotIn("--confirm_run_unsafe_code", script)

    def test_code_suite_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            run_dir, step = self._training_run(Path(directory_value))
            with self.assertRaisesRegex(ValueError, "allow-unsafe-code"):
                resolve_evaluation(
                    run_dir,
                    stage_key="main",
                    checkpoint_step=step,
                    suite_name="code",
                    config_path=EVAL_CONFIG,
                )
            evaluation = resolve_evaluation(
                run_dir,
                stage_key="main",
                checkpoint_step=step,
                suite_name="code",
                config_path=EVAL_CONFIG,
                allow_unsafe_code=True,
            )
            script = render_spellbook_script(evaluation, log_dir=Path("logs"))
            self.assertIn("--confirm_run_unsafe_code", script)
            self.assertIn('export HF_ALLOW_CODE_EVAL="1"', script)

    def test_render_creates_immutable_eval_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            run_dir, step = self._training_run(Path(directory_value))
            evaluation = resolve_evaluation(
                run_dir,
                stage_key="main",
                checkpoint_step=step,
                suite_name="smoke",
                config_path=EVAL_CONFIG,
            )
            record = create_evaluation_record(evaluation)

            self.assertTrue((record / "resolved.yaml").is_file())
            self.assertTrue((record / "submit.sh").is_file())
            self.assertTrue((record / "jobs.yaml").is_file())
            self.assertTrue((record / "slurm").is_dir())
            resolved = yaml.safe_load((record / "resolved.yaml").read_text())
            self.assertEqual(resolved["checkpoint_step"], step)
            self.assertEqual(resolved["suite"]["name"], "smoke")
            self.assertEqual(resolved["wandb_project"], "mask_pretraining")
            self.assertIn("--limit 20", (record / "submit.sh").read_text())


if __name__ == "__main__":
    unittest.main()
