from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from experiment_manager.speculative import (
    append_speculative_submission,
    create_speculative_record,
    load_speculative_record,
    render_script,
    resolve_speculative,
)


class SpeculativePlanTests(unittest.TestCase):
    def _fixture(self, root: Path):
        run = root / "run"
        run.mkdir()
        checkpoint = root / "checkpoints" / "main"
        workload = root / "prompts.jsonl"
        workload.write_text('{"id":"one","text":"Once"}\n')
        (run / "resolved.yaml").write_text(
            yaml.safe_dump(
                {
                    "config_hash": "training-hash",
                    "execution": {
                        "checkpoint_root": str(root / "checkpoints"),
                        "experiment_name": "fixture-model",
                    },
                    "condition": {"study": "fixture-study", "name": "fixture-condition"},
                    "stages": [
                        {
                            "key": "main",
                            "mode": "main",
                            "source_iteration": None,
                            "environment": {
                                "SEED": "3407",
                                "MTP_NUM_LAYERS": "1",
                                "INPUT_MASK_RATIO": "0",
                                "TP": "2",
                                "PP": "1",
                                "CP": "1",
                                "EP": "1",
                            },
                        }
                    ],
                }
            )
        )
        config = root / "spec.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "backend": {
                        "megatron_path": "/repo",
                        "lm_eval_source": "/lm-eval",
                        "output_root": str(root / "out"),
                        "account": "a",
                        "partition": "p",
                        "nodes": 1,
                        "gpus_per_node": 2,
                        "run_time": "00:10:00",
                    },
                    "suites": {
                        "smoke": {
                            "workload": str(workload),
                            "draft_depths": [1, 2, 3, 4, 6, 8],
                            "backend_overrides": {"run_time": "00:20:00"},
                            "drafters": ["ar", "mtp"],
                            "sampling_profiles": {
                                "greedy": {"temperature": 0},
                                "temperature_1": {"temperature": 1},
                            },
                        }
                    },
                }
            )
        )
        return run, config

    def test_plan_is_non_writing_and_script_is_distributed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, config = self._fixture(root)
            analysis = resolve_speculative(
                run,
                stage_key="main",
                checkpoint_step=7,
                suite_name="smoke",
                config_path=config,
                require_checkpoint=False,
            )
            self.assertEqual(list(run.glob("speculative/**")), [])
            script = render_script(analysis, "RECORD/resolved.yaml")
            self.assertIn("--ntasks-per-node=1", script)
            self.assertIn("python -m torch.distributed.run", script)
            self.assertIn("#SBATCH --requeue", script)
            self.assertNotIn("sbatch ", script)

    def test_render_is_immutable_and_freezes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, config = self._fixture(root)
            analysis = resolve_speculative(
                run,
                stage_key="main",
                checkpoint_step=7,
                suite_name="smoke",
                config_path=config,
                require_checkpoint=False,
            )
            first = create_speculative_record(analysis)
            second = create_speculative_record(analysis)
            self.assertNotEqual(first, second)
            frozen = yaml.safe_load((first / "resolved.yaml").read_text())
            self.assertEqual(frozen["config_hash"], analysis.config_hash)
            self.assertEqual(len(frozen["suite"]["dataset_fingerprint"]), 64)
            self.assertEqual(frozen["checkpoint"]["iteration"], 7)
            self.assertEqual(frozen["analysis_record"], str(first))
            self.assertEqual(
                Path(frozen["output"]["directory"]).parts[-3:],
                ("fixture-model--training-h--main--smoke", "step_7", "speculative"),
            )
            self.assertIn("results_", Path(frozen["output"]["aggregate"]).name)
            self.assertIn("progress_speculative_", Path(frozen["output"]["progress"]).name)
            self.assertEqual(frozen["training"]["experiment_name"], "fixture-model")
            self.assertEqual(frozen["suite"]["drafters"], ["ar", "mtp"])
            self.assertEqual(frozen["backend"]["run_time"], "00:20:00")
            self.assertNotIn("backend_overrides", frozen["suite"])
            self.assertEqual(len(frozen["provenance"]["analysis_manager"]["working_tree_hash"]), 64)
            append_speculative_submission(first, "12345")
            jobs = yaml.safe_load((first / "jobs.yaml").read_text())
            self.assertEqual(jobs["submissions"][0]["job_id"], "12345")
            self.assertEqual(jobs["submissions"][0]["action"], "submit")

            loaded_path, loaded = load_speculative_record(first)
            self.assertEqual(loaded_path, first)
            self.assertEqual(loaded["config_hash"], analysis.config_hash)
            resume_command = ["sbatch", "--time=00:05:00", str(first / "submit.sh")]
            append_speculative_submission(
                first,
                "12346",
                command=resume_command,
                action="resume",
            )
            jobs = yaml.safe_load((first / "jobs.yaml").read_text())
            self.assertEqual(jobs["submissions"][1]["action"], "resume")
            self.assertEqual(jobs["submissions"][1]["command"], resume_command)


if __name__ == "__main__":
    unittest.main()
