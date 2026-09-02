import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_manager.paired_eval import (
    CORE_TASKS,
    Condition,
    Example,
    PairingError,
    compare,
    exact_mcnemar,
    load_condition,
    main,
    markdown_report,
)


def example(task="arc_easy", doc_id="0", correct=1.0, scores=(-3.0, -1.0), gold=1):
    return Example(
        task, doc_id, f"hash-{doc_id}", "cfg", correct, scores, (-1.5, -0.25), gold, "acc_norm"
    )


def condition(name, examples_by_task):
    tasks = {task: {x.identity: x for x in values} for task, values in examples_by_task.items()}
    return Condition(name, Path(name), tasks)


class PairedEvalTests(unittest.TestCase):
    def test_exact_margins_and_option_probability(self):
        item = example(scores=(-4.0, -1.0), gold=1)
        self.assertEqual(item.margin(), 3.0)
        self.assertEqual(item.margin(normalized=True), 1.25)
        self.assertGreater(item.option_probability(), 0.95)

    def test_matching_is_independent_of_order(self):
        a = condition("a", {"arc_easy": [example(doc_id="0"), example(doc_id="1", correct=0)]})
        b = condition("b", {"arc_easy": [example(doc_id="1", correct=1), example(doc_id="0")]})
        result = compare(a, b, seed=7, resamples=100)
        self.assertEqual(result["tasks"]["arc_easy"]["accuracy_difference"]["estimate"], -0.5)

    def test_missing_task_and_example_fail(self):
        a = condition("a", {"arc_easy": [example()]})
        with self.assertRaisesRegex(PairingError, "task sets differ"):
            compare(a, condition("b", {"piqa": [example(task="piqa")]}), resamples=100)
        with self.assertRaisesRegex(PairingError, "example sets differ"):
            compare(a, condition("b", {"arc_easy": [example(doc_id="2")]}), resamples=100)

    def test_config_mismatch_fails(self):
        left = example()
        right = Example(**{**left.__dict__, "config_hash": "other"})
        with self.assertRaisesRegex(PairingError, "configuration/dataset"):
            compare(
                condition("a", {"arc_easy": [left]}),
                condition("b", {"arc_easy": [right]}),
                resamples=100,
            )

    def test_bootstrap_is_deterministic(self):
        a = condition("a", {"arc_easy": [example(doc_id=str(i), correct=i % 2) for i in range(8)]})
        b = condition(
            "b",
            {"arc_easy": [example(doc_id=str(i), correct=(i + 1) % 2) for i in range(8)]},
        )
        self.assertEqual(compare(a, b, 91, 200), compare(a, b, 91, 200))

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar(1, 5), 0.21875)

    def test_fixed_task_macro(self):
        a_tasks = {task: [example(task=task, correct=1)] for task in CORE_TASKS}
        b_tasks = {task: [example(task=task, correct=0)] for task in CORE_TASKS}
        result = compare(condition("a", a_tasks), condition("b", b_tasks), 1, 100)
        self.assertEqual(result["core_unweighted_macro_accuracy_difference"]["estimate"], 1.0)

    def test_realistic_lm_eval_sample_and_cli_outputs(self):
        fixture = Path(__file__).parent / "fixtures" / "lm_eval_v0412"
        loaded = load_condition("fixture", fixture)
        item = next(iter(loaded.tasks["arc_easy"].values()))
        self.assertEqual(item.raw_scores, (-2.0, -0.5))
        self.assertEqual(item.gold, 1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            arguments = [
                "--condition", f"A={fixture}", "--condition", f"B={fixture}",
                "--resamples", "100", "--json", str(out / "x.json"),
                "--markdown", str(out / "x.md"),
            ]
            self.assertEqual(main(arguments), 0)
            payload = json.loads((out / "x.json").read_text())
            self.assertEqual(payload["schema_version"], "paired-eval/v1")
            self.assertIn("schema: paired-eval-markdown/v1", (out / "x.md").read_text())

    def test_timestamp_matching_ignores_preexisting_aggregate_only_result(self):
        fixture = Path(__file__).parent / "fixtures" / "lm_eval_v0412"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in fixture.iterdir():
                (root / source.name).write_bytes(source.read_bytes())
            (root / "results_2025-12-31T00-00-00.json").write_text("{}")
            loaded = load_condition("fixture", root)
            self.assertEqual(set(loaded.tasks), {"arc_easy"})

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = {"configs": {"arc_easy": {}}, "versions": {}}
            (root / "results_2026-01-01T00-00-00.json").write_text(json.dumps(result))
            fixture = Path(__file__).parent / "fixtures/lm_eval_v0412"
            record = json.loads(
                (fixture / "samples_arc_easy_2026-01-01T00-00-00.jsonl").read_text()
            )
            duplicate = json.dumps(record) + "\n" + json.dumps(record) + "\n"
            (root / "samples_arc_easy_2026-01-01T00-00-00.jsonl").write_text(duplicate)
            with self.assertRaisesRegex(PairingError, "duplicate"):
                load_condition("bad", root)

    def test_markdown_schema(self):
        a = condition("a", {"arc_easy": [example()]})
        comparison = compare(a, condition("b", {"arc_easy": [example()]}), resamples=100)
        text = markdown_report([comparison])
        self.assertIn("Accuracy Δ", text)

    def test_slurm_submission_is_cpu_only_and_does_not_analyze_locally(self):
        fixture = Path(__file__).parent / "fixtures" / "lm_eval_v0412"
        arguments = [
            "--condition", f"A={fixture}", "--condition", f"B={fixture}",
            "--resamples", "100", "--json", "paired.json",
            "--markdown", "paired.md", "--submit-slurm",
        ]
        with patch("experiment_manager.paired_eval.submit", return_value="12345") as mocked:
            self.assertEqual(main(arguments), 0)
        command = mocked.call_args.args[0]
        self.assertEqual(command[0], "sbatch")
        self.assertIn("--cpus-per-task=1", command)
        self.assertIn("--ntasks=1", command)
        self.assertFalse(any("gres" in value or "gpu" in value for value in command))
        wrap = next(value for value in command if value.startswith("--wrap="))
        self.assertNotIn("--submit-slurm", wrap)
        self.assertIn("experiment_manager.paired_eval", wrap)
        self.assertIn("srun --ntasks=1 --mpi=pmix --environment=test-env python", wrap)
