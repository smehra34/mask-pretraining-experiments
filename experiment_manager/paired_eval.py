"""Paired sample-level analysis for lm-eval v0.4.12 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from experiment_manager.slurm import submit


SCHEMA_VERSION = "paired-eval/v1"
CORE_TASKS = (
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "boolq",
)


class PairingError(ValueError):
    """Raised when two evaluation artifacts cannot be paired safely."""


@dataclass(frozen=True)
class Example:
    task: str
    doc_id: str
    doc_hash: str
    config_hash: str
    correct: float
    raw_scores: tuple[float, ...] | None
    normalized_scores: tuple[float, ...] | None
    gold: int | None
    metric: str

    @property
    def identity(self) -> tuple[str, str]:
        return self.doc_id, self.doc_hash

    def margin(self, normalized: bool = False) -> float | None:
        scores = self.normalized_scores if normalized else self.raw_scores
        if scores is None or self.gold is None or len(scores) < 2:
            return None
        return scores[self.gold] - max(v for i, v in enumerate(scores) if i != self.gold)

    def option_probability(self) -> float | None:
        probabilities = self.option_probabilities()
        return (
            probabilities[self.gold]
            if probabilities is not None and self.gold is not None
            else None
        )

    def option_probabilities(self) -> tuple[float, ...] | None:
        if self.raw_scores is None or self.gold is None:
            return None
        peak = max(self.raw_scores)
        weights = [math.exp(value - peak) for value in self.raw_scores]
        total = sum(weights)
        return tuple(value / total for value in weights)

    def log_loss(self) -> float | None:
        probability = self.option_probability()
        return -math.log(probability) if probability is not None else None

    def brier_score(self) -> float | None:
        probabilities = self.option_probabilities()
        if probabilities is None or self.gold is None:
            return None
        return sum((value - (index == self.gold)) ** 2 for index, value in enumerate(probabilities))


@dataclass(frozen=True)
class Condition:
    name: str
    path: Path
    tasks: dict[str, dict[tuple[str, str], Example]]


class _BootstrapEngine:
    """Deterministic, bounded-memory bootstrap sampler with a stdlib fallback."""

    def __init__(self, seed: int, batch_elements: int = 8_000_000):
        self.batch_elements = batch_elements
        try:
            import numpy as np
        except ImportError:
            self.np = None
            self.rng = random.Random(seed)
        else:
            self.np = np
            self.rng = np.random.default_rng(seed)

    @property
    def vectorized(self) -> bool:
        return self.np is not None

    def means(self, values: Sequence[float], resamples: int) -> list[float]:
        n = len(values)
        if not n:
            raise ValueError("cannot bootstrap an empty sample")
        if self.np is None:
            return [
                sum(values[self.rng.randrange(n)] for _ in range(n)) / n
                for _ in range(resamples)
            ]
        array = self.np.asarray(values, dtype=self.np.float64)
        batch_size = max(1, min(resamples, self.batch_elements // n))
        draws = []
        for start in range(0, resamples, batch_size):
            size = min(batch_size, resamples - start)
            indices = self.rng.integers(0, n, size=(size, n))
            draws.extend(array[indices].mean(axis=1).tolist())
        return draws

    def fixed_task_macro(
        self, values: Sequence[Sequence[float]], resamples: int
    ) -> list[float]:
        if self.np is None:
            draws = []
            for _ in range(resamples):
                task_means = []
                for task_values in values:
                    n = len(task_values)
                    task_means.append(
                        sum(task_values[self.rng.randrange(n)] for _ in range(n)) / n
                    )
                draws.append(sum(task_means) / len(task_means))
            return draws
        largest_task = max(len(task) for task in values)
        batch_size = max(1, min(resamples, self.batch_elements // largest_task))
        draws = []
        for start in range(0, resamples, batch_size):
            size = min(batch_size, resamples - start)
            macro = self.np.zeros(size, dtype=self.np.float64)
            for task_values in values:
                array = self.np.asarray(task_values, dtype=self.np.float64)
                indices = self.rng.integers(0, len(array), size=(size, len(array)))
                macro += array[indices].mean(axis=1) / len(values)
            draws.extend(macro.tolist())
        return draws


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _stable_config(value), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _stable_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_config(item) for item in value]
    if isinstance(value, str):
        return re.sub(r" at 0x[0-9A-Fa-f]+(?=>)", "", value)
    return value


def _task_from_filename(path: Path) -> str:
    match = re.match(r"samples_(.+)_\d{4}-\d{2}-\d{2}T", path.name)
    if not match:
        raise PairingError(f"cannot determine task name from sample file: {path}")
    return match.group(1)


def _artifact_timestamp(path: Path) -> str | None:
    match = re.search(r"_(\d{4}-\d{2}-\d{2}T.+)\.(?:json|jsonl)$", path.name)
    return match.group(1) if match else None


def _result_metadata(
    root: Path, sample_files: Sequence[Path]
) -> tuple[dict[str, str], dict[str, Any]]:
    result_files = sorted(root.rglob("results_*.json"))
    sample_timestamps = {_artifact_timestamp(path) for path in sample_files}
    if None in sample_timestamps or len(sample_timestamps) != 1:
        raise PairingError(
            f"sample files below {root} do not share one unambiguous evaluation timestamp"
        )
    timestamp = next(iter(sample_timestamps))
    result_files = [path for path in result_files if _artifact_timestamp(path) == timestamp]
    if len(result_files) != 1:
        raise PairingError(
            f"expected exactly one results JSON matching sample timestamp {timestamp} "
            f"below {root}, found {len(result_files)}"
        )
    result = json.loads(result_files[0].read_text())
    configs = result.get("configs", {})
    if not isinstance(configs, dict):
        raise PairingError(f"results file has no task configs mapping: {result_files[0]}")
    versions = result.get("versions", {})
    hashes = {}
    for task, config in configs.items():
        comparable = dict(config)
        metadata = comparable.get("metadata", {})
        comparable["metadata"] = {
            key: metadata[key]
            for key in ("version", "config_source")
            if isinstance(metadata, dict) and key in metadata
        }
        comparable["metadata"].setdefault("version", versions.get(task))
        hashes[task] = _canonical_hash(comparable)
    return hashes, result


def _choice_values(sample: dict[str, Any]) -> tuple[tuple[float, ...], tuple[str, ...]] | None:
    filtered = sample.get("filtered_resps")
    arguments = sample.get("arguments")
    if not isinstance(filtered, list) or not isinstance(arguments, dict) or len(filtered) < 2:
        return None
    try:
        scores = tuple(float(item[0]) for item in filtered)
        choices = tuple(str(arguments[f"gen_args_{i}"]["arg_1"]) for i in range(len(scores)))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return scores, choices


def _gold_index(sample: dict[str, Any], choices: Sequence[str]) -> int:
    target = sample.get("target")
    doc = sample.get("doc", {})
    candidates = [target]
    if isinstance(doc, dict):
        candidates.extend(doc.get(key) for key in ("label", "answer", "answerKey", "gold"))
    for value in candidates:
        if isinstance(value, bool) and len(choices) == 2:
            labels = [choice.strip().lower() for choice in choices]
            wanted = "yes" if value else "no"
            if wanted in labels:
                return labels.index(wanted)
        if isinstance(value, int) and 0 <= value < len(choices):
            return value
        text = str(value).strip() if value is not None else ""
        if text.isdigit():
            number = int(text)
            if 0 <= number < len(choices):
                return number
            if 1 <= number <= len(choices):
                return number - 1
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if text.upper() in labels[: len(choices)]:
            return labels.index(text.upper())
        normalized = [choice.strip() for choice in choices]
        if text in normalized:
            return normalized.index(text)
    raise PairingError(
        "cannot identify gold option for "
        f"task={sample.get('task_name')} doc_id={sample.get('doc_id')}"
    )


def _parse_sample(task: str, config_hash: str, sample: dict[str, Any]) -> Example:
    embedded_task = sample.get("task_name")
    if embedded_task is not None and embedded_task != task:
        raise PairingError(f"sample task_name {embedded_task!r} disagrees with file task {task!r}")
    embedded_hash = sample.get("task_config_hash")
    if embedded_hash is not None:
        config_hash = str(embedded_hash)
    dataset_fingerprint = sample.get("dataset_fingerprint")
    if dataset_fingerprint is not None:
        config_hash = _canonical_hash(
            {"task_config_hash": config_hash, "dataset_fingerprint": dataset_fingerprint}
        )
    doc_id = sample.get("doc_id")
    doc_hash = sample.get("doc_hash")
    if doc_id is None or not doc_hash:
        raise PairingError(f"sample in task {task} lacks doc_id or doc_hash")
    choice_data = _choice_values(sample)
    metric = next(
        (name for name in ("acc_norm", "acc", "exact_match", "pass@1") if name in sample),
        "",
    )
    if not metric:
        raise PairingError(f"sample task={task} doc_id={doc_id} has no supported binary metric")
    if choice_data:
        scores, choices = choice_data
        gold = _gold_index(sample, choices)
        normalized = tuple(
            score / len(choice) for score, choice in zip(scores, choices, strict=True)
        )
    else:
        scores = normalized = None
        gold = None
    return Example(
        task=task,
        doc_id=str(doc_id),
        doc_hash=str(doc_hash),
        config_hash=config_hash,
        correct=float(sample[metric]),
        raw_scores=scores,
        normalized_scores=normalized,
        gold=gold,
        metric=metric,
    )


def load_condition(name: str, root: str | Path) -> Condition:
    """Load one lm-eval output directory and reject ambiguous artifacts."""
    root = Path(root).resolve()
    sample_files = sorted(root.rglob("samples_*.jsonl"))
    if not sample_files:
        raise PairingError(f"no samples_*.jsonl files below {root}; inference rerun required")
    config_hashes, _ = _result_metadata(root, sample_files)
    tasks: dict[str, dict[tuple[str, str], Example]] = {}
    for path in sample_files:
        task = _task_from_filename(path)
        if task not in config_hashes:
            raise PairingError(f"sample task {task!r} is absent from results configs")
        if task in tasks:
            raise PairingError(f"multiple sample files for task {task!r} below {root}")
        records: dict[tuple[str, str], Example] = {}
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            sample = json.loads(line)
            example = _parse_sample(task, config_hashes[task], sample)
            if example.identity in records:
                raise PairingError(f"duplicate identity {example.identity} in {path}:{line_number}")
            if any(item.doc_id == example.doc_id for item in records.values()):
                raise PairingError(
                    f"duplicate doc_id {example.doc_id} with differing hashes in {path}"
                )
            records[example.identity] = example
        tasks[task] = records
    return Condition(name=name, path=root, tasks=tasks)


def exact_mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value using the Binomial(n, 0.5) distribution."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap(
    deltas: Sequence[float], engine: _BootstrapEngine, resamples: int
) -> dict[str, float]:
    estimates = engine.means(deltas, resamples)
    estimates.sort()
    mean = sum(deltas) / len(deltas)
    boot_mean = sum(estimates) / resamples
    se = math.sqrt(sum((value - boot_mean) ** 2 for value in estimates) / max(1, resamples - 1))
    return {
        "estimate": mean,
        "bootstrap_se": se,
        "ci95_low": estimates[int(0.025 * resamples)],
        "ci95_high": estimates[min(resamples - 1, int(0.975 * resamples))],
    }


def _paired_task(
    task: str,
    a: dict[tuple[str, str], Example],
    b: dict[tuple[str, str], Example],
    engine: _BootstrapEngine,
    resamples: int,
) -> dict[str, Any]:
    if set(a) != set(b):
        missing_a = sorted(set(b) - set(a))[:5]
        missing_b = sorted(set(a) - set(b))[:5]
        raise PairingError(
            f"task {task} example sets differ (nA={len(a)}, nB={len(b)}, "
            f"missing_from_A={missing_a}, missing_from_B={missing_b})"
        )
    pairs = [(a[key], b[key]) for key in sorted(a)]
    for left, right in pairs:
        if left.config_hash != right.config_hash:
            raise PairingError(f"task {task} configuration/dataset fingerprint differs")
    correctness = [left.correct - right.correct for left, right in pairs]
    both_correct = sum(left.correct == 1 and right.correct == 1 for left, right in pairs)
    both_wrong = sum(left.correct == 0 and right.correct == 0 for left, right in pairs)
    only_a = sum(left.correct == 1 and right.correct == 0 for left, right in pairs)
    only_b = sum(left.correct == 0 and right.correct == 1 for left, right in pairs)
    output: dict[str, Any] = {
        "n": len(pairs),
        "matching": {"matched": len(pairs), "unmatched_A": 0, "unmatched_B": 0, "duplicates": 0},
        "accuracy_A": sum(left.correct for left, _ in pairs) / len(pairs),
        "accuracy_B": sum(right.correct for _, right in pairs) / len(pairs),
        "accuracy_difference": _bootstrap(correctness, engine, resamples),
        "discordance": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "only_A_correct": only_a,
            "only_B_correct": only_b,
            "mcnemar_exact_two_sided_p": exact_mcnemar(only_a, only_b),
        },
    }
    margin_types = ((False, "raw_margin_difference"), (True, "normalized_margin_difference"))
    for normalized, label in margin_types:
        values = []
        for left, right in pairs:
            lm, rm = left.margin(normalized), right.margin(normalized)
            if lm is None or rm is None:
                values = []
                break
            values.append(lm - rm)
        output[label] = _bootstrap(values, engine, resamples) if values else None
    probs = [(left.option_probability(), right.option_probability()) for left, right in pairs]
    if all(x is not None and y is not None for x, y in probs):
        output["option_set_gold_probability_difference"] = _bootstrap(
            [float(x) - float(y) for x, y in probs], engine, resamples
        )
        output["option_set_log_loss_difference"] = _bootstrap(
            [float(left.log_loss()) - float(right.log_loss()) for left, right in pairs],
            engine,
            resamples,
        )
        output["option_set_brier_difference"] = _bootstrap(
            [float(left.brier_score()) - float(right.brier_score()) for left, right in pairs],
            engine,
            resamples,
        )
    return output


def _macro_bootstrap(
    paired: dict[str, tuple[list[float], list[float]]],
    engine: _BootstrapEngine,
    resamples: int,
) -> dict[str, float]:
    point = sum(
        sum(a - b for a, b in zip(*values, strict=True)) / len(values[0])
        for values in paired.values()
    ) / len(paired)
    task_deltas = [
        [a - b for a, b in zip(a_values, b_values, strict=True)]
        for a_values, b_values in paired.values()
    ]
    draws = engine.fixed_task_macro(task_deltas, resamples)
    draws.sort()
    boot_mean = sum(draws) / resamples
    return {
        "estimate": point,
        "bootstrap_se": math.sqrt(sum((x - boot_mean) ** 2 for x in draws) / max(1, resamples - 1)),
        "ci95_low": draws[int(0.025 * resamples)],
        "ci95_high": draws[min(resamples - 1, int(0.975 * resamples))],
    }


def compare(
    a: Condition, b: Condition, seed: int = 12345, resamples: int = 10_000
) -> dict[str, Any]:
    """Compare two conditions with A-minus-B as the consistent direction."""
    if resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if set(a.tasks) != set(b.tasks):
        raise PairingError(f"task sets differ: A={sorted(a.tasks)}, B={sorted(b.tasks)}")
    engine = _BootstrapEngine(seed)
    task_results = {
        task: _paired_task(task, a.tasks[task], b.tasks[task], engine, resamples)
        for task in sorted(a.tasks)
    }
    aggregate = None
    if set(a.tasks) == set(CORE_TASKS):
        values = {}
        for task in CORE_TASKS:
            keys = sorted(a.tasks[task])
            values[task] = (
                [a.tasks[task][key].correct for key in keys],
                [b.tasks[task][key].correct for key in keys],
            )
        aggregate = _macro_bootstrap(values, engine, resamples)
    return {
        "schema_version": SCHEMA_VERSION,
        "subtraction": "A-minus-B",
        "condition_A": a.name,
        "condition_B": b.name,
        "seed": seed,
        "bootstrap_resamples": resamples,
        "bootstrap_backend": "numpy-vectorized" if engine.vectorized else "python-fallback",
        "tasks": task_results,
        "core_unweighted_macro_accuracy_difference": aggregate,
        "warnings": [
            "Intervals quantify evaluation-example uncertainty conditional on these "
            "checkpoints; they do not quantify pretraining-seed variance.",
            "Task and pairwise results are exploratory multiple comparisons; emphasize "
            "effect sizes and intervals.",
            "Margin aggregation is omitted because core tasks do not all use the same "
            "official raw-versus-length-normalized accuracy rule.",
            *(
                ["At least one task has fewer than 100 matched examples."]
                if any(x["n"] < 100 for x in task_results.values())
                else []
            ),
        ],
    }


def markdown_report(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "<!-- schema: paired-eval-markdown/v1 -->",
        "| Comparison (A−B) | Task | n | Accuracy Δ (95% CI) | "
        "Raw margin Δ (95% CI) | Discordant A/B | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        comparison = f"{result['condition_A']} − {result['condition_B']}"
        macro = result.get("core_unweighted_macro_accuracy_difference")
        if macro:
            lines.append(f"| {comparison} | **core macro** | — | {_fmt(macro)} | — | — | — |")
        for task, row in result["tasks"].items():
            discord = row["discordance"]
            margin = row["raw_margin_difference"]
            lines.append(
                f"| {comparison} | {task} | {row['n']} | {_fmt(row['accuracy_difference'])} | "
                f"{_fmt(margin) if margin else 'n/a'} | "
                f"{discord['only_A_correct']}/{discord['only_B_correct']} | "
                f"{discord['mcnemar_exact_two_sided_p']:.4g} |"
            )
    lines.extend(
        [
            "",
            "Intervals are paired percentile-bootstrap 95% intervals over evaluation "
            "examples, conditional on fixed trained checkpoints; they are not "
            "pretraining-seed uncertainty.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: dict[str, float]) -> str:
    return f"{value['estimate']:.4f} [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"


def _condition_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=OUTPUT_DIR")
    return tuple(value.split("=", 1))  # type: ignore[return-value]


def _local_argv(args: argparse.Namespace) -> list[str]:
    executable = "python" if args.slurm_environment else sys.executable
    command = [executable, "-m", "experiment_manager.paired_eval"]
    for name, path in args.condition:
        command.extend(["--condition", f"{name}={Path(path).resolve()}"])
    command.extend(
        [
            "--seed",
            str(args.seed),
            "--resamples",
            str(args.resamples),
            "--json",
            str(args.json.resolve()),
            "--markdown",
            str(args.markdown.resolve()),
        ]
    )
    return command


def slurm_command(args: argparse.Namespace) -> list[str]:
    """Build a CPU-only sbatch command that runs the local analysis CLI."""
    repository = Path(__file__).resolve().parents[1]
    log_path = (
        args.slurm_log.resolve()
        if args.slurm_log
        else args.markdown.resolve().with_suffix(".slurm-%j.out")
    )
    command = [
        "sbatch",
        "--job-name=paired-eval",
        f"--account={args.slurm_account}",
        f"--time={args.slurm_time}",
        f"--mem={args.slurm_mem}",
        "--ntasks=1",
        f"--cpus-per-task={args.slurm_cpus}",
        f"--chdir={repository}",
        f"--output={log_path}",
    ]
    if args.slurm_partition:
        command.append(f"--partition={args.slurm_partition}")
    local = _local_argv(args)
    if args.slurm_environment:
        local = [
            "srun",
            "--ntasks=1",
            "--mpi=pmix",
            f"--environment={args.slurm_environment}",
            *local,
        ]
    command.append(f"--wrap={shlex.join(local)}")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=_condition_arg, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--submit-slurm",
        action="store_true",
        help="submit this CPU-only analysis through sbatch instead of running locally",
    )
    parser.add_argument("--slurm-account", default="infra01")
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-time", default="00:30:00")
    parser.add_argument("--slurm-mem", default="4G")
    parser.add_argument("--slurm-cpus", type=int, default=1)
    parser.add_argument("--slurm-log", type=Path)
    parser.add_argument(
        "--slurm-environment",
        default="test-env",
        help="Slurm container environment providing NumPy; pass an empty value to disable",
    )
    args = parser.parse_args(argv)
    if len(args.condition) < 2:
        parser.error("provide at least two --condition NAME=OUTPUT_DIR arguments")
    if args.slurm_cpus < 1:
        parser.error("--slurm-cpus must be positive")
    if args.submit_slurm:
        job_id = submit(slurm_command(args))
        print(f"Submitted paired evaluation analysis as Slurm job {job_id}")
        return 0
    conditions = [load_condition(name, path) for name, path in args.condition]
    results = [
        compare(conditions[i], conditions[j], args.seed, args.resamples)
        for i in range(len(conditions))
        for j in range(i + 1, len(conditions))
    ]
    payload = {"schema_version": SCHEMA_VERSION, "comparisons": results}
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
