"""Resolve and record lm-eval jobs for immutable training runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiment_manager.records import load_run_record, resolved_stage


@dataclass(frozen=True)
class EvaluationSuite:
    name: str
    description: str
    tasks: tuple[str, ...]
    run_time: str
    limit: int | float | None
    num_fewshot: int | None
    gen_kwargs: dict[str, object]
    log_samples: bool
    write_out: bool
    unsafe_code: bool
    evaluator: str
    paloma_data_root: str
    paloma_split: str
    paloma_sources: tuple[str, ...]
    paloma_limit_tokens: int | None


@dataclass(frozen=True)
class ResolvedEvaluation:
    run_dir: Path
    stage_key: str
    checkpoint_dir: Path
    checkpoint_step: int
    suite: EvaluationSuite
    backend: dict[str, Any]
    training: dict[str, Any]
    metadata: dict[str, object]
    dataset_prefetch: dict[str, list[str]]
    evaluation_name: str
    wandb_id: str
    output_root: Path
    config_hash: str


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"evaluation configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"evaluation configuration must use schema_version 1: {path}")
    return value


def _expand(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in evaluation setting: {value!r}")
    return expanded


def load_evaluation_config(path: str | Path) -> tuple[dict[str, Any], dict[str, EvaluationSuite], dict[str, list[str]]]:
    """Load backend defaults, named suites, and dataset prefetch declarations."""
    config = _load_yaml(Path(path).expanduser().resolve())
    backend = config.get("backend")
    suites_value = config.get("suites")
    prefetch = config.get("dataset_prefetch", {})
    if not isinstance(backend, dict) or not isinstance(suites_value, dict):
        raise ValueError("evaluation backend and suites must be mappings")
    if not isinstance(prefetch, dict):
        raise ValueError("dataset_prefetch must be a mapping")

    resolved_backend = {
        key: _expand(value) if isinstance(value, str) else value
        for key, value in backend.items()
    }
    lm_eval_install = Path(
        str(
            resolved_backend.get("lm_eval_source")
            or resolved_backend.get("lm_eval_install", "")
        )
    )
    lm_eval_commit = str(resolved_backend.get("lm_eval_commit", ""))
    if lm_eval_install.is_absolute() and lm_eval_commit:
        marker = lm_eval_install / ".lm_eval_commit"
        if marker.is_file():
            actual_commit = marker.read_text().strip()
        else:
            result = subprocess.run(
                ["git", "-C", str(lm_eval_install), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            actual_commit = result.stdout.strip() if result.returncode == 0 else ""
        if actual_commit != lm_eval_commit:
            raise ValueError(
                f"lm-eval checkout {lm_eval_install} is at {actual_commit or 'unknown'}, "
                f"expected pinned commit {lm_eval_commit}"
            )
        expected_patch = str(resolved_backend.get("lm_eval_patch", ""))
        if expected_patch:
            patch_marker = lm_eval_install / ".lm_eval_patch"
            actual_patch = patch_marker.read_text().strip() if patch_marker.is_file() else ""
            if actual_patch != expected_patch:
                raise ValueError(
                    f"lm-eval source {lm_eval_install} has patch marker "
                    f"{actual_patch or 'missing'}, expected {expected_patch}"
                )
    suites: dict[str, EvaluationSuite] = {}
    for name, raw in suites_value.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError("each evaluation suite must be a named mapping")
        evaluator = str(raw.get("evaluator", "lm_eval"))
        tasks = raw.get("tasks", [])
        if (evaluator == "lm_eval" and (not isinstance(tasks, list) or not tasks)) or not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"evaluation suite {name!r} must contain a non-empty task list")
        limit = raw.get("limit")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0):
            raise ValueError(f"evaluation suite {name!r} limit must be positive")
        suites[name] = EvaluationSuite(
            name=name,
            description=str(raw.get("description", "")),
            tasks=tuple(tasks),
            run_time=str(raw.get("run_time", "03:00:00")),
            limit=limit,
            num_fewshot=raw.get("num_fewshot"),
            gen_kwargs=dict(raw.get("gen_kwargs", {})),
            log_samples=bool(raw.get("log_samples", False)),
            write_out=bool(raw.get("write_out", False)),
            unsafe_code=bool(raw.get("unsafe_code", False)),
            evaluator=evaluator,
            paloma_data_root=_expand(str(raw.get("paloma_data_root", ""))) if raw.get("paloma_data_root") else "",
            paloma_split=str(raw.get("paloma_split", "test")),
            paloma_sources=tuple(str(x) for x in raw.get("paloma_sources", [])),
            paloma_limit_tokens=raw.get("paloma_limit_tokens"),
        )

    resolved_prefetch: dict[str, list[str]] = {}
    for task, args in prefetch.items():
        if not isinstance(task, str) or not isinstance(args, list) or not all(
            isinstance(value, str) for value in args
        ):
            raise ValueError("dataset_prefetch values must be lists of strings")
        resolved_prefetch[task] = args
    return resolved_backend, suites, resolved_prefetch


def _checkpoint_dir(resolved: dict[str, Any], stage: dict[str, Any]) -> Path:
    root = Path(resolved["execution"]["checkpoint_root"])
    if stage["mode"] in {"main", "extension"}:
        return root / "main"
    source = int(stage["source_iteration"])
    return root / "cooldowns" / f"from_iter_{source:07d}" / "checkpoints"


def _latest_step(checkpoint_dir: Path) -> int:
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError(f"checkpoint tracker does not exist: {tracker}")
    value = tracker.read_text().strip()
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"invalid checkpoint iteration in {tracker}: {value!r}")
    return int(value)


def resolve_evaluation(
    run: str | Path,
    *,
    stage_key: str,
    checkpoint_step: int | None,
    suite_name: str,
    config_path: str | Path,
    allow_unsafe_code: bool = False,
    require_checkpoint: bool = True,
) -> ResolvedEvaluation:
    """Resolve one suite against one checkpoint in an immutable training run."""
    run_dir, resolved = load_run_record(run)
    stage = resolved_stage(resolved, stage_key)
    backend, suites, all_prefetch = load_evaluation_config(config_path)
    if suite_name not in suites:
        raise ValueError(
            f"unknown evaluation suite {suite_name!r}; available: {', '.join(sorted(suites))}"
        )
    suite = suites[suite_name]
    if suite.unsafe_code and not allow_unsafe_code:
        raise ValueError(
            f"suite {suite.name!r} executes generated code; pass --allow-unsafe-code "
            "only after accepting the isolation requirements"
        )
    if suite.evaluator not in {"lm_eval", "paloma"}:
        raise ValueError(f"unsupported evaluation evaluator {suite.evaluator!r}")
    if suite.evaluator == "paloma" and not suite.paloma_data_root:
        raise ValueError("Paloma suite must define paloma_data_root")

    checkpoint_dir = _checkpoint_dir(resolved, stage)
    step = _latest_step(checkpoint_dir) if checkpoint_step is None else checkpoint_step
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    checkpoint_path = checkpoint_dir / f"iter_{step:07d}"
    if require_checkpoint and not checkpoint_path.is_dir():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")

    training = {
        "experiment_name": resolved["execution"]["experiment_name"],
        "study": resolved["condition"]["study"],
        "condition": resolved["condition"]["name"],
        "recipe": resolved["recipe"]["name"],
        "stage": stage_key,
        "checkpoint_step": step,
        "tokens_per_iteration": resolved["derived"]["tokens_per_iteration"],
        "tokens_seen": step * resolved["derived"]["tokens_per_iteration"],
        "seed": stage["environment"].get("SEED"),
        "hidden_size": stage["environment"].get("HIDDEN_SIZE"),
        "objective": "mtp"
        if int(stage["environment"].get("MTP_NUM_LAYERS", "0")) > 0
        else (
            "masked"
            if float(stage["environment"].get("INPUT_MASK_RATIO", "0")) > 0
            else "ntp"
        ),
    }
    metadata_path = run_dir / "metadata.yaml"
    run_metadata = yaml.safe_load(metadata_path.read_text()) if metadata_path.is_file() else {}
    megatron_commit = (
        run_metadata.get("git", {}).get("megatron", {}).get("commit")
        if isinstance(run_metadata, dict)
        else None
    )
    metadata: dict[str, object] = {
        **training,
        "training_config_hash": resolved.get("config_hash"),
        "megatron_commit": megatron_commit,
        "lm_eval_install": backend.get("lm_eval_install"),
        "lm_eval_source": backend.get("lm_eval_source"),
        "lm_eval_commit": backend.get("lm_eval_commit"),
        "lm_eval_patch": backend.get("lm_eval_patch"),
        "suite": suite.name,
        "tasks": list(suite.tasks),
    }
    training_hash = str(resolved.get("config_hash", "unknown"))[:10]
    identity = f"{run_dir.name}__{training['experiment_name']}__{stage_key}__{suite.name}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    evaluation_name = (
        f"{training['experiment_name']}--{training_hash}--{stage_key}--{suite.name}"
    )
    wandb_id = f"eval-{digest}"
    output_root = Path(str(backend["output_root"]))
    prefetch = {task: all_prefetch[task] for task in suite.tasks if task in all_prefetch}

    canonical = json.dumps(
        {
            "backend": backend,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": step,
            "metadata": metadata,
            "suite": suite.__dict__,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ResolvedEvaluation(
        run_dir=run_dir,
        stage_key=stage_key,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=step,
        suite=suite,
        backend=backend,
        training=training,
        metadata=metadata,
        dataset_prefetch=prefetch,
        evaluation_name=evaluation_name,
        wandb_id=wandb_id,
        output_root=output_root,
        config_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _spellbook_module(spellbook_path: Path):
    path = str(spellbook_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("evals.megatron_eval")


def spellbook_config(evaluation: ResolvedEvaluation, *, log_dir: Path):
    """Translate a resolved manager evaluation into Spellbook's public config."""
    module = _spellbook_module(Path(str(evaluation.backend["spellbook_path"])))
    backend = evaluation.backend
    runtime_deps = str(backend.get("megatron_runtime_deps", ""))
    env_vars = {
        "LM_HARNESS_CACHE_PATH": str(backend["request_cache"]),
        "WANDB_RUN_GROUP": str(evaluation.training["study"]),
        "WANDB_NAME": evaluation.evaluation_name,
        "WANDB_JOB_TYPE": "evaluation",
        "WANDB_TAGS": ",".join(
            [
                f"study:{evaluation.training['study']}",
                f"condition:{evaluation.training['condition']}",
                f"stage:{evaluation.stage_key}",
                f"suite:{evaluation.suite.name}",
                "job_type:evaluation",
            ]
        ),
    }
    # Paloma is implemented as a Spellbook module (``evals.paloma_eval``),
    # so the Spellbook checkout itself must be importable inside the job.
    python_paths = [str(backend["spellbook_path"])]
    if runtime_deps:
        python_paths.append(str(runtime_deps))
    lm_eval_source = backend.get("lm_eval_source")
    if lm_eval_source:
        python_paths.append(str(lm_eval_source))
    python_paths.append("/opt/megatron")
    env_vars["PYTHONPATH"] = ":".join(python_paths)
    if evaluation.suite.unsafe_code:
        env_vars["HF_ALLOW_CODE_EVAL"] = "1"

    training_record = yaml.safe_load((evaluation.run_dir / "resolved.yaml").read_text())
    stage = resolved_stage(training_record, evaluation.stage_key)
    megatron_commit = evaluation.metadata.get("megatron_commit") or ""
    return module.MegatronEvalConfig(
        model_name=evaluation.evaluation_name,
        checkpoint_dir=str(evaluation.checkpoint_dir.parent),
        checkpoint_path=str(evaluation.checkpoint_dir),
        tokenizer_model=stage["environment"]["TOKENIZER_MODEL"],
        megatron_path=str(backend["megatron_path"]),
        megatron_commit=str(megatron_commit),
        tasks=list(evaluation.suite.tasks),
        evaluator=evaluation.suite.evaluator,
        paloma_data_root=evaluation.suite.paloma_data_root,
        paloma_split=evaluation.suite.paloma_split,
        paloma_sources=list(evaluation.suite.paloma_sources),
        paloma_limit_tokens=evaluation.suite.paloma_limit_tokens,
        batch_size=int(backend["batch_size"]),
        cache_requests="true",
        limit=evaluation.suite.limit,
        num_fewshot=evaluation.suite.num_fewshot,
        gen_kwargs=evaluation.suite.gen_kwargs,
        confirm_run_unsafe_code=evaluation.suite.unsafe_code,
        devices=int(backend["devices"]),
        tp=int(backend["tp"]),
        ep=int(backend["ep"]),
        seq_length=int(stage["environment"]["SEQ_LEN"]),
        micro_batch_size=int(backend["micro_batch_size"]),
        metadata=evaluation.metadata,
        output_dir=str(evaluation.output_root),
        log_samples=evaluation.suite.log_samples,
        write_out=evaluation.suite.write_out,
        account=str(backend["account"]),
        partition=str(backend.get("partition", "")),
        reservation=str(backend.get("reservation", "")),
        nodes=int(backend["nodes"]),
        gpus_per_node=int(backend["gpus_per_node"]),
        launch_mode=str(backend["launch_mode"]),
        srun_extra_args=str(backend.get("srun_extra_args", "")),
        run_time=evaluation.suite.run_time,
        log_dir=str(log_dir),
        container_edf=str(backend.get("container_edf", "")),
        container_mounts=str(backend.get("container_mounts", "")),
        wandb_project=stage["environment"].get("WANDB_PROJECT", "mask_pretraining"),
        wandb_id=evaluation.wandb_id,
        hf_home=str(backend.get("hf_home", "")),
        lm_eval_install=str(backend.get("lm_eval_install", "")),
        lm_eval_install_with_python=True,
        lm_eval_install_args=str(
            backend.get("lm_eval_install_args", "--no-build-isolation")
        ),
        dataset_prefetch=evaluation.dataset_prefetch,
        env_vars=env_vars,
    )


def render_spellbook_script(evaluation: ResolvedEvaluation, *, log_dir: Path) -> str:
    module = _spellbook_module(Path(str(evaluation.backend["spellbook_path"])))
    return module.render(
        spellbook_config(evaluation, log_dir=log_dir), evaluation.checkpoint_step
    )


def create_evaluation_record(evaluation: ResolvedEvaluation) -> Path:
    """Create an immutable evaluation record and frozen Spellbook sbatch script."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = (
        evaluation.run_dir
        / "evaluations"
        / evaluation.stage_key
        / f"step_{evaluation.checkpoint_step:07d}"
        / evaluation.suite.name
    )
    record_dir = base / f"{timestamp}-{evaluation.config_hash[:10]}"
    suffix = 1
    while record_dir.exists():
        record_dir = base / f"{timestamp}-{evaluation.config_hash[:10]}-{suffix}"
        suffix += 1
    (record_dir / "slurm").mkdir(parents=True)
    script = render_spellbook_script(evaluation, log_dir=record_dir / "slurm")
    script_path = record_dir / "submit.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    resolved = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_hash": evaluation.config_hash,
        "training_run": str(evaluation.run_dir),
        "stage": evaluation.stage_key,
        "checkpoint_dir": str(evaluation.checkpoint_dir),
        "checkpoint_step": evaluation.checkpoint_step,
        "suite": evaluation.suite.__dict__,
        "backend": evaluation.backend,
        "metadata": evaluation.metadata,
        "dataset_prefetch": evaluation.dataset_prefetch,
        "evaluation_name": evaluation.evaluation_name,
        "wandb_id": evaluation.wandb_id,
        "wandb_project": spellbook_config(evaluation, log_dir=record_dir / "slurm").wandb_project,
        "output_root": str(evaluation.output_root),
        "script": str(script_path),
    }
    (record_dir / "resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, width=120)
    )
    (record_dir / "jobs.yaml").write_text("schema_version: 1\nsubmissions: []\n")
    (base / "latest_record.txt").write_text(str(record_dir) + "\n")
    return record_dir


def append_evaluation_submission(record_dir: Path, job_id: str) -> None:
    path = record_dir / "jobs.yaml"
    value = yaml.safe_load(path.read_text())
    value["submissions"].append(
        {
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_id": job_id,
            "command": ["sbatch", str(record_dir / "submit.sh")],
        }
    )
    path.write_text(yaml.safe_dump(value, sort_keys=False, width=120))
