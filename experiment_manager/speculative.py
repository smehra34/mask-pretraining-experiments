"""Immutable planning records for native Megatron speculative analysis."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiment_manager.evaluation import _checkpoint_dir, _latest_step
from experiment_manager.records import load_run_record, resolved_stage


@dataclass(frozen=True)
class ResolvedSpeculativeAnalysis:
    """Fully frozen checkpoint, workload, algorithm matrix, and resource request."""

    run_dir: Path
    stage_key: str
    checkpoint_dir: Path
    checkpoint_step: int
    suite_name: str
    config: dict[str, Any]
    config_hash: str
    output_root: Path


def _git_metadata(path: str | Path) -> dict[str, object]:
    """Return reproducibility metadata without requiring a clean checkout."""
    root = Path(path)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True
    )
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"], capture_output=True
    )
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
    )
    state = hashlib.sha256()
    if diff.returncode == 0:
        state.update(diff.stdout)
    if untracked.returncode == 0:
        for raw_name in sorted(name for name in untracked.stdout.split(b"\0") if name):
            state.update(raw_name)
            candidate = root / raw_name.decode()
            if candidate.is_file():
                state.update(candidate.read_bytes())
    return {
        "path": str(root),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "working_tree_hash": state.hexdigest(),
    }


def resolve_speculative(
    run: str | Path,
    *,
    stage_key: str,
    checkpoint_step: int | None,
    suite_name: str,
    config_path: str | Path,
    require_checkpoint: bool,
) -> ResolvedSpeculativeAnalysis:
    """Resolve without writing or mutating a completed training record."""
    run_dir, training = load_run_record(run)
    stage = resolved_stage(training, stage_key)
    source = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("speculative configuration must use schema_version 1")
    suites = raw.get("suites", {})
    if suite_name not in suites:
        raise ValueError(f"unknown speculative suite {suite_name!r}")
    checkpoint_dir = _checkpoint_dir(training, stage)
    step = _latest_step(checkpoint_dir) if checkpoint_step is None else checkpoint_step
    checkpoint = checkpoint_dir / f"iter_{step:07d}"
    if require_checkpoint and not checkpoint.is_dir():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    environment = stage["environment"]
    objective = "mtp" if int(environment.get("MTP_NUM_LAYERS", "0")) else (
        "masked" if float(environment.get("INPUT_MASK_RATIO", "0")) else "ntp"
    )
    suite = dict(suites[suite_name])
    drafters_by_objective = suite.pop("drafters_by_objective", None)
    if drafters_by_objective is not None:
        if objective not in drafters_by_objective:
            raise ValueError(f"suite {suite_name!r} has no drafters for objective {objective!r}")
        suite["drafters"] = list(drafters_by_objective[objective])
    if "ar" not in suite.get("drafters", []):
        raise ValueError("every suite must include the checkpoint's own AR baseline")
    depths = suite.get("draft_depths", [1, 2, 3, 4, 6, 8])
    invalid_depths = any(value not in (1, 2, 3, 4, 6, 8) for value in depths)
    if sorted(set(depths)) != sorted(depths) or invalid_depths:
        raise ValueError("draft_depths must be unique values from 1,2,3,4,6,8")
    profiles = suite.get("sampling_profiles", {})
    if "greedy" not in profiles or "temperature_1" not in profiles:
        raise ValueError("suite must include greedy and temperature_1 sampling profiles")
    workload = Path(str(suite["workload"])).expanduser().resolve()
    fingerprint = hashlib.sha256(workload.read_bytes()).hexdigest() if workload.is_file() else None
    backend = dict(raw["backend"])
    backend.update(suite.pop("backend_overrides", {}))
    training_hash = str(training["config_hash"])
    experiment_name = str(training["execution"]["experiment_name"])
    evaluation_name = f"{experiment_name}--{training_hash[:10]}--{stage_key}--{suite_name}"
    config = {
        "schema_version": "mask-exp.speculative/v1",
        "checkpoint": {"load": str(checkpoint_dir), "iteration": step, "path": str(checkpoint)},
        "training": {
            "run_record": str(run_dir),
            "experiment_name": experiment_name,
            "study": training["condition"]["study"],
            "condition": training["condition"]["name"],
            "training_config_hash": training_hash,
            "stage": stage_key,
            "objective": objective,
            "seed": environment.get("SEED"),
            "mask_token": environment.get("INPUT_MASK_TOKEN"),
            # The tokenizer-resolved ID is added to the aggregate by the native
            # runner; training records currently freeze the token string.
            "mask_token_id": environment.get("INPUT_MASK_TOKEN_ID"),
            "mtp_num_layers": environment.get("MTP_NUM_LAYERS"),
            "tokenizer_model": environment.get("TOKENIZER_MODEL"),
            "sequence_length": environment.get("SEQ_LEN"),
            "topology": {key: environment.get(key) for key in ("TP", "PP", "CP", "EP")},
        },
        "suite": {
            "name": suite_name,
            **suite,
            "workload": str(workload),
            "dataset_fingerprint": fingerprint,
        },
        "backend": backend,
        "provenance": {
            "megatron": _git_metadata(backend["megatron_path"]),
            "analysis_manager": _git_metadata(Path(__file__).resolve().parents[1]),
            "lm_eval_source": _git_metadata(backend["lm_eval_source"]),
        },
        "evaluation_name": evaluation_name,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return ResolvedSpeculativeAnalysis(
        run_dir,
        stage_key,
        checkpoint_dir,
        step,
        suite_name,
        config,
        digest,
        Path(backend["output_root"]),
    )


def render_script(analysis: ResolvedSpeculativeAnalysis, config_file: str | Path) -> str:
    """Render but never submit a distributed Slurm analysis script."""
    backend = analysis.config["backend"]
    nodes = int(backend["nodes"])
    gpus = int(backend["gpus_per_node"])
    megatron = str(backend["megatron_path"])
    log_dir = Path(config_file).parent / "slurm"
    return f"""#!/bin/bash
#SBATCH --job-name=spec-{analysis.suite_name}
#SBATCH --account={backend['account']}
#SBATCH --partition={backend.get('partition', '')}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={backend['run_time']}
#SBATCH --requeue
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
set -euo pipefail
cd {megatron}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${{MASTER_PORT:-29500}}
srun --ntasks=${{SLURM_NNODES}} --ntasks-per-node=1 --mpi=pmix \
  --network=disable_rdzv_get --environment={backend.get('container_edf', 'test-env')} bash -c '
  NODE_RANK=${{SLURM_NODEID}}
  python -m torch.distributed.run \
    --nnodes='"${{SLURM_NNODES}}"' \
    --nproc-per-node={gpus} \
    --node-rank=${{NODE_RANK}} \
    --master-addr='"${{MASTER_ADDR}}"' \
    --master-port='"${{MASTER_PORT}}"' \
    tools/run_speculative_analysis.py \
      --config '"{config_file}"'
'
"""


def create_speculative_record(analysis: ResolvedSpeculativeAnalysis) -> Path:
    """Create a new immutable analysis record; never overwrite an existing record."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = (
        analysis.run_dir
        / "speculative"
        / analysis.stage_key
        / f"step_{analysis.checkpoint_step:07d}"
        / analysis.suite_name
    )
    record = base / f"{stamp}-{analysis.config_hash[:10]}"
    suffix = 1
    while record.exists():
        record = base / f"{stamp}-{analysis.config_hash[:10]}-{suffix}"
        suffix += 1
    (record / "slurm").mkdir(parents=True)
    created_at = dt.datetime.now(dt.timezone.utc)
    result_stamp = created_at.strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    output_dir = (
        analysis.output_root
        / analysis.config["evaluation_name"]
        / f"step_{analysis.checkpoint_step}"
        / "speculative"
    )
    frozen = {
        **analysis.config,
        "config_hash": analysis.config_hash,
        "created_at": created_at.isoformat(),
        "analysis_record": str(record),
        "output": {
            "directory": str(output_dir),
            "aggregate": str(output_dir / f"results_{result_stamp}.json"),
            "samples": str(output_dir / f"samples_speculative_{result_stamp}.jsonl"),
            "summary": str(output_dir / f"summary_{result_stamp}.md"),
            "progress": str(output_dir / f"progress_speculative_{result_stamp}.json"),
        },
    }
    config_file = record / "resolved.yaml"
    config_file.write_text(yaml.safe_dump(frozen, sort_keys=False, width=120))
    script = record / "submit.sh"
    script.write_text(render_script(analysis, config_file))
    script.chmod(0o755)
    (record / "jobs.yaml").write_text("schema_version: 1\nsubmissions: []\n")
    return record


def append_speculative_submission(
    record: Path,
    job_id: str,
    *,
    command: list[str] | None = None,
    action: str = "submit",
) -> None:
    """Attach scheduler identity to the immutable analysis record."""
    path = record / "jobs.yaml"
    value = yaml.safe_load(path.read_text())
    value["submissions"].append(
        {
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_id": str(job_id),
            "action": action,
            "command": command or ["sbatch", str(record / "submit.sh")],
        }
    )
    path.write_text(yaml.safe_dump(value, sort_keys=False, width=120))


def load_speculative_record(record: str | Path) -> tuple[Path, dict[str, Any]]:
    """Validate and load one immutable speculative-analysis record."""
    path = Path(record).expanduser().resolve()
    config_path = path / "resolved.yaml"
    script_path = path / "submit.sh"
    jobs_path = path / "jobs.yaml"
    for required in (config_path, script_path, jobs_path):
        if not required.is_file():
            raise ValueError(f"speculative analysis record is missing {required.name}: {path}")
    frozen = yaml.safe_load(config_path.read_text())
    if not isinstance(frozen, dict) or frozen.get("schema_version") != "mask-exp.speculative/v1":
        raise ValueError(f"unsupported speculative analysis record: {path}")
    if Path(frozen.get("analysis_record", "")).resolve() != path:
        raise ValueError("resolved speculative record does not point back to its directory")
    return path, frozen
