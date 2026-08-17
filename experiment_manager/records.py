"""Immutable render records and append-only submission metadata."""

from __future__ import annotations

import datetime as dt
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from experiment_manager.config import ResolvedExperiment, Stage


def _git_metadata(path: Path) -> dict[str, Any]:
    directory = path if path.is_dir() else path.parent

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(directory), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    root = run("rev-parse", "--show-toplevel")
    if root is None:
        return {"status": "not-git", "path": str(directory)}
    status = run("status", "--short") or ""
    return {
        "status": "dirty" if status else "clean",
        "root": root,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "remote": run("remote", "get-url", "origin"),
        "changes": status.splitlines(),
    }


def shell_command(experiment: ResolvedExperiment, stage: Stage) -> list[str]:
    """Build the exact sbatch argv for a stage."""
    exports = ["ALL"] + [f"{key}={value}" for key, value in sorted(stage.environment.items())]
    for value in exports:
        if "," in value or "\n" in value:
            raise ValueError(f"Slurm --export value cannot contain comma/newline: {value!r}")
    return [
        "sbatch",
        *experiment.sbatch_args,
        f"--chdir={experiment.submission_script.parent}",
        f"--job-name={experiment.experiment_name}--{stage.key}",
        f"--export={','.join(exports)}",
        str(experiment.submission_script),
    ]


def _shell_script(command: list[str]) -> str:
    import shlex

    return "#!/bin/bash\nset -euo pipefail\n\n" + shlex.join(command) + "\n"


def create_run_record(experiment: ResolvedExperiment) -> Path:
    """Create a new timestamped render containing the fully resolved configuration."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        experiment.artifacts_root
        / experiment.study
        / experiment.condition_name
        / f"{timestamp}-{experiment.config_hash[:10]}"
    )
    suffix = 1
    candidate = run_dir
    while candidate.exists():
        candidate = run_dir.with_name(f"{run_dir.name}-{suffix}")
        suffix += 1
    run_dir = candidate
    scripts_dir = run_dir / "scripts"
    source_dir = run_dir / "source"
    scripts_dir.mkdir(parents=True)
    source_dir.mkdir()

    # Freeze the launcher as executable source, not merely as metadata. Resume
    # operations therefore remain stable if a family launcher later exposes or
    # hides parameters, changes defaults, or is renamed.
    launcher_snapshot = source_dir / experiment.submission_script.name
    shutil.copy2(experiment.submission_script, launcher_snapshot)
    frozen_resolved = copy.deepcopy(experiment.resolved)
    frozen_resolved["execution"]["original_submission_script"] = str(
        experiment.submission_script
    )
    frozen_resolved["execution"]["submission_script"] = str(launcher_snapshot)
    frozen_resolved["execution"]["working_directory"] = str(
        experiment.submission_script.parent
    )
    frozen_resolved["execution"]["run_record"] = str(run_dir)
    for stage in frozen_resolved["stages"]:
        stage_artifacts = run_dir / "stages" / stage["key"]
        (stage_artifacts / "slurm").mkdir(parents=True)
        stage["environment"]["EXPERIMENT_ARTIFACTS_DIR"] = str(stage_artifacts)

    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(frozen_resolved, sort_keys=False, width=120)
    )
    for stage in frozen_resolved["stages"]:
        path = scripts_dir / f"submit-{stage['key']}.sh"
        path.write_text(_shell_script(command_from_record(frozen_resolved, stage)))
        path.chmod(0o755)

    manager_root = Path(__file__).resolve().parents[1]
    megatron_path = Path(
        os.path.expandvars(experiment.main.environment.get("MEGATRON_LM_DIR", ""))
    )
    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "user": os.environ.get("USER"),
        "host": os.uname().nodename,
        "config_hash": experiment.config_hash,
        "source_config": str(experiment.config_path),
        "source_recipe": str(experiment.recipe_path),
        "git": {
            "manager": _git_metadata(manager_root),
            "submission_script": _git_metadata(experiment.submission_script),
            "megatron": _git_metadata(megatron_path) if str(megatron_path) else None,
        },
    }
    (run_dir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, width=120)
    )
    (run_dir / "jobs.yaml").write_text("schema_version: 1\nsubmissions: []\n")

    latest = run_dir.parent / "latest_run.txt"
    latest.write_text(str(run_dir) + "\n")
    return run_dir


def load_run_record(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    run_dir = Path(run_dir).expanduser().resolve()
    resolved_path = run_dir / "resolved.yaml"
    if not resolved_path.is_file():
        raise ValueError(f"not an experiment run directory: {run_dir}")
    resolved = yaml.safe_load(resolved_path.read_text())
    if not isinstance(resolved, dict) or "stages" not in resolved:
        raise ValueError(f"invalid resolved run record: {resolved_path}")
    return run_dir, resolved


def append_submission(
    run_dir: Path,
    *,
    stage_key: str,
    job_id: str,
    command: list[str],
    action: str,
) -> None:
    """Append one Slurm submission event to jobs.yaml."""
    path = run_dir / "jobs.yaml"
    data = yaml.safe_load(path.read_text())
    data["submissions"].append(
        {
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "action": action,
            "stage": stage_key,
            "job_id": job_id,
            "command": command,
        }
    )
    path.write_text(yaml.safe_dump(data, sort_keys=False, width=120))


def resolved_stage(resolved: dict[str, Any], stage_key: str) -> dict[str, Any]:
    matches = [stage for stage in resolved["stages"] if stage["key"] == stage_key]
    if not matches:
        available = ", ".join(stage["key"] for stage in resolved["stages"])
        raise ValueError(f"unknown stage {stage_key!r}; available: {available}")
    return matches[0]


def command_from_record(resolved: dict[str, Any], stage: dict[str, Any]) -> list[str]:
    exports = ["ALL"] + [
        f"{key}={value}" for key, value in sorted(stage["environment"].items())
    ]
    submission_script = Path(resolved["execution"]["submission_script"])
    working_directory = resolved["execution"].get(
        "working_directory", str(submission_script.parent)
    )
    stage_artifacts = stage["environment"].get("EXPERIMENT_ARTIFACTS_DIR")
    logging_args = []
    if stage_artifacts:
        logging_args = [
            f"--output={stage_artifacts}/slurm/%x-%j.out",
            f"--error={stage_artifacts}/slurm/%x-%j.err",
        ]
    return [
        "sbatch",
        *resolved["execution"].get("sbatch_args", []),
        *logging_args,
        f"--chdir={working_directory}",
        f"--job-name={resolved['execution']['experiment_name']}--{stage['key']}",
        f"--export={','.join(exports)}",
        str(submission_script),
    ]


def dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
