"""Load, merge, validate, and resolve experiment configurations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a recipe or condition is invalid."""


@dataclass(frozen=True)
class Stage:
    """One executable stage in a resolved experiment."""

    key: str
    mode: str
    source_iteration: int | None
    environment: dict[str, str]


@dataclass(frozen=True)
class ResolvedExperiment:
    """Fully materialized condition and all of its main/cooldown stages."""

    config_path: Path
    recipe_path: Path
    recipe_name: str
    condition_name: str
    description: str
    study: str
    experiment_name: str
    submission_script: Path
    project_name: str
    checkpoint_root: Path
    artifacts_root: Path
    sbatch_args: tuple[str, ...]
    stages: tuple[Stage, ...]
    resolved: dict[str, Any]
    config_hash: str

    @property
    def main(self) -> Stage:
        return self.stages[0]

    @property
    def cooldowns(self) -> tuple[Stage, ...]:
        return self.stages[1:]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"configuration must contain a YAML mapping: {path}")
    return data


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key!r} must be a mapping")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer, got {value!r}")
    return value


def _stringify_env_value(value: Any, label: str) -> str:
    if value is None or isinstance(value, (dict, list)):
        raise ConfigError(f"{label} must be a scalar environment value, got {value!r}")
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    rendered = os.path.expandvars(os.path.expanduser(rendered))
    if "$" in rendered:
        raise ConfigError(f"{label} contains an unresolved environment variable: {rendered!r}")
    return rendered


def _expand_path(value: str, base: Path) -> Path:
    rendered = os.path.expandvars(os.path.expanduser(value))
    if "$" in rendered:
        raise ConfigError(f"path contains an unresolved environment variable: {rendered!r}")
    expanded = Path(rendered)
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def _render_template(value: str, fields: dict[str, str], label: str) -> str:
    value = os.path.expandvars(os.path.expanduser(value))
    if "$" in value:
        raise ConfigError(f"{label} contains an unresolved environment variable: {value!r}")
    try:
        return value.format_map(fields)
    except KeyError as exc:
        raise ConfigError(f"unknown placeholder {exc.args[0]!r} in {label}") from exc


def _validate_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ConfigError(
            f"{label} must contain only letters, digits, '.', '_' and '-', got {value!r}"
        )
    return value


def _validate_masking(env: dict[str, str]) -> None:
    try:
        ratio = float(env.get("INPUT_MASK_RATIO", "0"))
    except ValueError as exc:
        raise ConfigError("INPUT_MASK_RATIO must be numeric") from exc
    if not 0.0 <= ratio <= 1.0:
        raise ConfigError(f"INPUT_MASK_RATIO must be in [0, 1], got {ratio}")
    strategy = env.get("INPUT_MASK_STRATEGY", "random")
    if strategy not in {"random", "span"}:
        raise ConfigError("INPUT_MASK_STRATEGY must be 'random' or 'span'")
    span_length = int(env.get("INPUT_MASK_SPAN_LENGTH", "1"))
    if span_length <= 0:
        raise ConfigError("INPUT_MASK_SPAN_LENGTH must be positive")
    if ratio > 0.0 and not env.get("INPUT_MASK_TOKEN"):
        raise ConfigError("INPUT_MASK_TOKEN is required when input masking is enabled")


def load_experiment(config_path: str | Path) -> ResolvedExperiment:
    """Resolve a condition YAML against its referenced base recipe."""
    config_path = Path(config_path).expanduser().resolve()
    condition = _load_yaml(config_path)
    if condition.get("schema_version") != 1:
        raise ConfigError("condition schema_version must be 1")

    recipe_ref = condition.get("recipe")
    if not isinstance(recipe_ref, str) or not recipe_ref:
        raise ConfigError("condition must specify a recipe path")
    recipe_path = _expand_path(recipe_ref, config_path.parent)
    recipe = _load_yaml(recipe_path)
    if recipe.get("schema_version") != 1:
        raise ConfigError("recipe schema_version must be 1")

    recipe_name = _validate_name(recipe.get("name"), "recipe name")
    condition_name = _validate_name(condition.get("name"), "condition name")
    study = _validate_name(condition.get("study", recipe_name), "study name")
    description = str(condition.get("description", ""))

    execution = _require_mapping(recipe, "execution")
    stages_config = _require_mapping(recipe, "stages")
    main_config = _require_mapping(stages_config, "main")
    cooldown_config = _require_mapping(stages_config, "cooldown")
    base_environment = _require_mapping(recipe, "environment")
    overrides = condition.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ConfigError("condition overrides must be a mapping")

    unknown = sorted(set(overrides) - set(base_environment))
    if unknown:
        raise ConfigError(
            "condition overrides contain keys absent from the base recipe: " + ", ".join(unknown)
        )

    env = {
        key: _stringify_env_value(value, f"environment.{key}")
        for key, value in base_environment.items()
    }
    env.update(
        {
            key: _stringify_env_value(value, f"overrides.{key}")
            for key, value in overrides.items()
        }
    )
    _validate_masking(env)

    project_name = _validate_name(execution.get("project_name"), "project name")
    name_template = str(execution.get("experiment_name", "{recipe}__{condition}"))
    fields = {
        "recipe": recipe_name,
        "condition": condition_name,
        "study": study,
        "project": project_name,
    }
    experiment_name = _validate_name(
        _render_template(name_template, fields, "execution.experiment_name"),
        "resolved experiment name",
    )
    fields["experiment"] = experiment_name

    submission_script_value = execution.get("submission_script")
    if not isinstance(submission_script_value, str):
        raise ConfigError("execution.submission_script must be a path")
    submission_script = _expand_path(submission_script_value, recipe_path.parent)
    if not submission_script.is_file():
        raise ConfigError(f"submission script does not exist: {submission_script}")
    sbatch_args_value = execution.get("sbatch_args", [])
    if not isinstance(sbatch_args_value, list) or not all(
        isinstance(value, str) and value.startswith("--") for value in sbatch_args_value
    ):
        raise ConfigError("execution.sbatch_args must be a list of long-form '--...' options")
    sbatch_args = tuple(sbatch_args_value)

    checkpoint_template = str(execution.get("checkpoint_root", ""))
    artifacts_template = str(execution.get("artifacts_root", "runs"))
    checkpoint_root = _expand_path(
        _render_template(checkpoint_template, fields, "execution.checkpoint_root"),
        recipe_path.parent,
    )
    artifacts_root = _expand_path(
        _render_template(artifacts_template, fields, "execution.artifacts_root"),
        recipe_path.parent,
    )

    train_tokens = _require_positive_int(main_config.get("train_tokens"), "main.train_tokens")
    cooldown_tokens = _require_positive_int(
        cooldown_config.get("tokens"), "cooldown.tokens"
    )
    source_iterations = cooldown_config.get("source_iterations")
    if not isinstance(source_iterations, list) or not source_iterations:
        raise ConfigError("cooldown.source_iterations must be a non-empty list")
    source_iterations = [
        _require_positive_int(value, "cooldown source iteration") for value in source_iterations
    ]
    if len(source_iterations) != len(set(source_iterations)):
        raise ConfigError("cooldown source iterations must be unique")
    source_iterations.sort()

    gbs = _require_positive_int(int(env["GBS"]), "GBS")
    seq_len = _require_positive_int(int(env["SEQ_LEN"]), "SEQ_LEN")
    save_tokens = _require_positive_int(int(env["SAVE_EVERY_TOKENS"]), "SAVE_EVERY_TOKENS")
    tokens_per_iteration = gbs * seq_len
    save_interval = (save_tokens + tokens_per_iteration // 2) // tokens_per_iteration
    if save_interval <= 0:
        raise ConfigError("SAVE_EVERY_TOKENS rounds below one iteration")
    main_iterations = math.ceil(train_tokens / tokens_per_iteration)
    for source_iteration in source_iterations:
        if source_iteration > main_iterations:
            raise ConfigError(
                f"cooldown source iteration {source_iteration} exceeds main target iteration "
                f"{main_iterations}"
            )
        if source_iteration % save_interval != 0 and source_iteration != main_iterations:
            raise ConfigError(
                f"cooldown source iteration {source_iteration} is not a persistent checkpoint "
                f"milestone (save interval {save_interval}) or the final main iteration"
            )

    common_env = {
        **env,
        "PROJECT_NAME": project_name,
        "EXP_NAME": experiment_name,
        "CHECKPOINT_ROOT": str(checkpoint_root),
    }
    main_env = {**common_env, "RUN_MODE": "main", "TRAIN_TOKENS": str(train_tokens)}
    stages = [Stage("main", "main", None, main_env)]
    for source_iteration in source_iterations:
        stages.append(
            Stage(
                key=f"cooldown-from-{source_iteration:07d}",
                mode="cooldown",
                source_iteration=source_iteration,
                environment={
                    **common_env,
                    "RUN_MODE": "cooldown",
                    "SOURCE_ITER": str(source_iteration),
                    "COOLDOWN_TOKENS": str(cooldown_tokens),
                },
            )
        )

    resolved = {
        "schema_version": 1,
        "recipe": {"name": recipe_name, "path": str(recipe_path)},
        "condition": {
            "name": condition_name,
            "study": study,
            "description": description,
            "path": str(config_path),
            "overrides": overrides,
        },
        "execution": {
            "experiment_name": experiment_name,
            "project_name": project_name,
            "submission_script": str(submission_script),
            "sbatch_args": list(sbatch_args),
            "checkpoint_root": str(checkpoint_root),
            "artifacts_root": str(artifacts_root),
        },
        "derived": {
            "tokens_per_iteration": tokens_per_iteration,
            "persistent_save_interval": save_interval,
            "main_iterations": main_iterations,
        },
        "stages": [
            {
                "key": stage.key,
                "mode": stage.mode,
                "source_iteration": stage.source_iteration,
                "environment": stage.environment,
            }
            for stage in stages
        ],
    }
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()
    resolved["config_hash"] = config_hash
    return ResolvedExperiment(
        config_path=config_path,
        recipe_path=recipe_path,
        recipe_name=recipe_name,
        condition_name=condition_name,
        description=description,
        study=study,
        experiment_name=experiment_name,
        submission_script=submission_script,
        project_name=project_name,
        checkpoint_root=checkpoint_root,
        artifacts_root=artifacts_root,
        sbatch_args=sbatch_args,
        stages=tuple(stages),
        resolved=resolved,
        config_hash=config_hash,
    )


def load_collection(collection_path: str | Path) -> tuple[str, tuple[ResolvedExperiment, ...]]:
    """Load and validate every condition listed in a collection manifest."""
    collection_path = Path(collection_path).expanduser().resolve()
    collection = _load_yaml(collection_path)
    if collection.get("schema_version") != 1:
        raise ConfigError("collection schema_version must be 1")
    name = _validate_name(collection.get("name"), "collection name")
    condition_refs = collection.get("conditions")
    if not isinstance(condition_refs, list) or not condition_refs:
        raise ConfigError("collection.conditions must be a non-empty list of paths")
    if not all(isinstance(value, str) and value for value in condition_refs):
        raise ConfigError("collection.conditions must contain only non-empty paths")
    experiments = tuple(
        load_experiment(_expand_path(reference, collection_path.parent))
        for reference in condition_refs
    )
    names = [experiment.experiment_name for experiment in experiments]
    duplicates = sorted({value for value in names if names.count(value) > 1})
    if duplicates:
        raise ConfigError(
            "collection resolves to duplicate experiment names: " + ", ".join(duplicates)
        )
    return name, experiments
