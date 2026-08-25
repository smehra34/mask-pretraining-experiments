"""Command-line interface for planning, rendering, and submitting experiments."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml

from experiment_manager.config import ConfigError, load_collection, load_experiment
from experiment_manager.evaluation import (
    append_evaluation_submission,
    create_evaluation_record,
    render_spellbook_script,
    resolve_evaluation,
)
from experiment_manager.records import (
    append_submission,
    command_from_record,
    create_run_record,
    load_run_record,
    resolved_stage,
    shell_command,
)
from experiment_manager.slurm import query, submit, with_dependency


DEFAULT_EVALUATION_CONFIG = Path(__file__).resolve().parents[1] / "evaluations/suites.yaml"


def _print_plan(experiment) -> None:
    derived = experiment.resolved["derived"]
    print(f"Experiment: {experiment.experiment_name}")
    print(f"Study:      {experiment.study}")
    print(f"Recipe:     {experiment.recipe_name}")
    print(f"Condition:  {experiment.condition_name}")
    if experiment.description:
        print(f"Description: {experiment.description}")
    print(f"Config hash: {experiment.config_hash}")
    print(f"Checkpoint root: {experiment.checkpoint_root}")
    print(
        "Derived: "
        f"{derived['tokens_per_iteration']} tokens/iteration, "
        f"save every {derived['persistent_save_interval']} iterations, "
        f"main target {derived['main_iterations']} iterations"
    )
    print("\nStages:")
    for stage in experiment.stages:
        source = "-" if stage.source_iteration is None else str(stage.source_iteration)
        tokens = stage.environment.get("TRAIN_TOKENS") or stage.environment.get("COOLDOWN_TOKENS")
        masking = (
            f"{stage.environment.get('INPUT_MASK_STRATEGY')} "
            f"ratio={stage.environment.get('INPUT_MASK_RATIO')} "
            f"span={stage.environment.get('INPUT_MASK_SPAN_LENGTH')}"
        )
        print(f"  {stage.key:<30} source={source:<8} tokens={tokens:<14} {masking}")
    print("\nExact submission commands:")
    for stage in experiment.stages:
        print(f"  [{stage.key}]\n    {shlex.join(shell_command(experiment, stage))}")


def cmd_plan(args: argparse.Namespace) -> None:
    _print_plan(load_experiment(args.config))


def cmd_render(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.config)
    run_dir = create_run_record(experiment)
    print(f"Rendered immutable run record: {run_dir}")


def cmd_plan_collection(args: argparse.Namespace) -> None:
    name, experiments = load_collection(args.collection)
    print(f"Collection: {name} ({len(experiments)} conditions)")
    for index, experiment in enumerate(experiments):
        if index:
            print("\n" + "=" * 88 + "\n")
        _print_plan(experiment)


def cmd_render_collection(args: argparse.Namespace) -> None:
    name, experiments = load_collection(args.collection)
    print(f"Rendering collection: {name}")
    for experiment in experiments:
        run_dir = create_run_record(experiment)
        print(f"  {experiment.condition_name:<24} {run_dir}")


def _ensure_main_not_started(checkpoint_root: Path) -> None:
    main_dir = checkpoint_root / "main"
    tracker = main_dir / "latest_checkpointed_iteration.txt"
    numbered = list(main_dir.glob("iter_*")) if main_dir.is_dir() else []
    if tracker.exists() or numbered:
        raise RuntimeError(
            f"main checkpoint directory already contains training state: {main_dir}\n"
            "Use 'resume RUN_DIR --stage main' to continue an existing recorded run, or choose "
            "a new condition/experiment name."
        )


def cmd_submit_main(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.config)
    _ensure_main_not_started(experiment.checkpoint_root)
    run_dir = create_run_record(experiment)
    _, resolved = load_run_record(run_dir)
    main_stage = resolved_stage(resolved, "main")
    command = command_from_record(resolved, main_stage)
    job_id = submit(command)
    append_submission(
        run_dir, stage_key="main", job_id=job_id, command=command, action="submit-main"
    )
    print(f"Submitted main job {job_id}")
    print(f"Run record: {run_dir}")


def _source_checkpoint(resolved: dict, source_iteration: int) -> Path:
    checkpoint_root = Path(resolved["execution"]["checkpoint_root"])
    return checkpoint_root / "main" / f"iter_{source_iteration:07d}"


def cmd_submit_cooldowns(args: argparse.Namespace) -> None:
    run_dir, resolved = load_run_record(args.run)
    cooldowns = [stage for stage in resolved["stages"] if stage["mode"] == "cooldown"]
    for stage in cooldowns:
        source = int(stage["source_iteration"])
        checkpoint = _source_checkpoint(resolved, source)
        if not args.skip_checkpoint_check and args.dependency is None and not checkpoint.is_dir():
            raise RuntimeError(
                f"source checkpoint is missing for {stage['key']}: {checkpoint}\n"
                "Wait for the main run, pass --dependency MAIN_JOB_ID, or explicitly use "
                "--skip-checkpoint-check."
            )
        branch_tracker = (
            Path(resolved["execution"]["checkpoint_root"])
            / "cooldowns"
            / f"from_iter_{source:07d}"
            / "checkpoints"
            / "latest_checkpointed_iteration.txt"
        )
        if branch_tracker.exists():
            raise RuntimeError(
                f"cooldown branch has already started: {branch_tracker.parent}\n"
                f"Use 'resume {run_dir} --stage {stage['key']}' instead."
            )
    for stage in cooldowns:
        command = command_from_record(resolved, stage)
        job_id = submit(command, dependency=args.dependency)
        submitted_command = with_dependency(command, args.dependency)
        append_submission(
            run_dir,
            stage_key=stage["key"],
            job_id=job_id,
            command=submitted_command,
            action="submit-cooldown",
        )
        print(f"Submitted {stage['key']} as job {job_id}")
    print(f"Updated run record: {run_dir}")


def cmd_submit_all(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.config)
    _ensure_main_not_started(experiment.checkpoint_root)
    run_dir = create_run_record(experiment)
    _, resolved = load_run_record(run_dir)
    main_stage = resolved_stage(resolved, "main")
    main_command = command_from_record(resolved, main_stage)
    main_job_id = submit(main_command)
    append_submission(
        run_dir,
        stage_key="main",
        job_id=main_job_id,
        command=main_command,
        action="submit-all",
    )
    print(f"Submitted main as job {main_job_id}")
    cooldowns = [stage for stage in resolved["stages"] if stage["mode"] == "cooldown"]
    for stage in cooldowns:
        command = command_from_record(resolved, stage)
        job_id = submit(command, dependency=main_job_id)
        submitted_command = with_dependency(command, main_job_id)
        append_submission(
            run_dir,
            stage_key=stage["key"],
            job_id=job_id,
            command=submitted_command,
            action="submit-all",
        )
        print(f"Submitted {stage['key']} as dependent job {job_id}")
    print(f"Run record: {run_dir}")


def cmd_resume(args: argparse.Namespace) -> None:
    run_dir, resolved = load_run_record(args.run)
    stage = resolved_stage(resolved, args.stage)
    if stage["mode"] == "cooldown" and not args.skip_checkpoint_check:
        checkpoint = _source_checkpoint(resolved, int(stage["source_iteration"]))
        if not checkpoint.is_dir():
            raise RuntimeError(f"cooldown source checkpoint is missing: {checkpoint}")
    command = command_from_record(resolved, stage)
    job_id = submit(command)
    append_submission(
        run_dir, stage_key=stage["key"], job_id=job_id, command=command, action="resume"
    )
    print(f"Resubmitted {stage['key']} as job {job_id}")


def cmd_status(args: argparse.Namespace) -> None:
    run_dir, _ = load_run_record(args.run)
    jobs = yaml.safe_load((run_dir / "jobs.yaml").read_text())
    submissions = jobs.get("submissions", [])
    if submissions:
        print("Recorded submissions:")
        for item in submissions:
            print(
                f"  {item['submitted_at']}  {item['stage']:<30} "
                f"job={item['job_id']} action={item['action']}"
            )
        print()
    print(query([str(item["job_id"]) for item in submissions]))


def _evaluation_step(value: str) -> int | None:
    if value == "latest":
        return None
    if not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError("checkpoint step must be 'latest' or a positive integer")
    return int(value)


def _resolved_evaluations(args: argparse.Namespace, *, require_checkpoint: bool):
    return [
        resolve_evaluation(
            args.run,
            stage_key=args.stage,
            checkpoint_step=args.step,
            suite_name=suite,
            config_path=args.eval_config,
            allow_unsafe_code=args.allow_unsafe_code,
            require_checkpoint=require_checkpoint and not args.skip_checkpoint_check,
        )
        for suite in args.suite
    ]


def cmd_plan_eval(args: argparse.Namespace) -> None:
    evaluations = _resolved_evaluations(args, require_checkpoint=False)
    for index, evaluation in enumerate(evaluations):
        if index:
            print("\n" + "=" * 88 + "\n")
        print(f"Evaluation: {evaluation.evaluation_name}")
        print(f"Suite:      {evaluation.suite.name} ({', '.join(evaluation.suite.tasks)})")
        print(f"Checkpoint: {evaluation.checkpoint_dir}/iter_{evaluation.checkpoint_step:07d}")
        print(f"Tokens seen: {evaluation.training['tokens_seen']}")
        print(f"W&B ID:     {evaluation.wandb_id}")
        print(f"Config hash: {evaluation.config_hash}")
        print("\nRendered Spellbook sbatch script:\n")
        print(render_spellbook_script(evaluation, log_dir=Path("EVAL_RECORD/slurm")))


def cmd_render_eval(args: argparse.Namespace) -> None:
    for evaluation in _resolved_evaluations(args, require_checkpoint=True):
        record_dir = create_evaluation_record(evaluation)
        print(f"Rendered immutable evaluation record: {record_dir}")


def cmd_submit_eval(args: argparse.Namespace) -> None:
    for evaluation in _resolved_evaluations(args, require_checkpoint=True):
        record_dir = create_evaluation_record(evaluation)
        command = ["sbatch", str(record_dir / "submit.sh")]
        job_id = submit(command)
        append_evaluation_submission(record_dir, job_id)
        print(f"Submitted {evaluation.suite.name} evaluation as job {job_id}")
        print(f"Evaluation record: {record_dir}")


def _add_evaluation_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("run", help="Path to an immutable training run-record directory")
    command.add_argument("--stage", default="main", help="Training stage key (default: main)")
    command.add_argument(
        "--step",
        default=None,
        type=_evaluation_step,
        help="Checkpoint iteration or 'latest' (default: latest)",
    )
    command.add_argument(
        "--suite",
        action="append",
        required=True,
        help="Named suite from evaluations/suites.yaml; repeat to submit multiple suites",
    )
    command.add_argument(
        "--eval-config",
        default=str(DEFAULT_EVALUATION_CONFIG),
        help="Evaluation suite/backend YAML",
    )
    command.add_argument("--skip-checkpoint-check", action="store_true")
    command.add_argument(
        "--allow-unsafe-code",
        action="store_true",
        help="Allow suites such as HumanEval/MBPP that execute generated Python",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mask-exp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text, handler in (
        ("plan", "Validate and show a condition without writing files", cmd_plan),
        ("render", "Create an immutable run record without submitting", cmd_render),
        ("submit-main", "Create a run record and submit its main stage", cmd_submit_main),
        ("submit-all", "Submit main and afterok-dependent cooldown stages", cmd_submit_all),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("config", help="Path to a condition YAML")
        command.set_defaults(handler=handler)

    for name, help_text, handler in (
        ("plan-collection", "Validate and show every condition in a collection", cmd_plan_collection),
        (
            "render-collection",
            "Create a run record for every condition without submitting",
            cmd_render_collection,
        ),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("collection", help="Path to a collection YAML")
        command.set_defaults(handler=handler)

    cooldowns = subparsers.add_parser(
        "submit-cooldowns", help="Submit cooldowns from an existing immutable run record"
    )
    cooldowns.add_argument("run", help="Path to the run-record directory")
    cooldowns.add_argument("--dependency", help="Submit with afterok dependency on this job ID")
    cooldowns.add_argument("--skip-checkpoint-check", action="store_true")
    cooldowns.set_defaults(handler=cmd_submit_cooldowns)

    resume = subparsers.add_parser("resume", help="Resubmit one exact recorded stage")
    resume.add_argument("run", help="Path to the run-record directory")
    resume.add_argument("--stage", required=True, help="main or cooldown-from-NNNNNNN")
    resume.add_argument("--skip-checkpoint-check", action="store_true")
    resume.set_defaults(handler=cmd_resume)

    status = subparsers.add_parser("status", help="Show recorded and current Slurm status")
    status.add_argument("run", help="Path to the run-record directory")
    status.set_defaults(handler=cmd_status)

    plan_eval = subparsers.add_parser(
        "plan-eval", help="Resolve and print Spellbook evaluation jobs without writing files"
    )
    _add_evaluation_arguments(plan_eval)
    plan_eval.set_defaults(handler=cmd_plan_eval)

    render_eval = subparsers.add_parser(
        "render-eval", help="Create immutable Spellbook evaluation records without submitting"
    )
    _add_evaluation_arguments(render_eval)
    render_eval.set_defaults(handler=cmd_render_eval)

    submit_eval = subparsers.add_parser(
        "submit-eval", help="Create records and submit Spellbook lm-eval jobs"
    )
    _add_evaluation_arguments(submit_eval)
    submit_eval.set_defaults(handler=cmd_submit_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
