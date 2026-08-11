"""Small, testable Slurm command boundary."""

from __future__ import annotations

import subprocess


def with_dependency(command: list[str], dependency: str | None) -> list[str]:
    """Return the exact argv submitted to Slurm, including an optional dependency."""
    actual = list(command)
    if dependency:
        actual.insert(1, f"--dependency=afterok:{dependency}")
    return actual


def submit(command: list[str], dependency: str | None = None) -> str:
    actual = with_dependency(command, dependency)
    result = subprocess.run(actual, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    fields = result.stdout.strip().split()
    if not fields or not fields[-1].isdigit():
        raise RuntimeError(f"could not parse Slurm job ID from: {result.stdout!r}")
    return fields[-1]


def query(job_ids: list[str]) -> str:
    if not job_ids:
        return "No jobs have been submitted from this run record."
    joined = ",".join(job_ids)
    result = subprocess.run(
        ["sacct", "-j", joined, "--format=JobID,JobName,State,Elapsed,ExitCode", "-X"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sacct failed: {result.stderr.strip()}")
    return result.stdout.rstrip()
