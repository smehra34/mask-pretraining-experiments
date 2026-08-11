# Mask pretraining experiment manager

This is a standalone, YAML-driven layer over the existing
`mask-pretraining/1B-meap-config.sh` submission script. It does not modify that setup. A recipe
holds invariant model/data/stage settings; small condition files contain only ablation overrides.

## Layout

```text
recipes/                         fixed model, data, Slurm, main/cooldown setup
studies/1b_masking_ablation/     one YAML file per experimental condition
collections/                     ordered groups of conditions
runs/                            generated, ignored, immutable run records
experiment_manager/              resolver, validation, records, and Slurm interface
mask_exp.py                      local command-line entry point
```

The included `1b_meap` recipe defines 100B-token main runs and 10B-token cooldowns from three
persistent checkpoint milestones. Each condition gets a distinct experiment/checkpoint name,
such as `1b-masking-ablation__span-020-s2`.

## Basic workflow

Use Python 3.11 on the current system and make sure `SCRATCH` is defined:

```bash
cd /users/smehra/developer/mask-experiment-manager
export SCRATCH=/iopsstor/scratch/cscs/smehra

# Validate one condition and print every exact sbatch command. This writes nothing.
/usr/bin/python3.11 mask_exp.py plan studies/1b_masking_ablation/span_020_s2.yaml

# Validate the complete ablation matrix. This writes nothing.
/usr/bin/python3.11 mask_exp.py plan-collection collections/1b_masking_ablation.yaml

# Freeze a condition into an immutable, timestamped record without submitting it.
/usr/bin/python3.11 mask_exp.py render studies/1b_masking_ablation/span_020_s2.yaml

# Recommended: submit the main stage first.
/usr/bin/python3.11 mask_exp.py submit-main studies/1b_masking_ablation/span_020_s2.yaml
```

`submit-main` prints the resulting run directory. Once desired source checkpoints exist, submit
the recorded cooldowns:

```bash
/usr/bin/python3.11 mask_exp.py submit-cooldowns RUN_DIRECTORY
```

For an uninterrupted scheduled chain, `submit-all CONDITION.yaml` submits all cooldowns with an
`afterok` dependency on the main job. This means every cooldown waits for the entire main job,
not merely for its source checkpoint. Prefer the explicit two-step workflow when inspecting main
checkpoints before launching branches.

If Slurm interrupts a stage, resubmit its exact frozen configuration:

```bash
/usr/bin/python3.11 mask_exp.py resume RUN_DIRECTORY --stage main
/usr/bin/python3.11 mask_exp.py resume RUN_DIRECTORY --stage cooldown-from-0002384
/usr/bin/python3.11 mask_exp.py status RUN_DIRECTORY
```

The underlying submission script detects its rolling checkpoint and resumes. The manager refuses
to start a new main run in an occupied checkpoint namespace and refuses to start a cooldown branch
that already has a tracker; use `resume` for those cases.

## Defining experiments

Copy a condition file and change `name`, `description`, and `overrides`:

```yaml
schema_version: 1
recipe: ../../recipes/1b_meap.yaml
study: 1b-masking-ablation
name: span-030-s4
description: Replace 30 percent of eligible inputs in spans of four.
overrides:
  INPUT_MASK_RATIO: 0.30
  INPUT_MASK_STRATEGY: span
  INPUT_MASK_SPAN_LENGTH: 4
```

Only environment keys declared by the recipe may be overridden, so misspelled parameters fail
validation. `INPUT_MASK_RATIO: 0.0` is the vanilla NTP control. The recipe uses the reserved
`<SPECIAL_999>` token, and validation ensures mask ratios, strategies, span lengths, stage token
counts, and cooldown checkpoint alignment are coherent.

To add another model or dataset, create another recipe rather than duplicating every condition.
It can reference a different base submission script and declare a different environment and stage
schedule. Conditions remain small and can opt into any variables that recipe exposes, so this
structure is not intrinsically limited to masking experiments.

## Reproducibility records

Each render creates `runs/STUDY/CONDITION/TIMESTAMP-HASH/` containing:

- `resolved.yaml`: all merged values and fully materialized stages;
- `scripts/`: executable snapshots of exact `sbatch` commands;
- `metadata.yaml`: source paths and Git state for this manager, the submission setup, and Megatron;
- `jobs.yaml`: append-only submission history and job IDs.

Submission and resume operations use the frozen record rather than re-reading a potentially edited
condition. Commit recipes, condition files, and manager code; keep generated `runs/` as local run
artifacts or archive them with experiment outputs.

## Installation and tests

The local entry point works without installation. For an isolated editable install:

```bash
/usr/bin/python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/mask-exp plan studies/1b_masking_ablation/vanilla.yaml
```

The standard-library tests do not require pytest:

```bash
SCRATCH=/iopsstor/scratch/cscs/smehra /usr/bin/python3.11 -m unittest discover -v
```
