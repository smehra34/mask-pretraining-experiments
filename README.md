# Mask pretraining experiment manager

This is a self-contained, YAML-driven experiment layer. It includes its own copy of the 1B MEAP
submission script, leaving the older `mask-pretraining` setup untouched. A recipe holds invariant
model/data/stage settings; small condition files contain only ablation overrides.

## Layout

```text
recipes/                         fixed model, data, Slurm, main/cooldown setup
studies/1b_masking_ablation/     one YAML file per experimental condition
collections/                     ordered groups of conditions
submission/                      submission scripts owned by this repository
runs/                            generated, ignored, immutable run records
experiment_manager/              resolver, validation, records, and Slurm interface
mask_exp.py                      local command-line entry point
```

The included `1b_llama` recipe defines a modern approximately 1.15B-parameter Llama-family model
and a 200B-token WSD trunk. It uses tied embeddings, 4:1 grouped-query attention, and Muon for
eligible matrix weights with Adam for the remaining parameters. Explicit 10B-token cooldown
branches from persistent milestones near 30B, 50B, and 190B produce models near 40B, 60B, and
200B training-token budgets.
Each condition gets a distinct experiment/checkpoint name, such as
`1b-masking-ablation__span-020-s2`.

## Basic workflow

Use Python 3.11 on the current system and make sure `SCRATCH` is defined. It is
used for dataset-index caches; the Megatron checkout is under `~/developer`,
and durable checkpoints do not use it:

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

## Short Muon selection sweep

Before running objective comparisons, validate and submit the unmasked Muon sweep:

```bash
/usr/bin/python3.11 mask_exp.py plan-collection collections/1b_muon_sweep.yaml

for condition in studies/1b_muon_sweep/*.yaml; do
  /usr/bin/python3.11 mask_exp.py submit-main "$condition"
done
```

Each condition targets 5B tokens (4,769 updates), warms up for 500 updates, and evaluates clean
validation data every 100 updates. The five conditions compare peak learning rates of 2e-4,
4e-4, and 8e-4 at unit Muon scale, plus Muon scale multipliers 0.2 and 0.5 at the center learning
rate. Main mode deliberately uses only WSD warmup and stable training; cooldown is excluded from
optimizer selection. Inspect all conditions near 1B tokens and stop clearly unstable or inferior
runs, then select among the survivors using validation loss and its trend over the later stable
phase. Freeze the selected settings in `recipes/1b_llama.yaml` before defining masking or MTP
comparisons.

All studies log to the shared `mask_pretraining` W&B project. Runs are grouped by study and tagged
with their study, recipe, condition, and stage, allowing either cross-study comparison or filtered
study-specific views. At the measured ~5.88 seconds per update, a complete 5B-token Muon condition
takes about 7.8 hours of training time; the sweep requests 10 hours to cover startup, validation,
and checkpointing.

## Defining experiments

Copy a condition file and change `name`, `description`, and `overrides`:

```yaml
schema_version: 1
recipe: ../../recipes/1b_llama.yaml
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
`[control_768]` token from the Mistral v0.3 tokenizer, and validation ensures mask ratios, strategies, span lengths, stage token
counts, and cooldown checkpoint alignment are coherent.

Each cooldown source has an independent token budget. For example:

```yaml
stages:
  main:
    train_tokens: 100000000000
  cooldown:
    branches:
      - source_iteration: 2384
        tokens: 1000000000
      - source_iteration: 11920
        tokens: 10000000000
      - source_iteration: 21456
        tokens: 20000000000
```

This launches 1B-, 10B-, and 20B-token cooldowns respectively. `source_iteration` identifies the
exact persistent main checkpoint; `tokens` is additional training performed by that branch. The
older shared `cooldown.tokens` plus `source_iterations` syntax remains accepted for compatibility.

## Short end-to-end smoke test

The isolated `test_run` setup uses the production 1B architecture and four-GPU topology, but limits
each Slurm job to 45 minutes. It trains the main stage for 100M tokens (24 iterations), creates one
rolling checkpoint halfway through, saves persistently at the end, and defines a 10M-token cooldown
from the final iteration. It writes under the separate
`mask_pretraining_test/test_run` checkpoint
namespace and prints one augmented span-masking example per job.

```bash
export SCRATCH=/iopsstor/scratch/cscs/smehra
/usr/bin/python3.11 mask_exp.py plan studies/test_run/test_run.yaml
/usr/bin/python3.11 mask_exp.py submit-main studies/test_run/test_run.yaml
```

After the main job completes, use the run directory printed by `submit-main`:

```bash
/usr/bin/python3.11 mask_exp.py submit-cooldowns RUN_DIRECTORY
```

The `1b_llama` launcher intentionally fixes the architecture and distributed topology. Recipes
control data, schedules, batch sizes, seed, logging/evaluation cadence, regularization, masking,
checkpointing, and Slurm resources. Add another family launcher and recipe when architecture or
topology changes. The earlier `1B-meap-config.sh` and `1b_meap.yaml` are retained for compatibility.

To add another model or dataset, create another recipe rather than duplicating every condition.
It can reference a different base submission script and declare a different environment and stage
schedule. Conditions remain small and can opt into any variables that recipe exposes, so this
structure is not intrinsically limited to masking experiments.

## Reproducibility records

Each render creates `runs/STUDY/CONDITION/TIMESTAMP-HASH/` containing:

- `resolved.yaml`: all merged values and fully materialized stages;
- `source/`: an executable snapshot of the selected family launcher;
- `scripts/`: executable snapshots of exact `sbatch` commands;
- `metadata.yaml`: source paths and Git state for this manager, the submission setup, and Megatron;
- `jobs.yaml`: append-only submission history and job IDs.
- `stages/STAGE/slurm/`: the sole Slurm stdout/stderr location for each submission;
- `stages/STAGE/logging/`: TensorBoard and other reproducibility logs;
- `stages/STAGE/debug/`: optional NCCL and masking diagnostics.

Checkpoints and local W&B state are colocated under
`/capstor/scratch/cscs/smehra/megatron-runs/`. Every launcher reapplies the
configured composite Lustre default layout to that base before it creates any
experiment descendants. The tokenized DCLM-Edu dataset lives under
`/iopsstor/scratch/cscs/smehra/tokenized_datasets/`; raw source Parquet remains
on Capstor.

Submission and resume operations use the frozen record and its launcher snapshot rather than
re-reading a potentially edited condition or active launcher. This lets future recipes expose or
hide parameters without changing an already rendered experiment. Commit recipes, condition files,
and manager code; keep generated `runs/` as local run artifacts or archive them with experiment
outputs.

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
