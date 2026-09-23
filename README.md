# Mask pretraining experiment manager

This is a self-contained, YAML-driven experiment layer. It includes its own copy of the 1B MEAP
submission script, leaving the older `mask-pretraining` setup untouched. A recipe holds invariant
model/data/stage settings; small condition files contain only ablation overrides.

## Layout

```text
recipes/                         fixed model, data, Slurm, main/cooldown setup
studies/*_objective_screen/      one YAML file per experimental condition
collections/                     ordered groups of conditions
submission/                      submission scripts owned by this repository
runs/                            generated, ignored, immutable run records
experiment_manager/              resolver, validation, records, and Slurm interface
mask_exp.py                      local command-line entry point
```

The `300m_llama` and `1b_llama` recipes define frozen approximately 304M- and
1.15B-parameter members of one Llama family. Both use 24 layers, 64-dimensional
attention heads, 4:1 GQA, tied embeddings, and the same SwiGLU ratio. Their
12B- and 40B-token budgets each end with a 10% WSD cooldown. Muon optimizer
values marked TODO remain provisional until the queued calibration completes.

## 3B objective screen

`recipes/3b_llama.yaml` adds a 3,073,563,648-parameter dense member for a direct
scaling comparison. It has 30 layers, hidden size 3072, SwiGLU FFN size 8192,
48 64-dimensional query heads, and 12 KV groups. The count assumes a bias-free
Llama decoder with two RMSNorm weights per layer, one final RMSNorm, a padded
32,768-token vocabulary, tied input/output embeddings, and no MTP block. All
3,073,563,648 base parameters are trainable in every condition; input masking
adds no parameters. The MTP condition enables one sequential predictor layer
for two-token prediction using the recipe's shared MTP loss scale of 0.1.
Megatron's theoretical counter assigns that dense MTP block 99,096,576
objective-specific parameters, for 3,172,660,224 total trainable parameters in
the MTP condition.

The width is a hardware-aligned 1.5x increase over 1.1B. Thirty layers place
the model close to 3B without forcing an exact decimal target, while keeping
the depth moderate and all hidden, FFN, head, GQA, and tensor-parallel
divisibility constraints clean. Initialization follows the existing family
rule `sqrt(0.4 / hidden_size)`, giving `0.01141088661469096` rather than copying
the 1.1B constant.

All four conditions train for a nominal 120B tokens: a 108B stable WSD trunk and a
12B `minus_sqrt` cooldown loaded from the final trunk checkpoint without a new
warmup. At 256 sequences x 4096 tokens, an update contains 1,048,576 tokens.
The main stage therefore ends at iteration 102,997 and the cooldown runs for
11,445 more updates, ending at absolute iteration 114,442. Decimal-billion
budgets cannot be divided exactly by this update size: the launcher schedules
108,000,182,272 main tokens and 12,000,952,320 cooldown tokens
(120,001,134,592 total) while retaining the exact declared 108B/12B/120B
scientific budgets.

Persistent saves occur every 11,444 updates (11,999,903,744 tokens), producing
useful landmarks near 24B, 36B, 72B, and 108B; the launcher also saves the exact
final-main and final-cooldown endpoints. The single regular-save interval
cannot additionally make 3B permanent without retaining many checkpoints.
Rolling recovery saves occur every 1,431 updates (1,500,512,256 tokens); the
second rolling save at iteration 2,862 (3,001,024,512 tokens) is the early
learning-rate stability gate and is not a permanent scientific checkpoint.

The production 1.1B optimizer is transferred unchanged: Muon at peak LR
`0.0008`, minimum LR `0.00008`, momentum 0.95, Nesterov, spectral scaling,
extra scale 0.2, five Newton-Schulz steps, Adam for scalar parameters, weight
decay 0.05, 1.0 gradient clipping, Adam betas 0.9/0.95, epsilon 1e-8, and 1,000
warmup updates. This LR is transferred, not tuned or claimed optimal for 3B;
the ~3B gate must be checked for stability before committing the full budget.
NTP has masking disabled. MTP adds one sequential Megatron MTP layer that
predicts one additional future token, also without input masking. One masking
condition independently masks 15% of input tokens without explicit spans. The
other masks 15% with the existing truncated-geometric (`p=0.5`) variable-span
implementation and maximum span length five. Both disable MTP. Architecture,
optimizer, data, schedule, and execution settings remain invariant across all
four conditions.

Training uses eight four-GPU nodes, TP=PP=CP=1, data parallel size 32,
micro-batch size 2, and four gradient-accumulation microbatches per update.
The repository's Megatron estimator gives roughly 18.25 GiB per rank for model
and distributed-optimizer state and roughly 22 GiB for unrecomputed
activations, before runtime/workspace overhead. This makes the topology a
conservative fit for the available accelerator memory while preserving the
1.1B global batch and sequence length. Each allocation requests 72 CPUs and
460 GB host memory per node for up to 12 hours; rolling resume is expected for
the long trunk. Scaling the measured 1.1B 5.88-second update by parameter count
and twice as many GPUs suggests roughly 8 seconds per update before larger-run
communication effects. A conservative 9–12 seconds per update puts the main
trunk at about 258–343 elapsed hours (roughly 22–29 full allocations) and the
cooldown at about 29–38 hours. These are planning estimates, not measured 3B
throughput.

The local dataset manifest records 149,999,999,419 uint16 tokens. With the
unchanged `995,5,0` split, about 149.25B tokens are eligible for training, so a
120B run consumes roughly 0.80 corpus passes (before sequence-boundary and
sampling details). Dataset paths, tokenizer, cache, blend ordering, and seed
3407 are identical to the 1.1B screen. Existing zero-shot core, five-shot core,
GSM8K, and Paloma evaluation definitions work with the resulting standard
Megatron distributed checkpoints; this screen adds no evaluation tasks.

Settings changed from the 1.1B recipe are limited to model scale and execution
capacity: layers 24 -> 30, hidden size 2048 -> 3072, FFN 5632 -> 8192, heads
32 -> 48, KV groups 8 -> 12, initialization std 0.013975424859373685 ->
0.01141088661469096 (family scaling), micro-batch 8 -> 2 and nodes 4 -> 8
(memory/capacity), main tokens 36B -> 108B and cooldown tokens 4B -> 12B
(requested scaling budget), cooldown source 34,333 -> 102,997 (new final-main
update), persistent saves 10B -> 12B (scientific landmarks/storage), and
rolling saves 1B -> 1.5B (recovery cost and the ~3B gate). All other recipe
settings are mirrored.

Planning is read-only:

```bash
cd /users/smehra/developer/mask-experiment-manager
export SCRATCH=/iopsstor/scratch/cscs/smehra
/usr/bin/python3.11 mask_exp.py plan-collection collections/3b_objective_screen.yaml
```

After reviewing the plan and early-gate procedure, submit only the NTP main
stage with:

```bash
/usr/bin/python3.11 mask_exp.py submit-main studies/3b_objective_screen/ntp.yaml
```

## Basic workflow

Use Python 3.11 on the current system and make sure `SCRATCH` is defined. It is
used for dataset-index caches; the Megatron checkout is under `~/developer`,
and durable checkpoints do not use it:

```bash
cd /users/smehra/developer/mask-experiment-manager
export SCRATCH=/iopsstor/scratch/cscs/smehra

# Validate one condition and print every exact sbatch command. This writes nothing.
/usr/bin/python3.11 mask_exp.py plan studies/1b_objective_screen/meap-span-015-s5.yaml

# Validate the complete ablation matrix. This writes nothing.
/usr/bin/python3.11 mask_exp.py plan-collection collections/1b_objective_screen.yaml

# Freeze a condition into an immutable, timestamped record without submitting it.
/usr/bin/python3.11 mask_exp.py render studies/1b_objective_screen/meap-span-015-s5.yaml

# Recommended: submit the main stage first.
/usr/bin/python3.11 mask_exp.py submit-main studies/1b_objective_screen/meap-span-015-s5.yaml
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

Future training runs log instantaneous throughput and total theoretical work to
W&B. The cumulative counter is recorded as `train/cumulative_flops` at the
normal metrics interval and survives resumptions through checkpoint state.
Persistent checkpoint events also append cumulative FLOPs, tokens, and job and
cumulative throughput to `progress.txt` in the checkpoint directory.

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

## Checkpoint evaluation with lm-eval

Evaluation definitions live in `evaluations/suites.yaml`. The experiment
manager delegates execution to Spellbook, which uses lm-eval's native
`megatron_lm` backend to load distributed Megatron checkpoints directly. No
Hugging Face checkpoint conversion is required. The local lm-eval checkout and
its expected commit are pinned in the suite configuration so prompts, datasets,
metrics, and backend behavior remain reproducible without requiring outbound
GitHub access from compute nodes.

The pinned local checkout's declared dependencies are installed from
`evaluations/lm-eval-runtime-requirements.txt` once per evaluation node, so the
runtime remains complete when the selected EDF image changes. The job imports
the patched checkout directly through `PYTHONPATH` rather than rebuilding it.

Training and evaluation resolve Hugging Face repository IDs through the shared
`HF_HOME` on IOPS scratch. Online resolution stays enabled so a missing or newly
requested asset is downloaded again after automatic scratch cleanup.

Each evaluation targets an immutable training run record, a stage, and either
an explicit checkpoint iteration or the stage's latest checkpoint:

```bash
# Limited integration check. The explicit step can be planned before it exists.
/usr/bin/python3.11 mask_exp.py plan-eval RUN_DIRECTORY \
  --stage main --step 10300 --suite smoke --skip-checkpoint-check

# Freeze scripts and metadata without submitting.
/usr/bin/python3.11 mask_exp.py render-eval RUN_DIRECTORY \
  --stage main --step latest --suite core --suite math

# Submit multiple independent suite jobs for the same checkpoint.
/usr/bin/python3.11 mask_exp.py submit-eval RUN_DIRECTORY \
  --stage main --step latest --suite core --suite math

# Serialize evaluations that share a W&B run, and optionally override walltime.
/usr/bin/python3.11 mask_exp.py submit-eval RUN_DIRECTORY \
  --stage main --step 19074 --suite core --dependency PREVIOUS_EVAL_JOB_ID
/usr/bin/python3.11 mask_exp.py submit-eval RUN_DIRECTORY \
  --stage cooldown-from-0034333 --step latest --suite paloma \
  --sbatch-time 01:45:00
```

Training submissions accept explicit scheduler-only overrides without
changing the scientific configuration. `submit-main` accepts `--nodes` and
`--sbatch-time`; frozen runs accept `--sbatch-time` through `resume` and
`submit-cooldowns` when their recorded scheduler settings become stale:

```bash
/usr/bin/python3.11 mask_exp.py resume RUN_DIRECTORY --stage main \
  --without-reservation --sbatch-time 01:15:00

/usr/bin/python3.11 mask_exp.py submit-main CONDITION.yaml \
  --nodes 4 --sbatch-time 04:00:00
```

The exact overridden command is appended to the run's submission history.

Available suites are deliberately cost-tiered:

- `smoke`: 20 examples each from HellaSwag, ARC-Easy, and GSM8K. These limited
  results diagnose integration only and must not be reported as model results.
- `core`: HellaSwag, PIQA, Winogrande, ARC-Easy, ARC-Challenge, OpenBookQA, and
  BoolQ using relatively inexpensive likelihood/multiple-choice evaluation.
- `math`: the complete deterministic GSM8K evaluation for primary checkpoints.
- `code`: complete MBPP and HumanEval pass@1 evaluation. This executes generated
  Python and is rejected unless `--allow-unsafe-code` is passed.
- `paloma`: specialized document-level language-model-fit evaluation over the
  Paloma JSONL corpus. It uses the evaluated checkpoint's tokenizer, preserves
  document boundaries, and reports per-domain perplexity and bits-per-byte.
  The configured dataset path is
  `/iopsstor/scratch/cscs/smehra/eval_datasets/paloma`.

Code evaluation is not made safe merely by the opt-in flag. Run it only in an
appropriately isolated environment with no valuable credentials or writable
data exposed. The flag records explicit acknowledgement and enables lm-eval's
unsafe task guard; it is not a security sandbox.

Spellbook caches constructed evaluation requests across checkpoints but never
shares cached model responses. Full JSON results are written below the durable
evaluation root configured in `evaluations/suites.yaml`. Each training
condition/stage/suite receives a separate evaluation run in the shared
`mask_pretraining` W&B project, grouped by study and tagged with condition,
stage, and suite. Checkpoint iteration is used as the W&B logging step.
Reportable lm-eval suites save per-example samples but leave `write_out`
disabled: sample files contain stable document/task identities, task-definition
fingerprints, gold targets, choice scores, metrics, and filtered generations;
`write_out` only prints example prompts. Request caches contain constructed
requests, not model responses, and can therefore accelerate inference-only
sample-logging reruns without mixing checkpoint outputs.

After two or more matching evaluations have sample files, produce paired JSON
and Markdown reports with a fixed A-minus-B direction:

```bash
/usr/bin/python3.11 -m experiment_manager.paired_eval \
  --condition NTP=/path/to/ntp/core/step_38148 \
  --condition MTP=/path/to/mtp/core/step_38148 \
  --condition random_mask=/path/to/random/core/step_38148 \
  --seed 12345 --resamples 10000 \
  --json paired-core.json --markdown paired-core.md
```

The loader pairs by task, document ID, and document hash—not file order—and
rejects task/configuration differences, partial samples, and duplicate IDs.
Intervals quantify evaluation-example uncertainty for fixed checkpoints, not
variation across pretraining seeds.

The bootstrap uses bounded-memory NumPy batches when NumPy is installed and
records the selected backend in the JSON. To keep the analysis off a login
node, add `--submit-slurm`; this submits a CPU-only job that re-runs the same
command without recursively submitting another job:

```bash
/usr/bin/python3.11 -m experiment_manager.paired_eval \
  --condition NTP=/path/to/ntp/core/step_38148 \
  --condition MTP=/path/to/mtp/core/step_38148 \
  --seed 12345 --resamples 10000 \
  --json paired-core.json --markdown paired-core.md \
  --submit-slurm --slurm-account infra01 \
  --slurm-time 00:30:00 --slurm-mem 4G
```

No GPU or `--gres` request is made. Optional `--slurm-partition`,
`--slurm-cpus`, and `--slurm-log` flags override the small CPU-job defaults.
The job uses the configured `test-env` Slurm container so NumPy is available;
override it with `--slurm-environment` or pass an empty value to disable it.

For the initial scaling screen, run `core` at intermediate persistent
checkpoints and run the complete `math` and `code` suites only for the primary
final checkpoints. This controls generation cost without using truncated
datasets for reported results.

Paloma is not an lm-evaluation-harness task collection: its evaluator is a
document-level causal-language-model pass implemented in Spellbook. The
published Paloma corpus is already stratified to approximately 100k tokens per
domain; current experiments evaluate it without decontamination and therefore
should be interpreted as within-study diagnostics.

### Speculative natural-continuation workload

`evaluations/prepare_speculative_natural_v1.sh` deterministically builds a 1,000-prompt workload: 200
sequences from Megatron's DCLM-Edu validation partition and 200 token windows from each of four Paloma
groups (Wikipedia, academic prose, PTB news, and selected general-interest subreddits). Each prompt has
256 tokens and reserves at least 128 further source tokens. Manifests freeze inputs, extraction parameters,
source IDs/token offsets, and SHA-256 fingerprints.

Run the preparation script inside the standard `test-env`, where NumPy and the recorded tokenizer runtime
are available:

```bash
cd /users/smehra/developer/mask-experiment-manager
srun --environment=test-env bash evaluations/prepare_speculative_natural_v1.sh
```

The DCLM extractor exactly reproduces training's `--split 995,5,0`: for every sorted indexed shard it uses
sequence indices from `round(0.995 * sequence_count)` onward. These sequences received no training-gradient
updates, although they were used for periodic validation, so reports label them `dclm_edu_validation` rather
than an untouched test set. Paloma samples use deterministic non-overlapping token windows because several
Paloma domain files store one long concatenated text record.

After preparation, plan the suite without writing or submitting:

```bash
/usr/bin/python3.11 mask_exp.py plan-speculative RUN_DIRECTORY \
  --stage main --step latest --suite speculative-natural-v1
```

The suite keeps overall and per-domain aggregates. Its six-hour request reflects the deliberately sequential,
correctness-first verifier over 1,000 prompts and is not a production-speed benchmark. Results are appended
durably after every completed prompt/profile unit, while aggregate JSON and Markdown reports are refreshed
atomically at regular intervals. Render the immutable record once, then use the same idempotent submission
command both for its first run and for any manual restart:

```bash
/usr/bin/python3.11 mask_exp.py render-speculative RUN_DIRECTORY \
  --stage main --step latest --suite speculative-natural-v1

/usr/bin/python3.11 mask_exp.py submit-speculative-record ANALYSIS_RECORD \
  --sbatch-time 02:00:00 --sbatch-partition preemptable
```

The rendered script requests Slurm requeue, so a preempted allocation can restart the same job automatically.
Re-running the exact `submit-speculative-record` command is also safe after a failure or wall-time expiry:
completed result keys are skipped and are not double-counted. `resume-speculative` remains an alias for older
commands.

The per-record `progress_speculative_*.json` sidecar reports completed and expected prompt units and records.
The append-only `samples_speculative_*.jsonl` file is the source of truth used to rebuild partial or final
reports after interruption.

## Defining experiments

The 300M and original 1.1B screens contain NTP, native Megatron two-token MTP
(one sequential MTP layer), a three-token MTP variant that replays one shared
MTP layer to predict both the second and third next tokens, 15% random MEAP,
15% span-5 MEAP, and variable-span MEAP with maximum length five:

```bash
/usr/bin/python3.11 mask_exp.py plan-collection collections/300m_objective_screen.yaml
/usr/bin/python3.11 mask_exp.py plan-collection collections/1b_objective_screen.yaml
```

The follow-up `1b-objective-screen-2` study replicates the same 1.15B recipe
with seed 67. Its collection contains NTP, two-token MTP, and the selected
variable-span masking condition using the same truncated-geometric 1–5 span
distribution as the first screen:

```bash
/usr/bin/python3.11 mask_exp.py plan-collection collections/1b_objective_screen_2.yaml
```

Before launching the 300M objective screen, calibrate its Muon peak learning
rate and spectral scale with the short, unmasked sweep:

```bash
/usr/bin/python3.11 mask_exp.py plan-collection collections/300m_muon_sweep.yaml
```

The sweep matches the production 300M architecture, data, batch size, sequence
length, and regularization for 1.5B tokens. It brackets the selected 1B setting
and the provisional 300M baseline while varying one optimizer dimension at a
time around the transferred center.

Copy a condition file and change `name`, `description`, and `overrides`:

```yaml
schema_version: 1
recipe: ../../recipes/1b_llama.yaml
study: 1b-objective-screen
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

`INPUT_MASK_STRATEGY: variable_span` interprets `INPUT_MASK_SPAN_LENGTH` as the
positive truncation maximum. It samples complete span lengths from a truncated
geometric distribution with fixed p=0.5. For example, when the maximum is five,
lengths 1–5 have normalized weights proportional to `[16, 8, 4, 2, 1]`. This
standard memoryless distribution
strongly favors easier short spans while retaining a diminishing tail of harder
spans, without hand-designed irregular weights or a probability sweep. Lengths
are renormalized over those that fit the remaining token budget and an eligible
run; placement is uniform over feasible starts. Thus masking never crosses an
ineligible/document boundary, terminates without rejection loops, and aims for
`floor(eligible_tokens * ratio)` masked tokens. Adjacent independently sampled
spans may merge in the final bitmap and therefore produce effective mask
offsets longer than five.

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

The shared launcher accepts only the frozen architectural values declared by a
model-family recipe. Recipes also control data, schedules, batch sizes, seed,
logging/evaluation cadence, regularization, masking, MTP, checkpointing, and
Slurm resources. The earlier `1B-meap-config.sh` and `1b_meap.yaml` are retained
for compatibility.

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
