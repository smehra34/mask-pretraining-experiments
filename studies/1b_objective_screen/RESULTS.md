# 1.1B objective-screen results

## Summary

The four training objectives produce broadly similar downstream performance.
There is no consistently dominant method across checkpoints or tasks. MTP has
the strongest overall result at the final checkpoint, but its advantage is
modest and should not be interpreted as a definitive objective improvement
from this single-seed screen.

The most defensible conclusion is:

- MTP shows a small positive aggregate signal and is the best single choice if
  one objective must be selected from this screen.
- Span masking remains competitive on the downstream macro-average despite
  substantially worse clean-validation loss. Because validation inputs are
  uncorrupted for every objective, this indicates a real tradeoff between
  clean next-token modeling and the measured downstream tasks.
- Random masking is broadly neutral: it performs within the same downstream
  regime without a consistent advantage.
- Additional seeds would be required to establish small differences between
  these objectives reliably.

## Experiment

The screen compares four objectives under matched 1.1B-model training
conditions:

1. Standard next-token prediction (NTP).
2. Native Megatron two-token multi-token prediction (MTP).
3. Independent random masking of 15% of input tokens.
4. Span masking of 15% of input tokens with nominal span length 5.

The main-training checkpoints are at iterations 9,537, 19,074, 28,611, and
34,333. A cooldown from iteration 34,333 produces the final checkpoint at
iteration 38,148.

The `core` score below is an unweighted macro-average of the standard accuracy
metric for HellaSwag, PIQA, Winogrande, ARC-Easy, ARC-Challenge, OpenBookQA, and
BoolQ. GSM8K is reported separately using flexible-extraction exact match.

## Final results

| Objective | Core macro average | GSM8K flexible exact match | Final validation loss |
|---|---:|---:|---:|
| NTP | 52.53% | **2.05%** | 2.311 |
| MTP | **54.17%** | 1.82% | **2.292** |
| Random masking | 53.16% | 1.97% | 2.321 |
| Span masking | 52.98% | 1.90% | 2.383 |

MTP finishes approximately 1.0--1.6 percentage points above the other methods
on the core macro-average. It has the best final result on four of the seven
core tasks, while span masking has the best result on the other three. Most
individual task differences are within, or only slightly beyond, their
sampling uncertainty.

All GSM8K results are approximately 1.8--2.1%, with a standard error around
0.4 percentage points. They should therefore be treated as statistically
indistinguishable. Random masking has a lower strict-match score, but the
difference disappears under flexible extraction, suggesting an answer-format
difference rather than weaker mathematical ability.

## Paired evaluation-example uncertainty

The final-checkpoint core evaluations were rerun with sample logging and
matched by task name, document ID, document hash, task-configuration hash, and
dataset fingerprint. All 20,465 examples match uniquely for every condition.
The table uses the stated first-minus-second direction and 10,000 deterministic
paired bootstrap resamples (seed 12345). Accuracy is the task's reportable
metric: length-normalized accuracy where lm-eval defines it and raw accuracy
otherwise.

| Comparison | Core macro difference | Paired-bootstrap 95% CI |
|---|---:|---:|
| NTP − MTP | -1.64 points | [-2.43, -0.85] |
| NTP − random masking | -0.62 points | [-1.45, +0.19] |
| NTP − span masking | -0.45 points | [-1.27, +0.34] |
| MTP − random masking | +1.02 points | [+0.21, +1.82] |
| MTP − span masking | +1.19 points | [+0.37, +2.01] |
| Random masking − span masking | +0.17 points | [-0.61, +0.94] |

MTP's advantage over each of the other three objectives is distinguishable
from evaluation-example noise for these fixed checkpoints. NTP and both masking
methods remain mutually unresolved. For the primary MTP-versus-NTP comparison,
the task-level results are:

| Task | MTP − NTP accuracy (95% CI) | Only MTP/NTP correct | Exact McNemar p | Raw margin difference (95% CI) |
|---|---:|---:|---:|---:|
| ARC-Challenge | +2.47 [+0.26, +4.78] | 105/76 | 0.037 | +0.243 [+0.110, +0.374] |
| ARC-Easy | +2.48 [+0.97, +4.00] | 201/142 | 0.0017 | +0.150 [+0.069, +0.230] |
| BoolQ | +4.16 [+2.36, +5.99] | 531/395 | 8.8e-6 | +0.075 [+0.052, +0.098] |
| HellaSwag | +1.38 [+0.80, +1.96] | 515/376 | 3.6e-6 | +0.575 [+0.484, +0.668] |
| OpenBookQA | -0.40 [-3.20, +2.60] | 27/29 | 0.89 | +0.259 [+0.010, +0.518] |
| PIQA | +1.14 [-0.33, +2.61] | 101/80 | 0.14 | +0.124 [-0.008, +0.257] |
| WinoGrande | +0.24 [-3.08, +3.55] | 222/219 | 0.92 | +0.080 [+0.007, +0.154] |

The score margins are directionally favorable to MTP on every task, and their
intervals exclude zero on five of seven tasks. The accuracy advantage is most
clearly driven by HellaSwag and BoolQ, with additional support from the two ARC
tasks. MTP also exceeds random and span masking on the macro, driven especially
by HellaSwag and BoolQ; task-specific exceptions and wide intervals remain.
Raw margins are not aggregated across tasks because their scales differ;
normalized margins and option-set probability, log-loss, and Brier comparisons
are retained in `paired-core.json`.

This is a concrete reduction in evaluation-example uncertainty, but not proof
of a general training-objective effect. The analysis includes six model pairs
and seven exploratory task comparisons, treats the benchmark suite as fixed,
and conditions on one trained checkpoint per objective. It does not measure
pretraining-seed variance, which remains the principal limitation on stronger
scientific claims.

## Checkpoint trajectory

| Iteration | NTP | MTP | Random masking | Span masking | Best core macro |
|---:|---:|---:|---:|---:|---|
| 9,537 | 49.82% | **50.07%** | 49.23% | 49.26% | MTP |
| 19,074 | **50.28%** | 49.66% | 50.08% | 48.89% | NTP |
| 28,611 | 48.65% | 51.50% | 49.82% | **51.56%** | Span masking |
| 34,333 | 49.19% | 52.27% | 52.88% | **53.52%** | Span masking |
| 38,148 | 52.53% | **54.17%** | 53.16% | 52.98% | MTP |

The ordering is not stable. MTP leads at the first and final checkpoints,
whereas span masking leads at the two late pre-cooldown checkpoints. NTP's core
macro also falls at iterations 28,611 and 34,333 before improving substantially
during cooldown. Some of this movement is driven by noisy individual tasks,
particularly BoolQ, so small macro-average differences should not be
overinterpreted.

Across all five checkpoints, the mean core macro-averages are 50.09% for NTP,
51.53% for MTP, 51.03% for random masking, and 51.24% for span masking. These
means summarize the trajectory but are not independent repeated measurements,
because checkpoints from the same training run are highly correlated.

## Relationship to validation loss

Pooling all objectives and checkpoints gives a moderate inverse relationship
between validation loss and the core macro-average:

- Pearson correlation: approximately -0.65.
- Spearman rank correlation: approximately -0.59.

This pooled correlation is substantially driven by training progress: later
checkpoints generally have both lower loss and better downstream performance.
Within an individual checkpoint, the relationship is unstable and even changes
sign. Validation loss is therefore useful for tracking optimization within a
run, but weak evidence for selecting between these objectives.

Validation examples are not corrupted: input masking is applied only to the
training split. The loss comparison is therefore apples-to-apples as a measure
of clean next-token prediction. Span masking's systematically higher
validation loss is evidence that this training objective produces a worse
clean causal language model at the present scale and budget, even though it
can match or exceed the other methods on some downstream evaluations. The two
signals measure different capabilities and should both be reported rather
than explaining away the clean-loss regression as evaluation corruption.

MTP provides the cleanest alignment between the two signals at the final
checkpoint: it has both the lowest validation loss and the highest core macro.
That agreement supports a modest positive interpretation of MTP, but the
task-level and checkpoint-level variability prevents a stronger claim.

## Recommended interpretation

For subsequent experiments that require one fixed objective, MTP is a
scientifically defensible default based on this screen. The evidence supports
describing it as a modest aggregate improvement, not as a uniformly superior
method. If simplicity or direct comparability with standard language-model
training is more important, NTP also remains defensible because the absolute
differences are small.

Span masking is the most interesting non-trivial result: its downstream
performance is competitive despite worse clean next-token loss. Follow-up work
should evaluate additional seeds and explicitly test whether the harder masked
objective's capabilities compensate for this clean-language-model tradeoff.
Raw validation loss is directly comparable across objectives, but it should
not be treated as the sole measure of masked prediction or downstream utility.

## Paloma evaluation

Final cooldown checkpoints were evaluated on the Paloma test split (68.8M
tokens across 571 domains) with the training tokenizer. Lower is better.

| Objective | Perplexity | Bits/byte | Macro-domain PPL |
|---|---:|---:|---:|
| NTP | 10.636 | 0.9417 | 14.997 |
| MTP | **10.343** | **0.9306** | 14.244 |
| Random masking | 10.540 | 0.9381 | **14.024** |
| Span masking | 11.437 | 0.9706 | 15.209 |

MTP improves perplexity by about 2.8% relative to NTP and wins on 562/571
domains. Random masking is modestly better than NTP in aggregate and has the
best macro-domain score, suggesting a possible scale-dependent benefit. Span
masking remains clearly worse.

## Scale comparison

MTP is effectively neutral at 300M but clearly beneficial at 1.1B. Random
masking changes from a clear regression at 300M to a small aggregate improvement
at 1.1B, while span masking is harmful at both scales. This motivates testing
3B, while keeping claims provisional until downstream evaluations and additional
seeds are available.
