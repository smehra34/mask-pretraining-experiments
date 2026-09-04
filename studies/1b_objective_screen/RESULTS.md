# 1.1B objective-screen results

## Summary

The five training objectives produce broadly similar downstream performance.
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
- Variable-span masking is the strongest masking variant overall. It nearly
  matches random masking at the cooled endpoint, has the second-best average
  checkpoint trajectory, and avoids fixed-span masking's large clean-loss and
  Paloma regressions. Its remaining gap to MTP is modest but detectable in the
  paired final-core evaluation.
- Additional seeds would be required to establish small differences between
  these objectives reliably.

## Experiment

The screen compares five objectives under matched 1.1B-model training
conditions:

1. Standard next-token prediction (NTP).
2. Native Megatron two-token multi-token prediction (MTP).
3. Independent random masking of 15% of input tokens.
4. Span masking of 15% of input tokens with nominal span length 5.
5. Variable-span masking of 15% of input tokens, with span lengths 1--5 drawn
   from a truncated geometric distribution with p=0.5.

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
| Variable-span masking | 53.16% | **2.12%** | 2.321 |

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
Variable-span masking has the highest numerical flexible-extraction score, but
the 0.07-point lead over NTP is negligible relative to sampling uncertainty.

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
| NTP − variable-span masking | -0.63 points | [-1.43, +0.19] |
| MTP − random masking | +1.02 points | [+0.21, +1.82] |
| MTP − span masking | +1.19 points | [+0.37, +2.01] |
| MTP − variable-span masking | +1.01 points | [+0.21, +1.81] |
| Random masking − span masking | +0.17 points | [-0.61, +0.94] |
| Random masking − variable-span masking | -0.00 points | [-0.74, +0.75] |
| Span masking − variable-span masking | -0.18 points | [-0.95, +0.62] |

MTP's advantage over each of the other four objectives is distinguishable
from evaluation-example noise for these fixed checkpoints. NTP and the three
masking methods remain mutually unresolved. Variable span and random masking
are an especially close final-core tie: their estimated difference is less
than 0.01 points with a [-0.74, +0.75] interval. For the primary
MTP-versus-NTP comparison,
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
tasks. MTP also exceeds all three masking variants on the macro. Against
variable span, its +1.01-point advantage is driven most clearly by HellaSwag
(+1.54 points) and ARC-Easy (+2.02 points); most other task-level accuracy
intervals remain unresolved. Variable span's near-zero aggregate difference
from fixed span hides a task tradeoff: it gains 4.28 points on BoolQ while
losing 3.16 on ARC-Easy. Task-specific exceptions and wide intervals remain.
Raw margins are not aggregated across tasks because their scales differ;
normalized margins and option-set probability, log-loss, and Brier comparisons
are retained in `paired-core.json`.

This is a concrete reduction in evaluation-example uncertainty, but not proof
of a general training-objective effect. The analysis includes ten model pairs
and seven exploratory task comparisons, treats the benchmark suite as fixed,
and conditions on one trained checkpoint per objective. It does not measure
pretraining-seed variance, which remains the principal limitation on stronger
scientific claims.

## Checkpoint trajectory

| Iteration | NTP | MTP | Random masking | Span masking | Variable span | Best core macro |
|---:|---:|---:|---:|---:|---:|---|
| 9,537 | 49.82% | **50.07%** | 49.23% | 49.26% | 49.27% | MTP |
| 19,074 | **50.28%** | 49.66% | 50.08% | 48.89% | 50.22% | NTP |
| 28,611 | 48.65% | 51.50% | 49.82% | 51.56% | **51.76%** | Variable span |
| 34,333 | 49.19% | 52.27% | 52.88% | **53.52%** | 52.82% | Span masking |
| 38,148 | 52.53% | **54.17%** | 53.16% | 52.98% | 53.16% | MTP |

The ordering is not stable. MTP leads at the first and final checkpoints,
variable span leads at iteration 28,611, and fixed span leads at 34,333. NTP's core
macro also falls at iterations 28,611 and 34,333 before improving substantially
during cooldown. Some of this movement is driven by noisy individual tasks,
particularly BoolQ, so small macro-average differences should not be
overinterpreted.

Across all five checkpoints, the mean core macro-averages are 50.09% for NTP,
51.53% for MTP, 51.03% for random masking, 51.24% for span masking, and 51.45%
for variable-span masking. Variable span is therefore second only to MTP on
this trajectory summary and is the only masking strategy to improve at every
measured checkpoint through the cooldown. These means are not independent
repeated measurements because checkpoints from the same run are highly
correlated.

## Relationship to validation loss

Pooling the original four objectives and checkpoints gives a moderate inverse relationship
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
of clean next-token prediction. Fixed-span masking's systematically higher
validation loss is evidence that this training objective produces a worse
clean causal language model at the present scale and budget, even though it
can match or exceed the other methods on some downstream evaluations.
Variable-span masking finishes at 2.321, essentially matching random masking
and substantially improving on fixed span's 2.383. Favoring shorter spans
appears to preserve the masking signal without the same clean-language-model
penalty, although this comparison does not isolate span distribution from
run-to-run noise. The two signals measure different capabilities and should
both be reported rather than explaining away the clean-loss regression as
evaluation corruption.

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

Variable-span masking is the most promising masking result. It retains the
competitive downstream behavior of masking while materially repairing fixed
span's clean-loss and Paloma regressions. It does not beat MTP at the final
core checkpoint, and its near-tie with random masking means this screen does
not establish that geometric spans are better than independent masks. A
second-seed comparison of MTP and the selected masking variant is the most
useful next test. Raw validation loss is directly comparable across
objectives, but it should not be treated as the sole measure of masked
prediction or downstream utility.

## Paloma evaluation

Final cooldown checkpoints were evaluated on the Paloma test split (68.8M
tokens across 571 domains) with the training tokenizer. Lower is better.

| Objective | Perplexity | Bits/byte | Macro-domain PPL |
|---|---:|---:|---:|
| NTP | 10.636 | 0.9417 | 14.997 |
| MTP | **10.343** | **0.9306** | 14.244 |
| Random masking | 10.540 | 0.9381 | **14.024** |
| Span masking | 11.437 | 0.9706 | 15.209 |
| Variable-span masking | 10.501 | 0.9366 | 14.068 |

MTP improves perplexity by about 2.8% relative to NTP and remains the clear
winner. With all five objectives included, it has the lowest perplexity on
510/571 domains. Variable span beats fixed span on 569/571 domains, NTP on
339/571, and random masking on 323/571. Its aggregate perplexity is slightly
better than random masking, while random retains the best macro-domain score;
this suggests a small difference in how their gains are distributed across
domains rather than a decisive overall advantage. Fixed-span masking remains
clearly worse.

## Scale comparison

MTP is effectively neutral at 300M but clearly beneficial at 1.1B. Random
masking changes from a clear regression at 300M to a small aggregate
improvement at 1.1B, while fixed-span masking is harmful at both scales. The
variable-span result shows that the fixed-span conclusion should not be
generalized to all contiguous masking: a short-span-heavy distribution largely
closes the clean-model gap. This motivates additional seeds and larger-scale
testing, while keeping claims provisional until those runs are available.
