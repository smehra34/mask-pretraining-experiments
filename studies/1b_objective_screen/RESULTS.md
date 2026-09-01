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
- Span masking remains competitive despite substantially higher validation
  loss, demonstrating that validation loss is not directly comparable across
  these objectives and is not a reliable method-selection criterion here.
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

The loss comparison also has an important structural caveat. NTP and MTP see
the normal token context, whereas the masking objectives predict with part of
their input context deliberately corrupted. Span masking removes contiguous
context and is consequently an especially difficult token-prediction problem.
Its systematically higher validation loss is therefore not an apples-to-apples
measure of representation or downstream model quality. This explains why span
masking can have the worst validation loss while matching or exceeding the
other methods on downstream evaluations.

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
performance is competitive despite its worse objective loss. Follow-up work
on masking should evaluate additional seeds and should avoid using raw
validation loss to compare masked and unmasked objectives directly.

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
