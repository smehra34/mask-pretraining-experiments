# 300M objective-screen results

## Summary

The 300M objective screen does not identify a clearly superior training
objective. NTP, MTP, random masking, and span masking finish within 0.41
percentage points on the seven-task `core` macro-average, and their averages
across all five checkpoints fall within a range of only 0.20 points. The
ranking changes repeatedly over training, and no objective wins more than two
of the five checkpoints.

The scientifically defensible conclusions are:

- The four objectives are effectively tied at this model and data scale under
  the present single-seed design.
- The cooldown is a much clearer result than the objective comparison: all four
  methods improve from the main-final checkpoint to the cooldown-final
  checkpoint.
- MTP does not show a consistent downstream advantage at 300M. Its small final
  advantage over NTP is only 0.08 points and is not meaningful.
- Masking is not detectably beneficial or harmful overall. Random masking has
  the highest final macro-average, while span masking leads immediately before
  cooldown, but neither result is stable across checkpoints.
- Raw validation loss should not be used to rank masked against unmasked
  objectives because masking deliberately makes the validation prediction
  problem harder.

## Experiment and metrics

The screen compares four objectives under matched 300M-model training
conditions:

1. Standard next-token prediction (NTP).
2. Native Megatron two-token multi-token prediction (MTP).
3. Independent random masking of 15% of input tokens.
4. Span masking of 15% of input tokens with nominal span length 5.

The main checkpoints are at iterations 2,861, 5,722, 8,583, and 10,300. A
cooldown from iteration 10,300 produces the final checkpoint at iteration
11,445.

The `core` score is an unweighted macro-average of the standard accuracy metric
for HellaSwag, PIQA, Winogrande, ARC-Easy, ARC-Challenge, OpenBookQA, and BoolQ.
No GSM8K evaluation was run because open-ended mathematical performance is
expected to be near zero at this scale and would add little resolving power.

## Final results

| Objective | Core macro average | Final validation loss |
|---|---:|---:|
| NTP | 46.68% | 2.577 |
| MTP | 46.76% | **2.574** |
| Random masking | **47.09%** | 2.607 |
| Span masking | 46.70% | 2.617 |

The complete final spread is 0.41 percentage points. Random masking is
nominally first, but this difference is too small and inconsistent to support a
claim of superiority. MTP, span masking, and NTP are separated by only 0.08
points.

Final task scores are:

| Objective | HellaSwag | PIQA | WinoGrande | ARC-E | ARC-C | OpenBookQA | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| NTP | 40.48 | 66.87 | 51.46 | 52.61 | 26.54 | 31.40 | 57.37 |
| MTP | **40.57** | 66.38 | 51.46 | 51.56 | **27.73** | 32.00 | 57.61 |
| Random masking | 39.05 | 66.05 | 50.67 | **53.45** | 26.79 | **33.40** | **60.21** |
| Span masking | 38.94 | **67.30** | **51.85** | 52.82 | 27.65 | 31.40 | 56.91 |

Different objectives lead different tasks, with no coherent pattern. Several
of the apparently largest differences occur on the smaller or less stable
benchmarks. For example, random masking's macro lead is helped by BoolQ and
OpenBookQA, but it does not lead HellaSwag, PIQA, WinoGrande, or ARC-Challenge.

## Checkpoint trajectory

| Iteration | NTP | MTP | Random masking | Span masking | Best core macro |
|---:|---:|---:|---:|---:|---|
| 2,861 | 42.65% | **43.55%** | 41.81% | 42.69% | MTP |
| 5,722 | **43.36%** | 42.79% | 42.78% | 42.39% | NTP |
| 8,583 | **44.61%** | 44.42% | 44.55% | 44.15% | NTP |
| 10,300 | 45.72% | 45.51% | 45.77% | **46.18%** | Span masking |
| 11,445 | 46.68% | 46.76% | **47.09%** | 46.70% | Random masking |

The winner changes from MTP to NTP, then span masking, and finally random
masking. Mean core scores over the five checkpoints are:

| Objective | Mean core macro | Mean checkpoint rank |
|---|---:|---:|
| NTP | 44.60% | 2.4 |
| MTP | 44.60% | 2.4 |
| Random masking | 44.40% | 2.4 |
| Span masking | 44.42% | 2.8 |

These checkpoint means are not independent repeated measurements, because all
checkpoints for an objective come from the same run. They are nevertheless
useful evidence that none of the objectives maintains an advantage through
training.

## Cooldown effect

Every objective improves from iteration 10,300 to 11,445:

| Objective | Main final | Cooldown final | Change |
|---|---:|---:|---:|
| NTP | 45.72% | 46.68% | +0.96 points |
| MTP | 45.51% | 46.76% | +1.25 points |
| Random masking | 45.77% | 47.09% | +1.32 points |
| Span masking | 46.18% | 46.70% | +0.52 points |

This directionally consistent gain is the clearest result in the screen. It
supports retaining the cooldown schedule in subsequent experiments. The exact
size should still be interpreted cautiously because there is no matched
no-cooldown control continued to the same token budget.

## Relationship to validation loss

When all objectives and checkpoints are pooled, validation loss is strongly
inversely correlated with the core macro-average:

- Pearson correlation: approximately -0.87.
- Spearman rank correlation: approximately -0.92.

This mostly measures training progress: later checkpoints have lower loss and
higher downstream accuracy. It does not imply that validation loss reliably
selects an objective. Within a fixed checkpoint the association changes over
training, and at the main-final checkpoint it is reversed: the masking methods
have higher loss but the two best downstream macro-averages.

NTP and MTP are directly comparable and have nearly identical final losses
(2.577 and 2.574), matching their nearly identical downstream results. The
masking objectives are not directly comparable to those losses because their
validation inputs are deliberately corrupted. Span masking removes contiguous
context and therefore creates the hardest token-prediction problem. Its higher
loss does not indicate a proportionally worse downstream model.

## Scientific interpretation

At 300M, the evaluation suite has enough sensitivity to show improvement with
training and cooldown, but not enough evidence to resolve the small objective
effects. This is partly statistical noise and partly a real indication that
the methods have similar average performance. Many tasks remain close to weak
baseline performance at this scale, and task-specific fluctuations can move a
simple seven-task macro-average by more than the differences between methods.

The absence of a clear winner is itself useful: none of MTP or the two masking
objectives causes a large general-capability regression relative to NTP. It
also means the screen does not justify choosing a method on downstream accuracy
alone. Selection should instead depend on the primary scientific question,
compute and throughput costs, or stronger evidence from the 1.1B screen.

The 1.1B screen showed a modest final aggregate signal for MTP, whereas that
advantage is absent at 300M. A cautious interpretation is that any MTP benefit
is small, noisy, or scale-dependent. Establishing an objective effect would
require additional seeds and preferably a prespecified aggregate or a smaller
set of sufficiently sensitive tasks. Based on the current data, claims should
be limited to equivalence within the resolution of this single-seed screen and
the consistent benefit observed after cooldown.

## Paloma evaluation

Final cooldown checkpoints were evaluated on the Paloma test split (68.8M
tokens across 571 domains) with the training tokenizer. Lower is better.

| Objective | Perplexity | Bits/byte | Macro-domain PPL |
|---|---:|---:|---:|
| NTP | 13.272 | 1.0299 | 17.551 |
| MTP | **13.265** | **1.0297** | 17.925 |
| Random masking | 13.704 | 1.0427 | 18.146 |
| Span masking | 14.248 | 1.0581 | 19.305 |

MTP and NTP are effectively tied on token-weighted Paloma metrics. Random
masking is about 3.3% worse than NTP in perplexity, while span masking is about
7.4% worse. MTP improves perplexity on 364/571 domains, but losses on some
high-loss domains offset those gains in the aggregate. Objective effects are
therefore small at this scale.
