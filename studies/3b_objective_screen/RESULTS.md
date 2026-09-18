# 3B objective-screen results (early intermediate checkpoints)

## Summary

At 36B tokens, 30% of the planned 120B-token budget, all three runs have a
healthy downstream trajectory. Zero-shot and five-shot core scores rise
substantially from 12B to 36B for NTP, variable-span masking, and MTP. There is
no sign of a training failure or stalled masking run.

There is, however, no positive masking signal yet. Variable-span masking trails
NTP on both aggregate protocols at all three checkpoints. At 36B, NTP is ahead
on every one of the seven core tasks in both zero-shot and five-shot evaluation,
and the aggregate gaps are distinguishable from evaluation-example noise.
Masking also has worse Paloma perplexity and bits/byte. The current evidence
therefore makes a final masking improvement less likely than it looked before
these evaluations, but it does not rule one out: these are correlated
checkpoints from one pretraining seed, the run is still early, and the existing
1.1B screen showed substantial rank movement late in training and during
cooldown.

The most defensible interim conclusions are:

- All objectives are learning normally and should continue to the planned
  main-stage and cooldown endpoints.
- Variable-span masking improves steadily, but slightly more slowly than NTP
  on the zero-shot aggregate. Its five-shot deficit narrows at 24B and widens
  again at 36B, so there is no favorable crossover trajectory yet.
- The 36B NTP-over-masking advantage is coherent across tasks and evaluation
  protocols, not an isolated benchmark fluctuation.
- MTP is mixed on core accuracy: it trails NTP zero-shot but leads five-shot at
  36B. It has the best 36B Paloma result. This is encouraging but not yet a
  stable general downstream lead.
- No objective should be selected as the final winner from these early,
  single-seed results.

## Experiment and scope

The screen compares matched approximately 3.07B-parameter models trained with:

1. Standard next-token prediction (NTP).
2. Variable-span masking of 15% of input tokens, with span lengths 1--5 drawn
   from the existing truncated-geometric distribution.
3. Native Megatron two-token MTP (one additional future-token prediction).

The evaluated main-stage checkpoints are:

| Iteration | Exact tokens | Short name | Fraction of 120B total | Fraction of 108B main |
|---:|---:|---:|---:|---:|
| 11,444 | 11,999,903,744 | 12B | 10.0% | 11.1% |
| 22,888 | 23,999,807,488 | 24B | 20.0% | 22.2% |
| 34,332 | 35,999,711,232 | 36B | 30.0% | 33.3% |

Each checkpoint was evaluated on zero-shot `core` and fixed-demonstration
five-shot `core`. The 36B checkpoints were also evaluated on Paloma. All 21
evaluation jobs completed successfully.

`Core` is the unweighted macro-average of the standard reportable accuracy
metric for HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA,
and BoolQ. Length-normalized accuracy is used where lm-eval defines it.

## Core trajectory

### Zero-shot

| Tokens | NTP | Variable span | MTP | Masking − NTP | Best |
|---:|---:|---:|---:|---:|---|
| 12B | **53.14%** | 51.93% | 52.67% | -1.21 points | NTP |
| 24B | 55.21% | 53.76% | **55.63%** | -1.45 points | MTP |
| 36B | **57.02%** | 55.46% | 56.33% | -1.56 points | NTP |

From 12B to 36B, NTP gains 3.88 points, variable-span masking gains 3.53,
and MTP gains 3.66. The masking model is clearly learning, but its deficit to
NTP grows modestly at each measured point rather than closing.

### Five-shot

| Tokens | NTP | Variable span | MTP | Masking − NTP | Best |
|---:|---:|---:|---:|---:|---|
| 12B | 54.43% | 53.40% | **54.94%** | -1.03 points | MTP |
| 24B | 57.49% | 56.78% | **58.74%** | -0.71 points | MTP |
| 36B | 59.34% | 58.03% | **59.68%** | -1.31 points | MTP |

The masking deficit temporarily narrows at 24B but does not continue to close
at 36B. MTP leads all three five-shot checkpoints, although its 36B lead over
NTP is only 0.33 points and is unresolved by evaluation-example uncertainty.

Checkpoint scores are correlated observations from the same training runs;
their consistency is useful trajectory evidence but does not turn three
checkpoints into three independent replications.

## Matched-example uncertainty at 36B

The lm-eval sample files match all 20,465 examples uniquely across conditions.
Intervals below use 1,000 deterministic paired bootstrap resamples (seed
12345) and condition on the fixed trained checkpoints.

| Protocol and comparison (A−B) | Paired core macro difference | 95% CI |
|---|---:|---:|
| Zero-shot NTP − variable span | +1.49 points | [+0.67, +2.32] |
| Zero-shot NTP − MTP | +0.64 points | [-0.19, +1.49] |
| Zero-shot variable span − MTP | -0.84 points | [-1.76, -0.02] |
| Five-shot NTP − variable span | +1.19 points | [+0.42, +1.92] |
| Five-shot NTP − MTP | -0.43 points | [-1.15, +0.28] |
| Five-shot variable span − MTP | -1.62 points | [-2.40, -0.85] |

The paired sample-derived estimates can differ slightly from subtraction of
the rounded aggregate result JSON, but they give the same ordering. Both core
protocols resolve an NTP advantage over variable-span masking for these fixed
36B checkpoints. They do **not** quantify pretraining-seed uncertainty, task
selection uncertainty, or the probability of retaining the ordering at 120B.

At the task level, the direction is unusually coherent:

| Task | NTP zero-shot | Variable span zero-shot | NTP−mask | NTP five-shot | Variable span five-shot | NTP−mask |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 59.04% | 58.57% | +0.47 | 60.62% | 59.76% | +0.86 |
| PIQA | 73.45% | 72.74% | +0.71 | 74.21% | 72.69% | +1.52 |
| WinoGrande | 58.56% | 58.33% | +0.23 | 60.93% | 59.75% | +1.18 |
| ARC-Easy | 67.89% | 66.29% | +1.60 | 73.23% | 71.17% | +2.06 |
| ARC-Challenge | 38.31% | 37.03% | +1.28 | 40.44% | 39.68% | +0.76 |
| OpenBookQA | 38.00% | 36.20% | +1.80 | 42.00% | 39.80% | +2.20 |
| BoolQ | 63.85% | 59.05% | +4.80 | 63.98% | 63.39% | +0.59 |

NTP wins all 14 task/protocol cells. In the paired analysis, the clearest
zero-shot task differences are BoolQ (+4.80 points, 95% CI [+3.06, +6.64])
and ARC-Easy (+1.60, [+0.08, +3.16]); five-shot support is clearest on
HellaSwag (+0.86, [+0.26, +1.48]) and ARC-Easy (+2.06, [+0.63, +3.58]). Most
other individual-task intervals still include zero. The aggregate conclusion
comes from small, consistently directed differences rather than every task
being individually resolved.

## Paloma at 36B

Paloma evaluates clean causal language-model fit over 68.8M test tokens and
571 domains. Lower is better.

| Objective | Perplexity | Bits/byte | Macro-domain PPL |
|---|---:|---:|---:|
| NTP | 10.199 | 0.9250 | 13.875 |
| Variable-span masking | 10.500 | 0.9366 | 14.365 |
| MTP | **10.065** | **0.9197** | **13.728** |

Variable-span masking is 3.0% worse than NTP in token-weighted perplexity,
1.3% worse in bits/byte, and 3.5% worse in macro-domain perplexity. This agrees
with the downstream direction rather than revealing a hidden masking benefit.
It is also qualitatively consistent with earlier screens: corrupting training
inputs can impose a clean next-token-modeling cost. MTP is modestly better than
NTP on all three Paloma summaries at this checkpoint.

## Interpretation of the masking trajectory

The masking trajectory is operationally healthy but scientifically
unfavorable so far. It improves by 3.53 zero-shot points and 4.63 five-shot
points from 12B to 36B, so there is no evidence of divergence, optimization
failure, or an inability to learn. The issue is relative efficiency: NTP is
ahead at every matched aggregate checkpoint and has better clean-language
model fit at 36B.

The consistent 36B task direction and paired intervals are enough to say that
the current gap is meaningful for these evaluated checkpoints. They are not
enough to say that a masking improvement at the end is impossible. Only one
seed is represented, 70% of the total token budget remains, and the cooldown
has not begun. The 1.1B screen also demonstrated that objective rankings can
move late. Still, a masking win now requires a later crossover; the observed
trajectory provides no evidence for that crossover yet. A large or robust
final masking improvement should therefore be considered less likely than a
tie or an NTP advantage on the present evidence.

The next most informative checkpoints are 48B and 72B, followed by the 108B
main endpoint and 120B cooldown endpoint. A shrinking NTP-minus-masking gap
across both core protocols, accompanied by recovery on Paloma, would be the
first persuasive sign of a masking crossover. If the deficit persists or
widens through 72B, the case for a final masking advantage would become
substantially weaker. Final decisions should include the cooldown checkpoint
and preferably another pretraining seed.

## Reproducibility and limitations

- Training seed: 3407 for all three conditions.
- lm-eval version: v0.4.12 at commit
  `6d642546f4688648fced259eb3302efd36ece5af` with the repository's recorded
  Megatron inference/sample-identity patch.
- Core evaluations use the complete task test sets and logged samples; no
  evaluation limit was used.
- Five-shot demonstrations are fixed and deterministically sampled, but this
  analysis does not include demonstration-selection uncertainty.
- Paired intervals use 1,000 resamples rather than the 10,000 used for mature
  final reports. This is sufficient for the interim direction and interval
  scale; final reporting should rerun the conventional 10,000-resample
  analysis.
- Paloma is available only at 36B in this interim set, so it does not yet
  provide a checkpoint trajectory.
- These results cover only 30% of training and no cooldown. This document
  should be updated rather than treated as a final study result.
