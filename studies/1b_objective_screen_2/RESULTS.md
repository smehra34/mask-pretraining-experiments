# 1.1B objective-screen-2 results

## Status and scope

The seed-67 NTP, MTP, and variable-span main stages and their 4B-token
cooldowns completed successfully. Core results are complete at all four
intermediate main-stage checkpoints for NTP and MTP and at the final cooled-down
checkpoint (iteration 38,148) for all three objectives. Final five-shot core,
GSM8K, and Paloma evaluations are also complete for all three.

The comparison is otherwise matched to the original 1.1B objective screen:
the architecture, data, optimizer, token budget, cooldown, and evaluation
configuration are unchanged. Only the training seed changes from 3407 to 67.

## Final endpoint results

| Seed | Objective | Core macro | GSM8K flexible exact match | Validation loss | Paloma PPL | Paloma bits/byte | Paloma macro-domain PPL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3407 | NTP | 52.53% | 2.05% | 2.3110 | 10.636 | 0.9417 | 14.997 |
| 3407 | MTP | 54.17% | 1.82% | 2.2919 | 10.343 | 0.9306 | 14.244 |
| 67 | NTP | 54.38% | 1.82% | 2.2901 | 10.371 | 0.9316 | **14.070** |
| 67 | MTP | **54.41%** | 1.67% | **2.2886** | **10.311** | **0.9293** | 14.162 |
| 67 | Variable span | 52.68% | 2.12% | 2.3273 | 10.629 | 0.9414 | 14.371 |

At seed 67, MTP and NTP are tied on the core macro: MTP leads by only 0.03
percentage points. MTP remains directionally better on clean validation loss
and token-weighted Paloma metrics, but the differences are small: 0.0015 loss
and 0.58% Paloma perplexity. NTP has the better macro-domain Paloma score.
MTP has lower perplexity on 496 of 571 Paloma domains, compared with 562 of 571
at seed 3407, so the broad direction persists but is materially weaker.

GSM8K is approximately 1.7--2.1% for every seed/objective combination, with
standard errors around 0.35--0.40 percentage points. Seed-67 variable span
scores 2.12% flexible exact match (1.74% strict match, 1,319 examples), nearly
identical to the seed-3407 variable-span result. The small differences are not
actionable.

Variable span does not reproduce the favorable seed-3407 masking result at
seed 67. It trails seed-67 NTP by 1.70 core points and MTP by 1.73 points. Its
validation loss is worse by 0.0372 and 0.0387 respectively, while Paloma
perplexity is 2.49% above NTP and 3.09% above MTP. Compared with the seed-3407
variable-span run itself, its core macro is 0.48 points lower, validation loss
is 0.0063 higher, and Paloma perplexity is 1.22% higher. The variable-span
result is therefore weaker than both seed-67 controls across downstream core,
clean validation, and held-out language-modeling measurements.

## Five-shot core replication

The completed NTP and MTP checkpoints were also evaluated with five fixed,
deterministically sampled demonstrations per core task.

| Seed | Objective | Zero-shot core | Five-shot core | Five-shot minus zero-shot |
|---:|---|---:|---:|---:|
| 3407 | NTP | 52.53% | 56.47% | +3.94 |
| 3407 | MTP | 54.17% | 57.97% | +3.80 |
| 3407 | Variable span | 53.16% | 55.17% | +2.01 |
| 67 | NTP | 54.38% | 56.59% | +2.21 |
| 67 | MTP | **54.41%** | **57.92%** | +3.51 |
| 67 | Variable span | 52.68% | 55.92% | +3.25 |

This protocol produces a substantially more concordant MTP result. MTP leads
NTP by +1.50 points at seed 3407 and +1.33 points at seed 67. For seed 67, a
paired normal-approximation 95% interval over matched examples is
[+0.55, +2.10] points. The task-level MTP-minus-NTP effects at seed 67 are
-0.05 HellaSwag, -0.11 PIQA, +0.55 WinoGrande, +0.84 ARC-Easy, +1.88
ARC-Challenge, +3.60 OpenBookQA, and +2.57 BoolQ.

Five of seven task-effect signs agree across seeds under five-shot prompting,
versus three of seven under zero-shot, and the cross-seed task-effect Pearson
correlation changes from -0.48 to +0.48. In particular, both ARC tasks favor
MTP in aggregate across the two seeds, while the large zero-shot BoolQ reversal
no longer controls the conclusion. This is the clearest downstream evidence so
far for MTP, but it remains conditional on one fixed five-shot demonstration
sample. Repeating with other demonstration seeds or a prespecified prompt set
would be needed to separate robust in-context-learning behavior from exemplar
sensitivity.

Five-shot prompting does not rescue variable span relative to the controls.
At seed 67 it trails NTP by 0.67 points and MTP by 2.00 points. The paired
example bootstrap places NTP-minus-variable at +0.67 points with a 95% interval
of [-0.11, +1.46], while MTP-minus-variable is +2.00 points with an interval of
[+1.23, +2.76]. Variable span improves more from zero-shot to five-shot than
the seed-67 NTP model, but its final level remains lower. It also trails both
controls under five-shot at seed 3407, so this ordering agrees across the two
training seeds.

## Core checkpoint trajectory

The table uses the same unweighted seven-task core macro as the endpoint
analysis. Iterations 9,537 through 34,333 are main-stage checkpoints; iteration
38,148 is the cooled-down final checkpoint.

| Iteration | Seed 3407 NTP | Seed 3407 MTP | MTP−NTP | Seed 67 NTP | Seed 67 MTP | MTP−NTP |
|---:|---:|---:|---:|---:|---:|---:|
| 9,537 | 49.82% | 50.07% | +0.25 | 48.69% | 48.94% | +0.25 |
| 19,074 | 50.28% | 49.66% | -0.63 | 51.52% | 52.30% | +0.78 |
| 28,611 | 48.64% | 51.50% | +2.85 | 52.67% | 52.73% | +0.06 |
| 34,333 | 49.19% | 52.27% | +3.07 | 53.01% | 52.94% | -0.07 |
| 38,148 (cooled) | 52.53% | 54.17% | +1.64 | 54.38% | 54.41% | +0.03 |

The trajectory strengthens the non-replication result. Both seed-67 objectives
improve fairly smoothly after the first checkpoint and remain within 0.8 points
of each other throughout. By contrast, seed 3407 NTP falls at the middle and
late main-stage checkpoints while MTP continues improving, creating the large
late-stage MTP advantage that motivated the original conclusion. That divergence
does not recur at seed 67: the MTP advantage is +0.06 points at iteration 28,611,
-0.07 at 34,333, and +0.03 after cooldown.

The checkpoint-wise MTP-minus-NTP effect has the same sign in only three of five
comparisons across seeds and is strongly anticorrelated across the five aligned
checkpoints (Pearson -0.88). This estimate has only five points and should not be
overinterpreted, but it rules out the benign explanation that the seed-67 run
merely crossed over at an unlucky final checkpoint. Absolute trajectories are
more stable than objective deltas, especially for MTP (cross-seed Pearson 0.75;
NTP 0.37), although five checkpoints are too few for precise correlation
estimates.

## Core task comparison

| Task | Seed 3407 NTP | Seed 3407 MTP | MTP−NTP | Seed 67 NTP | Seed 67 MTP | MTP−NTP |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 54.65% | 56.03% | +1.38 | 56.26% | 56.69% | +0.43 |
| PIQA | 70.73% | 71.87% | +1.14 | 71.65% | 72.52% | +0.87 |
| WinoGrande | 56.83% | 57.06% | +0.24 | 58.72% | 58.09% | -0.63 |
| ARC-Easy | 62.33% | 64.81% | +2.48 | 63.43% | 65.03% | +1.60 |
| ARC-Challenge | 34.30% | 36.77% | +2.47 | 34.81% | 34.39% | -0.43 |
| OpenBookQA | 36.00% | 35.60% | -0.40 | 35.80% | 36.40% | +0.60 |
| BoolQ | 52.87% | 57.03% | +4.16 | 60.00% | 57.77% | -2.23 |
| **Core macro** | **52.53%** | **54.17%** | **+1.64** | **54.38%** | **54.41%** | **+0.03** |

The absolute task profiles are highly concordant between seeds: the Pearson
correlation of task scores is 0.985 for NTP and 0.997 for MTP. This is strong
evidence that the evaluation pipeline and broad model capabilities are stable.
The objective effect is not comparably stable. Only three of seven task-level
MTP-minus-NTP signs agree across seeds, and the two task-effect vectors have
Pearson correlation -0.48 and Spearman correlation -0.18. BoolQ is the largest
source of movement: seed-67 NTP is 7.13 points above seed-3407 NTP, reversing
the original apparent MTP advantage.

## Paired evaluation-example uncertainty

Within seed 67, all 20,465 core examples match uniquely across NTP, MTP, and
variable span. A 2,000-resample deterministic paired bootstrap gives the
seed-67 intervals below; the previously reported seed-3407 interval used
10,000 resamples:

| Comparison | Core macro difference | Paired-bootstrap 95% CI |
|---|---:|---:|
| MTP − NTP, seed 3407 | +1.64 points | [+0.85, +2.43] |
| MTP − NTP, seed 67 | +0.03 points | [-0.76, +0.80] |
| NTP − variable span, seed 67 | +1.71 points | [+0.90, +2.53] |
| MTP − variable span, seed 67 | +1.73 points | [+0.95, +2.52] |

At seed 67, ARC-Easy favors MTP by 1.60 points (paired 95% interval
[+0.13, +3.11]), while BoolQ favors NTP by 2.23 points ([+0.49, +3.98] in the
NTP-minus-MTP direction). The other task-level accuracy intervals include
zero. These opposing task effects explain the aggregate tie.

Paired intervals quantify finite evaluation-example uncertainty conditional
on fixed checkpoints. They do not quantify training-seed uncertainty. The fact
that the seed-3407 interval excludes zero while the seed-67 interval is centered
near zero therefore does not constitute contradictory evaluation machinery;
it shows that the original objective effect did not replicate across training
seeds.

The variable-span gaps do exclude zero under this conditional, example-level
analysis. That is stronger evidence than a macro difference alone, but it still
does not measure training-seed uncertainty. The five-shot paired result is
decisive for MTP over variable span and inconclusive-to-directional for NTP over
variable span, as reported above.

## Reliability and scaling implications

The results are not pure evaluation noise. Absolute benchmark profiles are
very stable, both seed-67 objectives show coherent improvement across the main
stage, MTP has lower final validation loss in both seeds, and MTP has better
token-weighted Paloma perplexity in both seeds. Those repeated directions are a
real signal that MTP may improve language-model fit. Five-shot core adds a
replicated downstream signal of about +1.3--1.5 points, while zero-shot remains
seed-sensitive: it changes from +1.64 points to +0.03, only three zero-shot
task-effect signs replicate, and the original late-stage NTP/MTP divergence is
absent across the seed-67 checkpoint trajectory. The five-shot result raises
confidence in an MTP benefit, but two training seeds and one fixed demonstration
sample are still insufficient to estimate both training-seed and prompt-example
variance precisely.

The mean MTP core advantage across the two seeds is +0.83 points, but with only
two paired training runs that average should not be treated as a stable effect
estimate. The between-seed spread is larger than the seed-67 effect itself.

The replicated five-shot result makes an exploratory MTP scale-up more defensible
than the zero-shot analysis alone, especially because validation and Paloma are
also directionally favorable. It still does not make a larger run a
confirmatory test of a proven general downstream benefit: the zero-shot
trajectory does not replicate, and the five-shot conclusion has not yet been
tested across demonstration samples. Any scale-up should therefore retain a
matched NTP control.

Variable span now has two training seeds and does not beat both matched controls
on either zero-shot or five-shot core. At seed 67 it is also clearly worse on
validation and Paloma. This is enough signal not to prefer variable span for a
scale-up from this screen: MTP is the more defensible experimental objective,
while NTP remains the necessary control. It is not evidence that masking can
never help; it is evidence that this particular 15%-mask, max-five truncated-
geometric recipe has not shown a robust advantage at 1.1B. The completed
variable-span GSM8K result is near floor like every other 1.1B condition and
does not alter that conclusion.

Detailed paired outputs are in `paired-core.md`, `paired-core.json`,
`paired-core-5shot.md`, and `paired-core-5shot.json`. The three-condition
reports use 2,000 deterministic
paired-bootstrap resamples with seed 12345; this is sufficient for the broad
intervals reported here and avoids implying training-seed uncertainty.
