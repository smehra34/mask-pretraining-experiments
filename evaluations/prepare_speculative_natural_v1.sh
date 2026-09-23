#!/bin/bash
set -euo pipefail

MEGATRON_ROOT=${MEGATRON_ROOT:-/users/smehra/developer/Megatron-LM}
DCLM_ROOT=${DCLM_ROOT:-/iopsstor/scratch/cscs/smehra/tokenized_datasets/dclm-edu__mistral-7b-v0.3}
PALOMA_ROOT=${PALOMA_ROOT:-/iopsstor/scratch/cscs/smehra/eval_datasets/paloma}
OUTPUT_ROOT=${OUTPUT_ROOT:-/users/smehra/developer/mask-experiment-manager/evaluations/workloads/speculative-natural-v1}
TOKENIZER_JSON=${TOKENIZER_JSON:-${DCLM_ROOT}/tokenizer/tokenizer.json}

mkdir -p "$OUTPUT_ROOT"
cd "$MEGATRON_ROOT"
export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python tools/prepare_speculative_workload.py dclm-validation \
  --dataset-root "$DCLM_ROOT" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --samples 200 --seed 3407 --prompt-tokens 256 --min-continuation-tokens 128 \
  --output "$OUTPUT_ROOT/dclm-edu-validation.jsonl"

python tools/prepare_speculative_workload.py paloma \
  --data-root "$PALOMA_ROOT" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --split test --samples-per-domain 200 --seed 3407 \
  --prompt-tokens 256 --min-continuation-tokens 128 \
  --source 'paloma_wikipedia=m2d2_wikipedia_unsplit/test/test_History*.jsonl.gz' \
  --source 'paloma_wikipedia=m2d2_wikipedia_unsplit/test/test_Culture*.jsonl.gz' \
  --source 'paloma_wikipedia=m2d2_wikipedia_unsplit/test/test_Society*.jsonl.gz' \
  --source 'paloma_academic=m2d2_s2orc_unsplit/test/test_Art.jsonl.gz' \
  --source 'paloma_academic=m2d2_s2orc_unsplit/test/test_Philosophy.jsonl.gz' \
  --source 'paloma_academic=m2d2_s2orc_unsplit/test/test_econ*.jsonl.gz' \
  --source 'paloma_news=ptb/test/*.jsonl.gz' \
  --source 'paloma_reddit=dolma_100_subreddits/test/test_00_AskReddit.jsonl.gz' \
  --source 'paloma_reddit=dolma_100_subreddits/test/test_08_todayilearned.jsonl.gz' \
  --source 'paloma_reddit=dolma_100_subreddits/test/test_20_explainlikeimfive.jsonl.gz' \
  --source 'paloma_reddit=dolma_100_subreddits/test/test_60_science.jsonl.gz' \
  --source 'paloma_reddit=dolma_100_subreddits/test/test_73_books.jsonl.gz' \
  --source 'paloma_reddit=dolma_100_subreddits/test/test_81_askscience.jsonl.gz' \
  --output "$OUTPUT_ROOT/paloma-balanced.jsonl"

python tools/prepare_speculative_workload.py combine \
  --input "$OUTPUT_ROOT/dclm-edu-validation.jsonl" \
  --input "$OUTPUT_ROOT/paloma-balanced.jsonl" \
  --output "$OUTPUT_ROOT/all.jsonl"
