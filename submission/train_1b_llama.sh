#!/bin/bash

# This launcher defines the fixed 1B Llama-family architecture and distributed
# topology. Recipes supply ordinary experimental, data, schedule, logging, and
# Slurm variation. Create another family launcher for architectural changes.

set -euo pipefail

# Modes:
#   Main:     sbatch --export=ALL,RUN_MODE=main,TRAIN_TOKENS=100000000000 train_1b_llama.sh
#   Cooldown: sbatch --export=ALL,RUN_MODE=cooldown,SOURCE_ITER=2384,COOLDOWN_TOKENS=10000000000 train_1b_llama.sh
#
# SOURCE_ITER is deliberately an iteration rather than a nominal token count: it
# identifies one exact checkpoint even when 10B is not iteration-aligned.

RUN_MODE=${RUN_MODE:-main}
TRAIN_TOKENS=${TRAIN_TOKENS:-}
SOURCE_ITER=${SOURCE_ITER:-}
COOLDOWN_TOKENS=${COOLDOWN_TOKENS:-}

DATASETS=${DATASETS:-/iopsstor/scratch/cscs/smehra/tokenized_datasets/dclm-edu__mistral-7b-v0.3}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-mistralai/Mistral-7B-v0.3}
MBS=${MBS:-8}
GBS=${GBS:-1024}
SEQ_LEN=${SEQ_LEN:-4096}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
PEAK_LR=${PEAK_LR:-0.00001}
MIN_LR=${MIN_LR:-0.0}
SEED=${SEED:-28}
LOG_INTERVAL=${LOG_INTERVAL:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
EVAL_ITERS=${EVAL_ITERS:-10}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.0}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
INPUT_MASK_RATIO=${INPUT_MASK_RATIO:-0.0}
INPUT_MASK_STRATEGY=${INPUT_MASK_STRATEGY:-random}
INPUT_MASK_SPAN_LENGTH=${INPUT_MASK_SPAN_LENGTH:-1}
INPUT_MASK_TOKEN=${INPUT_MASK_TOKEN:-[control_768]}
INPUT_MASK_DEBUG=${INPUT_MASK_DEBUG:-false}
INPUT_MASK_DEBUG_TOKENS=${INPUT_MASK_DEBUG_TOKENS:-128}
SAVE_EVERY_TOKENS=${SAVE_EVERY_TOKENS:-10000000000}
ROLLING_CHECKPOINTS=${ROLLING_CHECKPOINTS:-true}
ROLLING_SAVE_EVERY_TOKENS=${ROLLING_SAVE_EVERY_TOKENS:-}
COOLDOWN_STYLE=${COOLDOWN_STYLE:-minus_sqrt}
EXTENSION_COOLDOWN_STEPS=${EXTENSION_COOLDOWN_STEPS:-100}
SOURCE_CHECKPOINT_DIR=${SOURCE_CHECKPOINT_DIR:-}
CP_SIZE=${CP_SIZE:-1}
CP_COMM_TYPE=${CP_COMM_TYPE:-p2p}
ROTARY_BASE=${ROTARY_BASE:-500000}

AUTO_JOB_REQUEUE=${AUTO_JOB_REQUEUE:-false}
LOG_NCCL=${LOG_NCCL:-false}
NSYS_PROFILER=${NSYS_PROFILER:-false}
MOCK_DATA=${MOCK_DATA:-false}
BACKUP_CODEBASE=${BACKUP_CODEBASE:-false}
RUN_CAPSTOR_DIAGNOSTICS=${RUN_CAPSTOR_DIAGNOSTICS:-false}

MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/users/smehra/developer/Megatron-LM}
MEGATRON_RUNTIME_DEPS=${MEGATRON_RUNTIME_DEPS:-/users/smehra/developer/megatron-runtime-deps/nvrx-0.6.0}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-/iopsstor/scratch/cscs/$USER/datasets/cache}
PROJECT_NAME=${PROJECT_NAME:-mask_pretraining}
EXP_NAME=${EXP_NAME:-llama_1b_wsd}
EXPERIMENT_ARTIFACTS_DIR=${EXPERIMENT_ARTIFACTS_DIR:?The experiment manager must provide EXPERIMENT_ARTIFACTS_DIR}
CHECKPOINT_STORAGE_ROOT=${CHECKPOINT_STORAGE_ROOT:-/capstor/scratch/cscs/$USER/megatron-runs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$CHECKPOINT_STORAGE_ROOT/checkpoints/$PROJECT_NAME/$EXP_NAME}
MAIN_CKPT_DIR=$CHECKPOINT_ROOT/main

# Establish the composite default layout before creating any experiment/run
# descendants, so checkpoint and W&B files inherit the intended striping.
mkdir -p "$CHECKPOINT_STORAGE_ROOT"
lfs setstripe --component-end 4M --stripe-count 1 --component-end 64M --stripe-count 4 --component-end -1 --stripe-count 32 --stripe-size 4M "$CHECKPOINT_STORAGE_ROOT"

ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }
round_div() { echo $(( ($1 + $2 / 2) / $2 )); }

TOKENS_PER_ITER=$((GBS * SEQ_LEN))
SAVE_INTERVAL=$(round_div "$SAVE_EVERY_TOKENS" "$TOKENS_PER_ITER")
(( SAVE_INTERVAL > 0 )) || { echo "SAVE_EVERY_TOKENS is too small" >&2; exit 2; }
ACTUAL_SAVE_TOKENS=$((SAVE_INTERVAL * TOKENS_PER_ITER))
[[ $ROLLING_CHECKPOINTS == true || $ROLLING_CHECKPOINTS == false ]] || {
  echo "ROLLING_CHECKPOINTS must be 'true' or 'false'" >&2
  exit 2
}
if [[ $ROLLING_CHECKPOINTS == true ]]; then
  # By default, keep one rolling recovery checkpoint roughly ten times more
  # frequently than persistent milestones. Never round below one iteration.
  if [[ -z $ROLLING_SAVE_EVERY_TOKENS ]]; then
    ROLLING_SAVE_EVERY_TOKENS=$((SAVE_EVERY_TOKENS / 10))
  fi
  ROLLING_SAVE_INTERVAL=$(round_div "$ROLLING_SAVE_EVERY_TOKENS" "$TOKENS_PER_ITER")
  (( ROLLING_SAVE_INTERVAL > 0 )) || ROLLING_SAVE_INTERVAL=1
  ACTUAL_ROLLING_SAVE_TOKENS=$((ROLLING_SAVE_INTERVAL * TOKENS_PER_ITER))
fi
WARMUP_SAMPLES=$((WARMUP_STEPS * GBS))

case "$RUN_MODE" in
  main)
    : "${TRAIN_TOKENS:?Set TRAIN_TOKENS to the total main-run token budget}"
    TRAIN_ITERS=$(ceil_div "$TRAIN_TOKENS" "$TOKENS_PER_ITER")
    TRAIN_SAMPLES=$((TRAIN_ITERS * GBS))
    LOAD_DIR=$MAIN_CKPT_DIR
    SAVE_DIR=$MAIN_CKPT_DIR
    RUN_NAME=${EXP_NAME}-main
    # Keep the main run entirely within WSD's warmup/stable region. The decay
    # boundary is one sample beyond the run's target and is never entered.
    LR_ARGS=(
      --lr "$PEAK_LR" --min-lr "$MIN_LR"
      --lr-decay-style WSD
      --lr-decay-samples $((TRAIN_SAMPLES + 1))
      --lr-wsd-decay-samples 1
      --lr-warmup-samples "$WARMUP_SAMPLES"
    )
    ;;
  cooldown)
    : "${SOURCE_ITER:?Set SOURCE_ITER to the exact main checkpoint iteration}"
    : "${COOLDOWN_TOKENS:?Set COOLDOWN_TOKENS to the cooldown token budget}"
    [[ $SOURCE_ITER =~ ^[0-9]+$ ]] || { echo "SOURCE_ITER must be an integer" >&2; exit 2; }

    SOURCE_SAMPLES=$((SOURCE_ITER * GBS))
    COOLDOWN_ITERS=$(ceil_div "$COOLDOWN_TOKENS" "$TOKENS_PER_ITER")
    COOLDOWN_SAMPLES=$((COOLDOWN_ITERS * GBS))
    TRAIN_SAMPLES=$((SOURCE_SAMPLES + COOLDOWN_SAMPLES))
    PADDED_ITER=$(printf '%07d' "$SOURCE_ITER")
    SOURCE_CKPT=$MAIN_CKPT_DIR/iter_$PADDED_ITER
    BRANCH_DIR=$CHECKPOINT_ROOT/cooldowns/from_iter_$PADDED_ITER
    SAVE_DIR=$BRANCH_DIR/checkpoints
    LOAD_DIR=$SAVE_DIR
    RUN_NAME=${EXP_NAME}-cooldown-from-$PADDED_ITER

    [[ -d $SOURCE_CKPT ]] || { echo "Missing source checkpoint: $SOURCE_CKPT" >&2; exit 2; }
    if [[ ! -d $LOAD_DIR/iter_$PADDED_ITER ]]; then
      [[ ! -e $LOAD_DIR/latest_checkpointed_iteration.txt ]] || {
        echo "Refusing to initialize a cooldown in a non-empty checkpoint directory: $LOAD_DIR" >&2
        exit 2
      }
      mkdir -p "$LOAD_DIR"
      cp -a --reflink=auto "$SOURCE_CKPT" "$LOAD_DIR/iter_$PADDED_ITER"
      printf '%s\n' "$SOURCE_ITER" > "$LOAD_DIR/latest_checkpointed_iteration.txt"
    fi

    LR_ARGS=(
      --lr "$PEAK_LR" --min-lr "$MIN_LR"
      --lr-decay-style WSD
      --lr-wsd-decay-style "$COOLDOWN_STYLE"
      --lr-decay-samples "$TRAIN_SAMPLES"
      --lr-wsd-decay-samples "$COOLDOWN_SAMPLES"
      --lr-warmup-samples 0
      --override-opt-param-scheduler
    )
    ;;
  extension)
    : "${TRAIN_TOKENS:?Set TRAIN_TOKENS to the extension token budget}"
    : "${SOURCE_CHECKPOINT_DIR:?Set SOURCE_CHECKPOINT_DIR to the base checkpoint root}"
    [[ -f $SOURCE_CHECKPOINT_DIR/latest_checkpointed_iteration.txt ]] || {
      echo "Missing checkpoint tracker: $SOURCE_CHECKPOINT_DIR/latest_checkpointed_iteration.txt" >&2
      exit 2
    }
    TRAIN_ITERS=$(ceil_div "$TRAIN_TOKENS" "$TOKENS_PER_ITER")
    TRAIN_SAMPLES=$((TRAIN_ITERS * GBS))
    EXTENSION_STEPS=$(ceil_div "$TRAIN_SAMPLES" "$GBS")
    (( WARMUP_STEPS > 0 && WARMUP_STEPS < EXTENSION_STEPS )) || {
      echo "WARMUP_STEPS must be positive and below extension steps ($EXTENSION_STEPS)" >&2
      exit 2
    }
    (( EXTENSION_COOLDOWN_STEPS > 0 && WARMUP_STEPS + EXTENSION_COOLDOWN_STEPS < EXTENSION_STEPS )) || {
      echo "Extension warmup + cooldown must leave a non-empty stable phase" >&2
      exit 2
    }
    SAVE_DIR=$MAIN_CKPT_DIR
    RUN_NAME=${EXP_NAME}-extension
    if [[ -f $SAVE_DIR/latest_checkpointed_iteration.txt ]]; then
      LOAD_DIR=$SAVE_DIR
      EXTENSION_LOAD_ARGS=()
    else
      LOAD_DIR=$SOURCE_CHECKPOINT_DIR
      EXTENSION_LOAD_ARGS=(--reset-training-progress)
    fi
    LR_ARGS=(
      --lr "$PEAK_LR" --min-lr "$MIN_LR"
      --lr-decay-style WSD
      --lr-decay-samples "$TRAIN_SAMPLES"
      --lr-wsd-decay-style "$COOLDOWN_STYLE"
      --lr-wsd-decay-samples $((EXTENSION_COOLDOWN_STEPS * GBS))
      --lr-warmup-samples $((WARMUP_STEPS * GBS))
      --override-opt-param-scheduler
      "${EXTENSION_LOAD_ARGS[@]}"
    )
    ;;
  *)
    echo "RUN_MODE must be 'main' or 'cooldown', got: $RUN_MODE" >&2
    exit 2
    ;;
esac

LOGGING_DIR=$EXPERIMENT_ARTIFACTS_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
DEBUG_DIR=$EXPERIMENT_ARTIFACTS_DIR/debug/$SLURM_JOB_ID
BACKUP_CODEBASE_DIR=$EXPERIMENT_ARTIFACTS_DIR/source/Megatron-LM
WANDB_DIR=$CHECKPOINT_ROOT/wandb/$RUN_NAME

mkdir -p "$SAVE_DIR" "$LOGGING_DIR" "$DEBUG_DIR" "$WANDB_DIR"

echo "Mode: $RUN_MODE"
echo "Training target: $TRAIN_SAMPLES samples ($((TRAIN_SAMPLES * SEQ_LEN)) tokens)"
echo "Input masking: ratio=$INPUT_MASK_RATIO strategy=$INPUT_MASK_STRATEGY span_length=$INPUT_MASK_SPAN_LENGTH token=$INPUT_MASK_TOKEN"
echo "Checkpoint interval: $SAVE_INTERVAL iterations ($ACTUAL_SAVE_TOKENS tokens; requested $SAVE_EVERY_TOKENS)"
if [[ $ROLLING_CHECKPOINTS == true ]]; then
  echo "Rolling recovery interval: $ROLLING_SAVE_INTERVAL iterations ($ACTUAL_ROLLING_SAVE_TOKENS tokens; requested $ROLLING_SAVE_EVERY_TOKENS)"
else
  echo "Rolling recovery checkpoints: disabled"
fi
echo "Load: $LOAD_DIR"
echo "Save: $SAVE_DIR"

export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-25679}

# Per-process caches on local storage, retained from the original run script.
export TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor/${SLURM_PROCID:-0}
export TRITON_HOME_DIR=/tmp/.triton/${SLURM_PROCID:-0}
export PYTHON_CACHE_DIR=/tmp/.python_cache/${SLURM_PROCID:-0}
export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache
mkdir -p "$TORCH_INDUCTOR_CACHE_DIR" "$TRITON_HOME_DIR" "$TRITON_CACHE_DIR" "$PYTHON_CACHE_DIR"
ulimit -c 0

if [[ $BACKUP_CODEBASE == true ]]; then
  mkdir -p "$BACKUP_CODEBASE_DIR"
  if [[ -z $(ls -A "$BACKUP_CODEBASE_DIR") ]]; then
    rsync -a --exclude-from="$MEGATRON_LM_DIR/.gitignore" "$MEGATRON_LM_DIR/" "$BACKUP_CODEBASE_DIR/"
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

cd "$MEGATRON_LM_DIR"
[[ -d "$MEGATRON_RUNTIME_DEPS" ]] || {
  echo "Missing Megatron runtime dependency overlay: $MEGATRON_RUNTIME_DEPS" >&2
  exit 2
}
export PYTHONPATH=$MEGATRON_RUNTIME_DEPS:$MEGATRON_LM_DIR:${PYTHONPATH:-}

################ Megatron argument groups ################
# These remain separate so model, optimizer, parallelism, and data settings can
# be edited in the same way as submit-1B_dev.sh.

TRANSFORMER_ENGINE_ARGS=(
  --transformer-impl transformer_engine
  # Require Transformer Engine's FlashAttention backend instead of allowing
  # the default "auto" selection to fall back to fused or unfused attention.
  --attention-backend flash
  # --use-precision-aware-optimizer
  # --main-grads-dtype bf16
)

NETWORK_SIZE_ARGS=(
  # MEAP 1.1B architecture. The paper's prose gives 24 layers and 32 heads;
  # Table 12 appears to transpose those two values (2048 is not divisible by 24).
  --num-layers 24
  --hidden-size 2048
  --ffn-hidden-size 5632
  --num-attention-heads 32
  --group-query-attention
  # The released tiny_LLaMA_1b_mask config uses two KV/query groups.
  --num-query-groups 2
  --max-position-embeddings "$SEQ_LEN"
  --position-embedding-type rope
  --rotary-base "$ROTARY_BASE"
  --rotary-percent 1.0
  --make-vocab-size-divisible-by 128
  --normalization RMSNorm
  --norm-epsilon 1e-5
  --swiglu
  # The released model instantiates independent token-embedding and LM-head
  # matrices rather than tying their weights.
  --untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
  --log-throughput
  --tensorboard-dir "$TENSORBOARD_DIR"
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
  --attention-dropout "$ATTENTION_DROPOUT"
  --hidden-dropout "$HIDDEN_DROPOUT"
  --weight-decay "$WEIGHT_DECAY"
  --clip-grad 1.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-08
)

TRAINING_ARGS=(
  --micro-batch-size "$MBS"
  --global-batch-size "$GBS"
  --train-samples "$TRAIN_SAMPLES"
  --log-interval "$LOG_INTERVAL"
  --eval-interval "$EVAL_INTERVAL"
  --eval-iters "$EVAL_ITERS"
  --no-check-for-nan-in-loss-and-grad
  --cross-entropy-loss-fusion
  --disable-bias-linear
  --optimizer adam
  --dataloader-type single
  --manual-gc
  --manual-gc-interval 100
)

INITIALIZATION_ARGS=(
  --seed "$SEED"
  # sqrt(2 / 5 / hidden_size), matching MEAP's base Linear/Embedding init.
  --init-method-std 0.013975424859373685
)

LEARNING_RATE_ARGS=("${LR_ARGS[@]}")

CHECKPOINTING_ARGS=(
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
  --load "$LOAD_DIR"
  --ckpt-format torch_dist
  --async-save
)
if [[ $ROLLING_CHECKPOINTS == true ]]; then
  CHECKPOINTING_ARGS+=(
    --non-persistent-save-interval "$ROLLING_SAVE_INTERVAL"
    --non-persistent-ckpt-type global
  )
fi

MIXED_PRECISION_ARGS=(
  --bf16
)

DISTRIBUTED_ARGS=(
  # Fixed topology for this model-family launcher.
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size "$CP_SIZE"
  --cp-comm-type "$CP_COMM_TYPE"
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
)

TOKENIZER_ARGS=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "$TOKENIZER_MODEL"
)

DATA_ARGS=(
  --split 100,0,0
  --seq-length "$SEQ_LEN"
  --num-workers 2
  # --num-dataset-builder-threads 1
)

INPUT_MASKING_ARGS=(
  --input-mask-ratio "$INPUT_MASK_RATIO"
  --input-mask-strategy "$INPUT_MASK_STRATEGY"
  --input-mask-span-length "$INPUT_MASK_SPAN_LENGTH"
  --input-mask-token "$INPUT_MASK_TOKEN"
)
[[ $INPUT_MASK_DEBUG == true || $INPUT_MASK_DEBUG == false ]] || {
  echo "INPUT_MASK_DEBUG must be 'true' or 'false'" >&2
  exit 2
}
if [[ $INPUT_MASK_DEBUG == true ]]; then
  INPUT_MASKING_ARGS+=(
    --input-mask-debug
    --input-mask-debug-tokens "$INPUT_MASK_DEBUG_TOKENS"
  )
fi

if [[ $MOCK_DATA == true ]]; then
  DATA_ARGS+=(--mock-data)
else
  read -r -a DATA_PATHS <<< "$(python3 scripts/tools/create_data_config.py -p "$DATASETS")"
  DATA_ARGS+=(--data-path "${DATA_PATHS[@]}" --data-cache-path "$DATASET_CACHE_DIR")
fi

TORCHRUN_ARGS=(
  --nproc-per-node "$SLURM_GPUS_PER_NODE"
  --nnodes "$SLURM_NNODES"
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT"
  --rdzv_backend c10d
  # --max_restarts 0
  # --tee 3
)

WANDB_ARGS=()
if [[ -n ${WANDB_API_KEY:-} ]]; then
  export WANDB_DIR
  export WANDB_DATA_DIR=$WANDB_DIR/data
  export WANDB_ARTIFACT_DIR=$WANDB_DIR/artifacts
  mkdir -p "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
  WANDB_ARGS=(
    --wandb-save-dir "$WANDB_DIR"
    --wandb-project "$PROJECT_NAME"
    --wandb-exp-name "$RUN_NAME-$SLURM_JOB_ID"
  )
else
  export WANDB_MODE=disabled
fi

CMD_PREFIX=(numactl --membind=0-3)
if [[ $LOG_NCCL == true ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-%p.txt
fi

if [[ $RUN_CAPSTOR_DIAGNOSTICS == true && $MOCK_DATA != true ]]; then
  FIRST_DATA_BIN=$(find "$DATASETS" -type f -name '*.bin' -print -quit)
  [[ -n $FIRST_DATA_BIN && -f ${FIRST_DATA_BIN%.bin}.idx ]] \
    && echo "CAPSTOR DATASET PAIR OK FROM SUBMISSION NODE: $FIRST_DATA_BIN" \
    || echo "CANNOT FIND A COMPLETE DATASET PAIR FROM SUBMISSION NODE"
  srun --environment=test-env bash -c \
    "find '$DATASETS' -type f -name '*.bin' -print -quit | grep -q . && echo 'CAPSTOR DATASET OK' || echo 'CAPSTOR DATASET FAILED'"
fi

printf 'Training command:'
printf ' %q' "${CMD_PREFIX[@]}" torchrun "${TORCHRUN_ARGS[@]}" \
  "$MEGATRON_LM_DIR/pretrain_gpt.py" \
  "${TRANSFORMER_ENGINE_ARGS[@]}" "${NETWORK_SIZE_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" "${REGULARIZATION_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" "${INITIALIZATION_ARGS[@]}" \
  "${LEARNING_RATE_ARGS[@]}" "${CHECKPOINTING_ARGS[@]}" \
  "${MIXED_PRECISION_ARGS[@]}" "${DISTRIBUTED_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" "${DATA_ARGS[@]}" "${INPUT_MASKING_ARGS[@]}" "${WANDB_ARGS[@]}"
printf '\n'

echo "START TIME: $(date)"
srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --mpi=pmix \
  --distribution=block:block --network=disable_rdzv_get --environment=test-env \
  "${CMD_PREFIX[@]}" torchrun "${TORCHRUN_ARGS[@]}" \
  "$MEGATRON_LM_DIR/pretrain_gpt.py" \
  "${TRANSFORMER_ENGINE_ARGS[@]}" \
  "${NETWORK_SIZE_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "${REGULARIZATION_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${INITIALIZATION_ARGS[@]}" \
  "${LEARNING_RATE_ARGS[@]}" \
  "${CHECKPOINTING_ARGS[@]}" \
  "${MIXED_PRECISION_ARGS[@]}" \
  "${DISTRIBUTED_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${INPUT_MASKING_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
echo "END TIME: $(date)"
