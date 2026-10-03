#!/usr/bin/env bash
# Portable production Slime launcher for harness-in-the-loop SQL RL.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

required=(SLIME_ROOT HF_CHECKPOINT REF_LOAD SAVE_DIR PROMPT_DATA)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required" >&2
    exit 2
  fi
done

export PYTHONPATH="$repo_root/training/rl:$SLIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for name in HARNESS_SQL_SYNTH_ROOT HARNESS_SQL_SPIDER_DB_DIR \
            HARNESS_SQL_DBT_SYNTH_ROOT HARNESS_SQL_DBT_DB_DIR \
            HARNESS_SQL_KTX_BIN HARNESS_SQL_KTX_PROJECT; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required; see .env.example" >&2
    exit 2
  fi
done

ALGORITHM=${ALGORITHM:-grpo}
MODEL_SIZE=${MODEL_SIZE:-qwen3_8b}
SFT_CONTEXT_LEN=${SFT_CONTEXT_LEN:-8192}
ROLLOUT_CONTEXT_LEN=${ROLLOUT_CONTEXT_LEN:-$SFT_CONTEXT_LEN}
ROLLOUT_RESPONSE_LEN=${ROLLOUT_RESPONSE_LEN:-$((SFT_CONTEXT_LEN / 8))}
ROLLOUT_BS=${ROLLOUT_BS:-4}
N_SAMPLES=${N_SAMPLES:-8}
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-2}
GLOBAL_BS=${GLOBAL_BS:-$((ROLLOUT_BS * N_SAMPLES / NUM_STEPS_PER_ROLLOUT))}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-150}
RL_LR=${RL_LR:-1e-6}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TP_SIZE=${TP_SIZE:-4}
TRAIN_MAX_TOKENS=${TRAIN_MAX_TOKENS:-$ROLLOUT_CONTEXT_LEN}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen25}

case "$MODEL_SIZE" in
  qwen3_8b)
    MODEL_CONFIG_SCRIPT=${MODEL_CONFIG_SCRIPT:-$repo_root/training/rl/model_configs/qwen3-8B.sh}
    TRAIN_GRAD_DTYPE=${TRAIN_GRAD_DTYPE:-fp32}
    ;;
  qwen3_14b)
    MODEL_CONFIG_SCRIPT=${MODEL_CONFIG_SCRIPT:-$repo_root/training/rl/model_configs/qwen3-14B.sh}
    TRAIN_GRAD_DTYPE=${TRAIN_GRAD_DTYPE:-bf16}
    ;;
  *) echo "MODEL_SIZE must be qwen3_8b or qwen3_14b" >&2; exit 2 ;;
esac

case "$SFT_CONTEXT_LEN" in
  8192|16384|32768) ;;
  *) echo "SFT_CONTEXT_LEN must be 8192, 16384, or 32768" >&2; exit 2 ;;
esac

if (( TRAIN_MAX_TOKENS < ROLLOUT_CONTEXT_LEN )); then
  TRAIN_MAX_TOKENS=$ROLLOUT_CONTEXT_LEN
fi

if (( (ROLLOUT_BS * N_SAMPLES) % NUM_STEPS_PER_ROLLOUT != 0 )); then
  echo "ROLLOUT_BS*N_SAMPLES must be divisible by NUM_STEPS_PER_ROLLOUT" >&2
  exit 2
fi

case "$ALGORITHM" in
  grpo)
    ADVANTAGE_ESTIMATOR=grpo
    DYNAMIC_FILTER=text2sql.filters.check_no_aborted
    REF_KL_COEF=${REF_KL_COEF:-0.01}
    EPS_CLIP=${EPS_CLIP:-0.2}
    EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.2}
    ;;
  dapo)
    ADVANTAGE_ESTIMATOR=grpo
    DYNAMIC_FILTER=text2sql.filters.check_no_aborted_and_reward_nonzero_std
    REF_KL_COEF=${REF_KL_COEF:-0.0}
    EPS_CLIP=${EPS_CLIP:-0.2}
    EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.28}
    ;;
  gspo)
    ADVANTAGE_ESTIMATOR=gspo
    DYNAMIC_FILTER=text2sql.filters.check_no_aborted
    REF_KL_COEF=${REF_KL_COEF:-0.0}
    EPS_CLIP=${EPS_CLIP:-0.0003}
    EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.0003}
    ;;
  *) echo "ALGORITHM must be grpo, dapo, or gspo" >&2; exit 2 ;;
esac

for path in "$SLIME_ROOT/train.py" "$HF_CHECKPOINT/config.json" \
            "$REF_LOAD/latest_checkpointed_iteration.txt" "$PROMPT_DATA" \
            "$MODEL_CONFIG_SCRIPT" "$HARNESS_SQL_KTX_BIN" \
            "$HARNESS_SQL_KTX_PROJECT/ktx.yaml"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

EVAL_ARGS=()
if [[ "${EVAL_INTERVAL:-0}" != 0 ]]; then
  if [[ -z "${EVAL_DATA:-}" || ! -s "$EVAL_DATA" ]]; then
    echo "EVAL_DATA must name a non-empty JSONL/Parquet file when EVAL_INTERVAL is nonzero" >&2
    exit 2
  fi
  case "$EVAL_DATA" in
    *.jsonl|*.parquet) ;;
    *) echo "EVAL_DATA must end in .jsonl or .parquet" >&2; exit 2 ;;
  esac
  EVAL_ARGS=(
    --eval-prompt-data sqlval "$EVAL_DATA"
    --eval-interval "$EVAL_INTERVAL"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-concurrency "${SQL_EVAL_CONCURRENCY:-8}"
  )
fi

if [[ -e "$SAVE_DIR/latest_checkpointed_iteration.txt" && "${ALLOW_RESUME:-0}" != 1 ]]; then
  echo "$SAVE_DIR already contains a run; set ALLOW_RESUME=1 to resume" >&2
  exit 2
fi
mkdir -p "$SAVE_DIR"

export SQL_MAX_CONCURRENT_ROLLOUTS=${SQL_MAX_CONCURRENT_ROLLOUTS:-8}
export SQL_ROLLOUT_THREADS=${SQL_ROLLOUT_THREADS:-256}
export SQL_MAX_TURNS=${SQL_MAX_TURNS:-24}
export SQL_TURN_TIMEOUT_SEC=${SQL_TURN_TIMEOUT_SEC:-300}
export SQL_QUERY_TIMEOUT_SEC=${SQL_QUERY_TIMEOUT_SEC:-30}
export SQL_GENERATE_GUARD_SEC=${SQL_GENERATE_GUARD_SEC:-1200}
export MAX_TOOL_OUTPUT_CHARS=${MAX_TOOL_OUTPUT_CHARS:-6000}
export DSH_SQL_MAX_TOKENS=$ROLLOUT_RESPONSE_LEN
export DSH_SQL_VLLM_CONTEXT=$ROLLOUT_CONTEXT_LEN

# The Slime model config populates MODEL_ARGS for the selected architecture.
source "$MODEL_CONFIG_SCRIPT"

cd "$SLIME_ROOT"
echo "HarnessSQL RL: algorithm=$ALGORITHM model=$MODEL_SIZE context=$ROLLOUT_CONTEXT_LEN response=$ROLLOUT_RESPONSE_LEN"
python train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --ref-load "$REF_LOAD" \
  --load "$SAVE_DIR" \
  --save "$SAVE_DIR" \
  --save-interval "${SAVE_INTERVAL:-10}" \
  --no-load-optim \
  --no-save-optim \
  --custom-generate-function-path text2sql.generate.generate \
  --custom-rollout-log-function-path text2sql.health_guard.abort_on_generation_collapse \
  --dynamic-sampling-filter-path "$DYNAMIC_FILTER" \
  --prompt-data "$PROMPT_DATA" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout "$NUM_ROLLOUTS" \
  --rollout-batch-size "$ROLLOUT_BS" \
  --n-samples-per-prompt "$N_SAMPLES" \
  --rollout-max-context-len "$ROLLOUT_CONTEXT_LEN" \
  --rollout-max-response-len "$ROLLOUT_RESPONSE_LEN" \
  --rollout-temperature "${ROLLOUT_TEMP:-0.6}" \
  --rollout-top-p "${ROLLOUT_TOP_P:-0.95}" \
  --rollout-top-k "${ROLLOUT_TOP_K:-20}" \
  --num-steps-per-rollout "$NUM_STEPS_PER_ROLLOUT" \
  --global-batch-size "$GLOBAL_BS" \
  --micro-batch-size 1 \
  --balance-data \
  --tensor-model-parallel-size "$TP_SIZE" \
  --sequence-parallel \
  --pipeline-model-parallel-size "${TRAIN_PP:-1}" \
  --context-parallel-size "${TRAIN_CP:-1}" \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --calculate-per-token-loss \
  --max-tokens-per-gpu "$TRAIN_MAX_TOKENS" \
  --log-probs-chunk-size "${LOGPROB_CHUNK:-1024}" \
  --advantage-estimator "$ADVANTAGE_ESTIMATOR" \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef "$REF_KL_COEF" \
  --entropy-coef 0.0 \
  --eps-clip "$EPS_CLIP" \
  --eps-clip-high "$EPS_CLIP_HIGH" \
  --optimizer adam \
  --lr "$RL_LR" \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.45}" \
  --sglang-disable-custom-all-reduce \
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-32}" \
  --sglang-tool-call-parser "$TOOL_CALL_PARSER" \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --optimizer-cpu-offload \
  --use-precision-aware-optimizer \
  --main-grads-dtype "$TRAIN_GRAD_DTYPE" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  "${EVAL_ARGS[@]}"
