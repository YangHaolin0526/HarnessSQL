#!/usr/bin/env bash
# Start the single-node Ray runtime used by the production RL job, then submit the
# portable training driver as a Ray job. All paths and credentials remain in
# the caller's environment; only HarnessSQL/Slime variables are forwarded.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
command -v ray >/dev/null || { echo "ray is required" >&2; exit 2; }

if command -v nvidia-smi >/dev/null 2>&1; then
  detected_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
  detected_gpus=0
fi
NUM_GPUS=${NUM_GPUS:-$detected_gpus}
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -le 0 ]]; then
  echo "NUM_GPUS must be set when GPUs cannot be detected" >&2
  exit 2
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export MASTER_ADDR NUM_GPUS

ray start --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "$RAY_DASHBOARD_PORT"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

runtime_env_json=$(python3 - <<'PY'
import json
import os

names = {
    "SLIME_ROOT", "HF_CHECKPOINT", "REF_LOAD", "SAVE_DIR", "PROMPT_DATA",
    "EVAL_DATA", "EVAL_INTERVAL", "MODEL_SIZE", "MODEL_CONFIG_SCRIPT",
    "ALGORITHM", "SFT_CONTEXT_LEN", "ROLLOUT_CONTEXT_LEN",
    "ROLLOUT_RESPONSE_LEN", "MASTER_ADDR", "NUM_GPUS", "PATH", "PYTHONPATH",
    "ROLLOUT_BS", "N_SAMPLES", "NUM_STEPS_PER_ROLLOUT", "GLOBAL_BS",
    "NUM_ROLLOUTS", "RL_LR", "ACTOR_GPUS", "ROLLOUT_GPUS", "TP_SIZE",
    "TRAIN_MAX_TOKENS", "TOOL_CALL_PARSER", "TRAIN_GRAD_DTYPE",
    "REF_KL_COEF", "EPS_CLIP", "EPS_CLIP_HIGH", "ALLOW_RESUME",
    "SAVE_INTERVAL", "N_SAMPLES_PER_EVAL_PROMPT", "SQL_EVAL_CONCURRENCY",
    "ROLLOUT_TEMP", "ROLLOUT_TOP_P", "ROLLOUT_TOP_K", "TRAIN_PP", "TRAIN_CP",
    "LOGPROB_CHUNK", "SGLANG_MEM_FRACTION", "SGLANG_CUDA_GRAPH_MAX_BS",
}
prefixes = ("HARNESS_SQL_", "SQL_", "DSH_SQL_", "SHIM_", "MAX_TOOL_")
forward = {
    key: value
    for key, value in os.environ.items()
    if key in names or key.startswith(prefixes)
}
forward["PYTHONUNBUFFERED"] = "1"
forward["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
print(json.dumps({"env_vars": forward}))
PY
)

ray job submit \
  --address "http://127.0.0.1:$RAY_DASHBOARD_PORT" \
  --runtime-env-json "$runtime_env_json" \
  -- bash "$repo_root/training/rl/train_rl_dsh.sh"
