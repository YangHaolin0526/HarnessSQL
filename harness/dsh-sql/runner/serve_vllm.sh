#!/bin/bash
# Serve one local model for the dsh SQL harness on an OpenAI-compatible port.
#
set -uo pipefail

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH is required" >&2
  exit 2
fi
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPUS="${GPUS:-1}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-65536}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_SEQS="${MAX_SEQS:-32}"
# Qwen3.5/3.6 emit XML function blocks (<tool_call><function=name><parameter=x>);
# the original Qwen3 series emits JSON inside <tool_call>, which is `hermes`.
# Read the model's chat_template.jinja before picking one — a mismatched parser
# looks like a model that cannot call tools.
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
LOG="${LOG:-runs/vllm_${SERVED_NAME}_${VLLM_PORT}.log}"

mkdir -p "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPUS"
# The proxy in this shell would otherwise swallow loopback calls to the server.
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="$no_proxy"

echo "[$SERVED_NAME] model=$MODEL_PATH gpus=$GPUS tp=$TP port=$VLLM_PORT max_len=$MAX_LEN util=$GPU_UTIL parser=$TOOL_PARSER"
echo "[$SERVED_NAME] log -> $LOG"

setsid nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$VLLM_PORT" \
  --host 127.0.0.1 \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  --trust-remote-code \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
PID_FILE="${PID_FILE:-runs/vllm_${SERVED_NAME}_${VLLM_PORT}.pid}"
mkdir -p "$(dirname "$PID_FILE")"
echo "$PID" > "$PID_FILE"
echo "[$SERVED_NAME] pid $PID"
