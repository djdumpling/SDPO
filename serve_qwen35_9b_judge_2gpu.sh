#!/bin/bash

# Serve Qwen3.5-9B through a vLLM OpenAI-compatible endpoint on a 2-GPU inference box.
#
# Run this on the rented H100/H200 machine after installing the SDPO environment.
# The training job should point OPENAI_BASE_URL at this server's /v1 endpoint.

set -eo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"

echo "----------------------------------------------------------------"
echo "Starting external LitBench judge endpoint"
echo "Model: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tensor parallel size: $TP_SIZE"
echo "Max model len: $MAX_MODEL_LEN"
echo "----------------------------------------------------------------"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"
