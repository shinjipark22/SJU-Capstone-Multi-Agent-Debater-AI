#!/bin/bash
# vLLM 서빙 스크립트 — Qwen2.5-32B-Instruct AWQ
# GPU 0+1 텐서병렬 (2장, KV 캐시 여유 확보)

set -e

HF_HOME="${HF_HOME:-/disk1/SJ/huggingface/hub}"
export HF_HOME

echo "[GPU 0+1] Qwen2.5-32B-Instruct-AWQ 시작 (포트 8000)..."
echo "  텐서병렬: 2장"
echo "  양자화: AWQ INT4"
echo "  tool calling: hermes"
echo ""

vllm serve \
    Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --quantization awq \
    --dtype float16 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --download-dir "$HF_HOME"
