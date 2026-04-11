#!/bin/bash
# vLLM 서빙 스크립트 — Qwen2.5-32B-Instruct AWQ
# GPU 0 단독 (1장, ~18GB)

set -e

HF_HOME="${HF_HOME:-/disk1/SJ/huggingface/hub}"
export HF_HOME

echo "[GPU 0] Qwen2.5-32B-Instruct-AWQ 시작 (포트 8000)..."
echo "  양자화: AWQ INT4"
echo "  tool calling: hermes"
echo ""

CUDA_VISIBLE_DEVICES=0 vllm serve \
    Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --quantization awq \
    --dtype float16 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --download-dir "$HF_HOME"
