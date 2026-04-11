#!/bin/bash
# Dual GPU vLLM 서빙 스크립트
# GPU 0: DeepSeek-R1-Distill-Qwen-14B AWQ INT4 (발언 생성)
# GPU 1: Qwen2.5-7B-Instruct (약점 분석 + 검색 판단 + 실시간 분석)

set -e

HF_HOME="${HF_HOME:-/disk1/SJ/huggingface/hub}"
export HF_HOME

echo "[GPU 0] DeepSeek-R1-Distill-Qwen-14B AWQ 시작 (포트 8000)..."
CUDA_VISIBLE_DEVICES=0 vllm serve \
    Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --quantization awq \
    --dtype float16 \
    --enforce-eager \
    --download-dir "$HF_HOME" \
    &

echo "[GPU 1] Qwen2.5-7B-Instruct 시작 (포트 8001)..."
CUDA_VISIBLE_DEVICES=1 vllm serve \
    Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager \
    --download-dir "$HF_HOME" \
    &

echo "두 모델 서빙 시작. 포트 8000 (DeepSeek), 8001 (Qwen 7B)"
echo "종료: kill %1 %2"
wait
