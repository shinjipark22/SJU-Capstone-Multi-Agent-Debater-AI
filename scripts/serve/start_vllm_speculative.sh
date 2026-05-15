#!/bin/bash
# vLLM 서빙 스크립트 (어시스턴트 최적화 실험용) — Qwen2.5-32B-AWQ_MARLIN + prefix caching
# GPU 2+3 텐서병렬 (운영 vLLM 인스턴스의 GPU 0+1 과 분리)
# 포트 8002 사용 (운영 8000 과 분리)
#
# 1단계 변경: AWQ → AWQ_MARLIN (~30% 가속), prefix-caching ON
# speculative decoding 은 효과 미미해서 제거 (필요시 다시 추가)
#
# 사용:
#   bash scripts/serve/start_vllm_speculative.sh
#
# 어시스턴트가 이 인스턴스를 사용하게 하려면:
#   VLLM_BASE_URL=http://localhost:8002/v1 python tests/assistant_guide_e2e.py

set -e

# ~/.local/.../vllm (시스템 pip) 가 conda env vllm 보다 먼저 잡히는 문제 차단
export PYTHONNOUSERSITE=1

# conda env 의 libstdc++ 가 시스템보다 먼저 로드되도록 명시
export LD_LIBRARY_PATH="/home/user/miniconda3/envs/sj_agent/lib:${LD_LIBRARY_PATH:-}"
export PATH="/home/user/miniconda3/envs/sj_agent/bin:$PATH"

HF_HOME="${HF_HOME:-/disk1/SJ/huggingface/hub}"
export HF_HOME
export CUDA_VISIBLE_DEVICES=2,3

echo "[GPU 2+3] Qwen2.5-32B-AWQ_MARLIN + prefix-caching 시작 (포트 8002)..."
echo "  텐서병렬: 2장 (GPU 2, 3)"
echo "  양자화: AWQ_MARLIN (W4A16 marlin 커널, ~30% 가속)"
echo "  prefix caching: ON"
echo "  tool calling: hermes"
echo ""

/home/user/miniconda3/envs/sj_agent/bin/vllm serve \
    Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 8002 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --quantization awq_marlin \
    --dtype float16 \
    --enable-prefix-caching \
    --compilation-config '{"mode":0,"cudagraph_mode":2}' \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --download-dir "$HF_HOME"
