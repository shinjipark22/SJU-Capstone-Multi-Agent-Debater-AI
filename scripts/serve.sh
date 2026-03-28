#!/usr/bin/env bash
# serve.sh — Qwen/Qwen3.5-9B vLLM 서버 실행 스크립트
#
# [왜 --enforce-eager 인가]
#   Qwen3.5-9B는 Mamba+Transformer 하이브리드 아키텍처(qwen3_next)를 사용한다.
#   vLLM의 CUDA graph 캡처 단계에서 causal_conv1d_update AssertionError 가 발생하는
#   미지원 버그가 있으므로 --enforce-eager 로 CUDA graph / torch.compile 을 비활성화한다.
#
#   GCP 배포 시에도 동일하게 이 옵션을 유지할 것.
#   향후 vLLM이 Qwen3.5 아키텍처를 완전 지원하면 이 옵션을 제거해도 된다.
#
# 사용법:
#   bash scripts/serve.sh            # 기본 포트 8000
#   PORT=9000 bash scripts/serve.sh  # 포트 지정

set -e

MODEL="Qwen/Qwen3.5-9B"
PORT="${PORT:-8000}"

echo "▶ vLLM 서버 시작"
echo "  모델 : $MODEL"
echo "  포트 : $PORT"
echo "  모드  : enforce-eager (CUDA graph 비활성화)"
echo ""

vllm serve "$MODEL" \
    --port "$PORT" \
    --enforce-eager
