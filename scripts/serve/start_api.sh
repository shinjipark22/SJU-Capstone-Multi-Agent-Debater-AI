#!/bin/bash
# FastAPI 서버 실행 스크립트
# vLLM이 먼저 실행되어 있어야 함 (start_vllm.sh)

set -e

cd "$(dirname "$0")/../.."

# ~/.local/lib/python3.10/site-packages 의 낡은 langchain-core(0.3.29)가 conda 환경의
# 1.4.9 를 가려 langgraph import 가 깨진다. user site 를 꺼서 환경 패키지만 쓰게 한다.
export PYTHONNOUSERSITE=1

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "[FastAPI] 서버 시작 (포트 8001)..."
echo "  vLLM 연결: http://localhost:8000/v1"
echo "  Swagger: http://localhost:8001/docs"
echo ""

uvicorn src.main:app --host 0.0.0.0 --port 8001
