#!/bin/bash
# FastAPI 서버 실행 스크립트
# vLLM이 먼저 실행되어 있어야 함 (start_vllm.sh)

set -e

cd "$(dirname "$0")/../.."

echo "[FastAPI] 서버 시작 (포트 8001)..."
echo "  vLLM 연결: http://localhost:8000/v1"
echo "  Swagger: http://localhost:8001/docs"
echo ""

uvicorn src.main:app --host 0.0.0.0 --port 8001
