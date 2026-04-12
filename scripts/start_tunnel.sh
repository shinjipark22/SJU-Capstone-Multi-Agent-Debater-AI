#!/bin/bash
# Cloudflare Quick Tunnel — FastAPI를 외부에 노출
# FastAPI가 먼저 실행되어 있어야 함 (start_api.sh)

set -e

PORT="${1:-8001}"

echo "[Cloudflare Tunnel] http://localhost:$PORT → https://xxx.trycloudflare.com"
echo "  생성된 URL을 Spring Boot application.yml에 설정하세요"
echo ""

cloudflared tunnel --url "http://localhost:$PORT"
