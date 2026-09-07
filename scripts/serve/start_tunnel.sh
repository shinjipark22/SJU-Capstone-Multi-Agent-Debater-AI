#!/bin/bash
# [주의] 배포에는 쓰지 말 것 — 잠깐 시연·공유용 임시 터널이다.
#
#   - Cloudflare 는 원본 응답을 100초에서 끊는다(524). 이 시스템의 LLM 호출은
#     실측 최대 164초라 역할반전·종합 같은 긴 턴이 그대로 실패한다.
#   - trycloudflare 주소는 무인증이라 주소를 아는 누구나 접근할 수 있다.
#
# 배포 서버 ↔ 연구실 GPU 연결은 scripts/serve/start_gpu_tunnel.sh (SSH 리버스 터널)를,
# 서버 구성은 deploy/README.md 를 참고할 것.
#
# Cloudflare Quick Tunnel — FastAPI를 외부에 노출
# FastAPI가 먼저 실행되어 있어야 함 (start_api.sh)

set -e

PORT="${1:-8001}"

echo "[Cloudflare Tunnel] http://localhost:$PORT → https://xxx.trycloudflare.com"
echo "  생성된 URL을 Spring Boot application.yml에 설정하세요"
echo ""

CLOUDFLARED="${CLOUDFLARED:-$(which cloudflared 2>/dev/null || echo "$HOME/bin/cloudflared")}"
"$CLOUDFLARED" tunnel --url "http://localhost:$PORT"
