#!/bin/bash
# 연구실 GPU(vLLM)를 배포 서버에 SSH 리버스 터널로 붙인다.
#
# 서버의 127.0.0.1:8000 → 이 머신의 vLLM(8000). 서버 쪽 방화벽을 열 필요가 없고
# 외부에 GPU 가 노출되지 않는다. autossh 가 끊긴 연결을 자동 복구한다.
#
#   export NCP_HOST=ubuntu@1.2.3.4
#   bash scripts/serve/start_gpu_tunnel.sh
#
# 공개 터널(trycloudflare)을 쓰지 않는 이유: Cloudflare 는 원본 응답을 100초에서
# 끊는데(524) 이 시스템의 LLM 호출은 실측 최대 164초다. 게다가 주소만 알면
# 누구나 이 GPU 로 추론을 돌릴 수 있다.

set -e

: "${NCP_HOST:?NCP_HOST 를 지정하세요 (예: export NCP_HOST=ubuntu@1.2.3.4)}"
NCP_PORT="${NCP_PORT:-22}"
VLLM_PORT="${VLLM_PORT:-8000}"

if ! command -v autossh > /dev/null; then
    echo "autossh 가 없습니다: sudo apt install -y autossh" >&2
    exit 1
fi

if ! curl -s -m 5 "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; then
    echo "[경고] 로컬 vLLM(${VLLM_PORT})이 응답하지 않습니다. start_vllm.sh 를 먼저 실행하세요." >&2
fi

echo "[터널] ${NCP_HOST}:${VLLM_PORT} → localhost:${VLLM_PORT} (autossh, 자동 재연결)"

# -M 0 + ServerAlive 옵션으로 끊김 감지, -N 은 원격 명령 실행 없음
exec autossh -M 0 -N \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3" \
    -o "ExitOnForwardFailure yes" \
    -o "StrictHostKeyChecking accept-new" \
    -p "${NCP_PORT}" \
    -R "127.0.0.1:${VLLM_PORT}:127.0.0.1:${VLLM_PORT}" \
    "${NCP_HOST}"
