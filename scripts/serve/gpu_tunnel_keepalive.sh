#!/bin/bash
# GPU 터널 상시 유지 — 끊기면 자동 재연결한다.
#
# autossh 가 설치돼 있으면 그걸 쓰고, 없으면 ssh 를 루프로 감싸 같은 효과를 낸다
# (autossh 설치에는 sudo 가 필요해서 없는 환경도 지원).
#
#   bash scripts/serve/gpu_tunnel_keepalive.sh            # 포그라운드
#   crontab 에 @reboot 로 등록하면 재부팅 후에도 자동 복구 (install_tunnel_cron.sh 참고)
#
# 서버의 127.0.0.1:8000 → 이 머신의 vLLM(8000).

NCP_HOST="${NCP_HOST:-ubuntu@211.233.220.12}"
NCP_PORT="${NCP_PORT:-22}"
VLLM_PORT="${VLLM_PORT:-8000}"
LOG="${TUNNEL_LOG:-$HOME/gpu_tunnel.log}"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

SSH_OPTS=(
    -N
    -o "ServerAliveInterval 30"
    -o "ServerAliveCountMax 3"
    -o "ExitOnForwardFailure yes"
    -o "StrictHostKeyChecking accept-new"
    -o "BatchMode yes"
    -p "$NCP_PORT"
    -R "127.0.0.1:${VLLM_PORT}:127.0.0.1:${VLLM_PORT}"
    "$NCP_HOST"
)

log "터널 시작: ${NCP_HOST} (vLLM ${VLLM_PORT})"

if command -v autossh > /dev/null; then
    exec autossh -M 0 "${SSH_OPTS[@]}"
fi

# autossh 없이 재연결 루프. 실패가 반복되면 대기 시간을 늘려 로그 폭주를 막는다.
backoff=5
while true; do
    ssh "${SSH_OPTS[@]}"
    code=$?
    log "터널 종료(code=$code). ${backoff}초 후 재연결"
    sleep "$backoff"
    if [ "$backoff" -lt 60 ]; then backoff=$((backoff * 2)); else backoff=60; fi
    # 연결이 한 번이라도 오래 유지되면 backoff 를 초기화하기 위해 재확인
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -p "$NCP_PORT" "$NCP_HOST" true 2>/dev/null; then
        backoff=5
    fi
done
