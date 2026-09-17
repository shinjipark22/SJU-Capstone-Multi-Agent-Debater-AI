# 배포 (네이버 클라우드 서버 + 로컬 GPU)

```
사용자 브라우저
      │ HTTPS
      ▼
[네이버 클라우드 서버]  nginx → 정적 프론트(dist/) + /api 경로를 FastAPI(127.0.0.1:8001)로 프록시
      │ SSH 리버스 터널 (127.0.0.1:8000)
      ▼
[연구실 머신]  vLLM (RTX 4090 × 2, Qwen2.5-32B-AWQ)
```

앱 서버는 GPU·임베딩 모델을 쓰지 않는다. 벡터 캐시는 Pinecone REST, LLM 은 전부 원격 호출이라
CPU 는 대부분 대기 상태다 (실측 API 프로세스 RSS 0.6 GB, 수집 DB 252 KB).

네이버 클라우드 콘솔에서 서버를 처음 만드는 절차는 [ncp-setup.md](ncp-setup.md) 참고.

## 서버 사양

| 항목 | 권장 |
| --- | --- |
| 사양 | 2 vCPU / 8 GB (동시 참가자 5명 이상이면 4 vCPU / 16 GB) |
| 디스크 | 50 GB SSD |
| OS | Ubuntu 22.04 |
| 열어야 할 포트 | 80, 443 (SSH 22 는 관리자 IP 한정) |

FastAPI 는 127.0.0.1 에만 바인딩하고 nginx 뒤에 둔다. 8001 을 외부에 열지 않는다.

## 1. GPU 연결 — SSH 리버스 터널

vLLM 을 공개 터널(trycloudflare)로 노출하지 않는다. 이유 두 가지:

- Cloudflare 는 원본 응답을 **100초**에서 끊는데(524), 이 시스템의 LLM 호출은 실측 **최대 164초**다.
- 주소만 알면 누구나 연구실 GPU 로 추론을 돌릴 수 있다.

연구실 머신에서 서버로 나가는 리버스 터널을 건다 (방화벽 설정 불필요, 바깥에서 GPU 가 안 보임).

```bash
# 연구실 머신에서 1회 준비
sudo apt install -y autossh
ssh-copy-id -p 22 ubuntu@<서버IP>          # 키 인증 필수 (비번 입력 없이 재연결돼야 함)

# 상시 실행
export NCP_HOST=ubuntu@<서버IP>
bash scripts/serve/start_gpu_tunnel.sh      # 서버의 127.0.0.1:8000 → 이 머신의 vLLM
```

서버 쪽 `.env` 는 그대로 `VLLM_BASE_URL=http://127.0.0.1:8000/v1` 를 쓰면 된다.

## 2. 서버 세팅

```bash
sudo apt update && sudo apt install -y python3.10-venv nginx
git clone <repo> /opt/debate && cd /opt/debate
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

cp deploy/env.example .env && vi .env        # 키 채우기
sudo cp deploy/debate-api.service /etc/systemd/system/
sudo systemctl enable --now debate-api
```

## 3. 프론트 배포

프론트는 API 를 상대 경로로 호출하므로 nginx 가 같은 도메인에서 프록시하면 별도 설정이 없다.

```bash
# 로컬(빌드 환경)에서
cd frontend && npm run build
rsync -av dist/ ubuntu@<서버IP>:/var/www/debate/

# 서버에서
sudo cp deploy/nginx.conf /etc/nginx/sites-available/debate
sudo ln -sf /etc/nginx/sites-available/debate /etc/nginx/sites-enabled/debate
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d <도메인>             # HTTPS
```

## 4. 확인

```bash
curl https://<도메인>/health                  # {"status":"ok",...}
curl https://<도메인>/topics | head -c 200    # 논제 목록
curl https://<도메인>/sessions                # 수집 세션 (초기엔 빈 배열)
```

## 운영 메모

- **수집 데이터는 `data/sessions.db` 파일 하나**다. 정기 백업:
  `sqlite3 data/sessions.db ".backup /backup/sessions-$(date +%F).db"`
- 내보내기: `GET /sessions/export.csv` (23컬럼), `GET /surveys/export.csv` (58컬럼),
  서버 접속 없이 쓰려면 `python scripts/export_sessions.py -o sessions.csv`
- **API 재시작 시 진행 중이던 토론은 이어갈 수 없다** (LangGraph MemorySaver). 실험 중에는
  재배포하지 말 것. 이미 끝난 세션의 최종 리포트는 `turn_analyses` 에서 복구된다.
- 설문 문항(`src/survey/schema.json`)을 고치면 API 재시작이 필요하다 (기동 시 1회 로드).
