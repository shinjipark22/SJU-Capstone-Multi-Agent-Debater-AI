# SJU-Capstone-Multi-Agent-Debater-AI
세종대학교 Capstone Project 멀티 에이전트 토론 참여를 통한 문제 이해 지원 시스템 AI 파트 깃허브입니다.

## 구성

| 디렉토리 | 설명 |
| --- | --- |
| `src/` | LangGraph 토론 그래프 + FastAPI 서버 (`src/main.py`) |
| `src/storage/` | 토론 세션 데이터 수집 (SQLite → CSV) |
| `frontend/` | 서비스 프론트엔드 (React + Vite). 실행법은 `frontend/README.md` |
| `web/` | 이전 TypeScript 프로토타입 프론트엔드 |

## 서버 실행

```bash
bash scripts/serve/start_vllm.sh   # vLLM (포트 8000)
bash scripts/serve/start_api.sh    # FastAPI (포트 8001)
```

## 데이터 수집

토론 세션은 `data/sessions.db` (SQLite) 에 누적된다. 경로는 `DEBATE_DB_PATH` 로 바꿀 수 있다.

- `POST /debate/init` — 세션 행 생성 (`nickname`, `email`, `topic`, `user_stance`, `user_intensity`,
  `debate_format`(1:1/2:2/3:3), `status=ACTIVE`). 응답 SSE `session` 이벤트가 `record_id` 를 준다.
- `POST /evaluation?topic_id=...&session_id=<record_id>` — 토론 전·후 답변 원문과 채점 결과
  (진영별 pre/post 100점 환산·delta·요약) 를 같은 행에 채우고 `status=COMPLETED` 로 바꾼다.
- `GET /sessions` — 수집된 세션 목록 (JSON)
- `GET /sessions/export.csv` — `data-*.csv` 와 **동일한 23개 컬럼·순서**의 CSV 다운로드

서버 없이 덤프하려면:

```bash
python scripts/export_sessions.py -o sessions.csv
```
