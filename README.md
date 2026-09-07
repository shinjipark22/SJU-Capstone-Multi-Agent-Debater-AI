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

23컬럼 CSV 형식을 지키기 위해, 아래 값들은 CSV 에 넣지 않고 별도 컬럼·테이블에 둔다.

| 저장 위치 | 내용 |
| --- | --- |
| `debate_sessions.mode` | `debate` / `constructive` |
| `debate_sessions.synthesis_draft` | 구성적 논쟁에서 합의한 최적해 |
| `turn_analyses` | 턴별 실시간 분석(논증·근거·표현 점수, 차원별 코멘트, EMA 스냅샷) |

`turn_analyses` 는 서버 재시작 대비용이기도 하다. 최종 리포트는 메모리의 judge 를
우선 쓰고, 없으면 이 테이블에서 복구해 생성한다.

서버 없이 덤프하려면:

```bash
python scripts/export_sessions.py -o sessions.csv
```

## 연구 설문 (구글폼 대체)

연구팀 구글폼 4종("토론"/"구성적 논쟁" × 사전/사후)의 문항을 그대로 추출해
`src/survey/schema.json` 에 넣고, 웹에서 직접 받아 같은 DB 에 저장한다.

- 사전 18문항 (연구 동의 · 인구통계 · 토론 경험 · 입장/확신 · 관점 수용성 7문항)
- 사후 33문항 (토론·구성적 논쟁 경험 7 · 학습 효과 5 · 관점 수용성 7 · AI 만족도 11 ·
  입장/확신 2 · 자유 의견 1)
- 구글폼의 닉네임 / 주제 / 사용자 진영 3개 항목은 세션 정보로 자동 기록되므로 문항에서 제외
- 주제·모드 의존 문구(`{topic}`, `{issue}`, `{mode_label}`)는 세션 값으로 치환해서 내려간다

| 엔드포인트 | 설명 |
| --- | --- |
| `GET /survey/schema/{pre\|post}?mode=&topic_id=` | 문항 스키마 (프론트가 이걸로 렌더링) |
| `POST /sessions/{id}/survey` | `{phase, answers}` 저장 (같은 단계 재제출 시 덮어씀) |
| `GET /sessions/{id}/survey/{phase}` | 저장된 응답 조회 |
| `GET /surveys/export.csv` | 세션당 한 행(`pre_*`, `post_*` 컬럼)으로 펼친 CSV |
