# capstone_frontend (React + Vite)

`https://github.com/without-think/capstone_frontend.git` 를 가져와 FastAPI 백엔드(`src/main.py`)에 연결한 프론트엔드.

## 실행

이 서버엔 시스템 Node.js가 없어서 conda 환경(`sju_web`)의 Node 를 쓴다.

```bash
export PATH=/home/hth/anaconda3/envs/sju_web/bin:$PATH
cd frontend
npm install
cp .env.example .env        # VITE_API_BASE_URL 을 백엔드 주소로
npm run dev -- --host 0.0.0.0 --port 5173
npm run build               # 정적 배포물은 dist/
```

## 화면 흐름

| 단계 | 컴포넌트 | 백엔드 |
| --- | --- | --- |
| 카테고리 선택 | `TopicGrid` | — |
| 세부 논제 선택 | `SubTopicView` | `GET /topics` (논제 원본, topic id 사용) |
| 찬반·강경도 | `StanceView` | — |
| 참여 설정 (진행 방식 / 닉네임 / N:N / AI 성향) | `ParamsView` | — |
| 사전 설문 (18문항) | `debate/SurveyForm` | `GET /survey/schema/pre` |
| 토론 전 답변 | `debate/AnswerForm` | — (제출 시점엔 로컬 보관) |
| 토론 | `debate/DebateRoom` | `POST /debate/init`, `POST /debate/{id}/submit` (SSE), `GET /debate/{id}/assistant/{phase}` |
| 토론 후 답변 | `debate/AnswerForm` | — |
| 사후 설문 (33문항) | `debate/SurveyForm` | `POST /sessions/{id}/survey`, `POST /evaluation?topic_id=&session_id=` |
| 결과 | `debate/ResultView` | `GET /debate/{id}/final-report` |

사전 설문은 세션 행이 생기기 전에 받으므로 답변을 들고 있다가 `session` 이벤트로
`record_id` 를 받는 즉시 저장한다 (`DebatePage.handleSessionStart`).

원본 저장소는 마지막 커밋에서 `src/components/debate/` 를 삭제해 `App.jsx` 의 import 가 깨져 있었다.
그 자리를 위 표의 `debate/` 컴포넌트들로 새로 구현했다.

## 백엔드 계약 주의점

- `agent_intensities` 순서는 **상대 진영 AI 먼저, 그 다음 내 진영 AI** (`src/phase0/persona_factory.py` 의 `STANCE_DISTRIBUTION`).
- `debate_format` 은 참여 규모 그대로 `1:1` / `2:2` / `3:3`, AI 수는 각각 1 / 3 / 5명.
- `mode: "debate"` 로 입론·연쇄논박·자유논박까지 진행하고, 전·후 답변 채점은 `/evaluation` 으로 따로 한다.
- `/debate/init` 의 `session` 이벤트가 주는 `record_id` 를 `/evaluation?session_id=` 에 넘겨야
  답변·점수가 수집 테이블의 같은 행에 저장된다.
