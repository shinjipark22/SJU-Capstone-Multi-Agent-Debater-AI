"""judge 모듈 설정값."""

import os

# 3차원 평가 가중치 (argument 3, evidence 2, language 2 → 총 7)
DIMENSION_WEIGHTS = {"argument": 3, "evidence": 2, "language": 2}
TOTAL_WEIGHT = 7

# live_debate 파이 차트 변환 파라미터
PERCENT_BASELINE = 5.0
PERCENT_STEP = 3.0

# 반박·종합 단계 (평가 컨텍스트가 입론과 다름)
REBUTTAL_PHASES = {"chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"}

# 실시간 분석기는 자유논박까지만 평가한다 (역할반전·종합은 대상 외)
ANALYZED_PHASES = {"opening", "chained_rebuttal", "free_rebuttal"}

# 지수이동평균 평활 계수 (높을수록 최근 발언 가중)
EMA_ALPHA = 0.6

# 최근 컨텍스트 유지 개수
RECENT_WINDOW = 5

# LLM 호출 설정 — 기존 프로젝트 vLLM 재사용 (추가 GPU 리소스 불필요)
VLLM_BASE_URL = os.environ.get("JUDGE_VLLM_BASE_URL") or os.environ.get(
    "VLLM_BASE_URL", "http://localhost:8000/v1"
)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.3"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "300"))
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "60"))
