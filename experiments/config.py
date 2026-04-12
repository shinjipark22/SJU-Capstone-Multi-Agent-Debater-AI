"""
config.py -- A/B 테스트 파이프라인 설정

모델 목록, 실험 조합, 평가 항목, 경로, 가중치 등 전역 설정을 정의한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

# ── 경로 ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TOPICS_PATH = DATA_DIR / "topics_20260323_processed.json"
USER_INPUTS_PATH = PROJECT_ROOT / "tests" / "user_inputs_all_topics.json"

LOGS_DIR = DATA_DIR / "logs"
EVALS_DIR = PROJECT_ROOT / "experiments" / "evals"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

HF_CACHE_DIR = "/disk1/SJ/huggingface/hub"

# ── 모델 설정 ───────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """모델별 서빙/호출 설정."""
    model_id: str                                       # 짧은 식별자 (디렉토리명)
    model_name: str                                     # HuggingFace 모델명 or OpenAI 모델명
    model_type: Literal["vllm", "openai"] = "vllm"     # 서빙 방식
    base_url: str = "http://localhost:8002/v1"          # 실험용 vLLM (8001은 FastAPI)
    api_key_env: str = "LLM_API_KEY"                    # API 키 환경변수명
    gpu_devices: str = "2,3"                            # CUDA_VISIBLE_DEVICES
    port: int = 8002                                    # 실험용 vLLM 포트 (8001은 FastAPI)
    tensor_parallel: int = 2                            # 텐서 병렬 수
    quantization: Optional[str] = None                  # awq / gptq / None
    max_model_len: int = 16384
    extra_vllm_args: Dict = field(default_factory=dict) # 추가 vLLM 인자


MODELS: Dict[str, ModelConfig] = {
    # ── 상한선 (API) ──
    "GPT-5.4": ModelConfig(
        model_id="GPT-5.4",
        model_name="gpt-5.4",
        model_type="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        gpu_devices="",
    ),
    # ── 32B 동체급 ──
    "Qwen-2.5-32B-Instruct": ModelConfig(
        model_id="Qwen-2.5-32B-Instruct",
        model_name="Qwen/Qwen2.5-32B-Instruct-AWQ",
        quantization="awq",
    ),
    "Qwen3-32B": ModelConfig(
        model_id="Qwen3-32B",
        model_name="Qwen/Qwen3-32B-AWQ",
        quantization="awq",
        extra_vllm_args={"is_cot": True},  # CoT 활성화 (thinking 허용)
    ),
    "Gemma-3-27b-it": ModelConfig(
        model_id="Gemma-3-27b-it",
        model_name="pytorch/gemma-3-27b-it-AWQ-INT4",
        quantization="awq",
    ),
    "EXAONE-3.5-32B-Instruct": ModelConfig(
        model_id="EXAONE-3.5-32B-Instruct",
        model_name="LGAI-EXAONE/EXAONE-3.5-32B-Instruct-AWQ",
        quantization="awq",
    ),
    # ── 14B급 ──
    "DeepSeek-R1-Distill-Qwen-14B": ModelConfig(
        model_id="DeepSeek-R1-Distill-Qwen-14B",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        extra_vllm_args={"is_cot": True, "max_tokens_multiplier": 2},
    ),
    # ── 소형 ──
    "Qwen-2.5-7B-Instruct": ModelConfig(
        model_id="Qwen-2.5-7B-Instruct",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    ),
}

# ── 실험 조합 ───────────────────────────────────────────────────────────────

TOPIC_IDS: List[str] = [
    "tech_001", "tech_002", "tech_003",
    "econ_001", "econ_002", "econ_003",
    "poli_001", "poli_002", "poli_003",
    "env_001",  "env_002",  "env_003",
]

DEBATE_FORMATS: List[str] = ["2:2", "3:3"]

# 강경도 프리셋: 같은 포맷이라도 강경도 조합이 다르면 다른 토론이 나온다
# label은 결과 파일명/분석에 사용
INTENSITY_PRESETS: Dict[str, Dict] = {
    "2v2_balanced": {
        "format": "2:2",
        "intensities": [3, 3, 3],
        "label": "균형형",
    },
    "2v2_polarized": {
        "format": "2:2",
        "intensities": [5, 1, 4],
        "label": "극단형",
    },
    "3v3_balanced": {
        "format": "3:3",
        "intensities": [3, 3, 3, 3, 3],
        "label": "균형형",
    },
    "3v3_mixed": {
        "format": "3:3",
        "intensities": [5, 2, 4, 1, 3],
        "label": "혼합형",
    },
}

# 실험 전용 모드: 사용자 없이 전원 AI
USER_STANCE_DEFAULT = "PRO"

# 12 topics × 4 presets = 48
EXPERIMENTS_PER_MODEL = len(TOPIC_IDS) * len(INTENSITY_PRESETS)

# ── LLM Judge 설정 ──────────────────────────────────────────────────────────

JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_API_KEY_ENV = "ANTHROPIC_API_KEY"
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY = 5  # 초 (지수 백오프 기준)
JUDGE_CONCURRENCY = 5  # 동시 API 호출 수

# ── 평가 항목 (LLM Judge) ───────────────────────────────────────────────────

LLM_EVAL_CRITERIA = [
    "self_repetition",
    "team_repetition",
    "role_consistency",
    "persona_tone_toxicity",
    "web_search_tool_use",
    "faithfulness_hallucination_control",
    "logic_evidence_synthesis",
    "korean_language_compliance",
]

# ── 점수 가중치 ─────────────────────────────────────────────────────────────

# 총점 = (Rule 1개 + LLM 8개) / 9  → 모든 항목 균등 가중치
# Rule 점수는 0~100 → 1~5로 정규화 후 LLM 점수와 동일 스케일에서 평균

# ── 승무패 기준 ─────────────────────────────────────────────────────────────

WIN_THRESHOLD = 0.15   # Final Score 차이가 이 값 초과 시 승/패, 이하 시 무승부

# ── vLLM 서버 설정 ──────────────────────────────────────────────────────────

VLLM_STARTUP_TIMEOUT = 300   # 초
VLLM_HEALTH_POLL_INTERVAL = 5  # 초

# ── 사용자 대행 에이전트 ─────────────────────────────────────────────────────

USER_AGENT_MODEL = "gpt-4o-mini"
USER_AGENT_TEMPERATURE = 0       # 재현성 보장
USER_AGENT_SEED = 42             # 고정 시드

# ── 비용 안전장치 ───────────────────────────────────────────────────────────

# API 모델(GPT-5.4) 실험 전 확인 프롬프트 표시
API_COST_CONFIRM = True

# API 모델 실험 수 제한 (실수 방지, 0=무제한)
API_MAX_EXPERIMENTS = 48  # 48개 초과 시 중단

# ── 실험 타임아웃 ───────────────────────────────────────────────────────────

SINGLE_EXPERIMENT_TIMEOUT = 900  # 초 (15분)
