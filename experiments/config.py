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
    base_url: str = "http://localhost:8001/v1"          # vLLM 또는 OpenAI 엔드포인트
    api_key_env: str = "LLM_API_KEY"                    # API 키 환경변수명
    gpu_devices: str = "2,3"                            # CUDA_VISIBLE_DEVICES
    port: int = 8001                                    # vLLM 포트
    tensor_parallel: int = 2                            # 텐서 병렬 수
    quantization: Optional[str] = None                  # awq / gptq / None
    max_model_len: int = 16384
    extra_vllm_args: Dict = field(default_factory=dict) # 추가 vLLM 인자


MODELS: Dict[str, ModelConfig] = {
    "GPT-5.4": ModelConfig(
        model_id="GPT-5.4",
        model_name="gpt-5.4",
        model_type="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        gpu_devices="",
    ),
    "Qwen-2.5-32B-Instruct": ModelConfig(
        model_id="Qwen-2.5-32B-Instruct",
        model_name="Qwen/Qwen2.5-32B-Instruct-AWQ",
        base_url="http://localhost:8000/v1",  # 이미 GPU 0,1에서 서빙 중
        gpu_devices="0,1",
        port=8000,
        quantization="awq",
    ),
    "Gemma-2-27b-it": ModelConfig(
        model_id="Gemma-2-27b-it",
        model_name="google/gemma-2-27b-it",
        quantization="awq",
    ),
    "EXAONE-3.5-32B-Instruct": ModelConfig(
        model_id="EXAONE-3.5-32B-Instruct",
        model_name="LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
        quantization="awq",
    ),
    "DeepSeek-R1-Distill-Qwen-14B": ModelConfig(
        model_id="DeepSeek-R1-Distill-Qwen-14B",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ),
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

USER_STANCES: List[str] = ["PRO", "CON"]
DEBATE_FORMATS: List[str] = ["2:2", "3:3"]

FORMAT_INTENSITIES: Dict[str, List[int]] = {
    "2:2": [3, 2, 4],
    "3:3": [3, 2, 4, 3, 2],
}

# 12 topics × 2 stances × 2 formats = 48
EXPERIMENTS_PER_MODEL = len(TOPIC_IDS) * len(USER_STANCES) * len(DEBATE_FORMATS)

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
]

# ── 점수 가중치 ─────────────────────────────────────────────────────────────

WEIGHT_FORMAT = 0.10   # Rule-based 포맷 점수 (0~100 → 1~5 정규화 후)
WEIGHT_LLM = 0.90      # LLM Judge 평균 점수

# ── 승무패 기준 ─────────────────────────────────────────────────────────────

WIN_THRESHOLD = 0.15   # Final Score 차이가 이 값 초과 시 승/패, 이하 시 무승부

# ── vLLM 서버 설정 ──────────────────────────────────────────────────────────

VLLM_STARTUP_TIMEOUT = 300   # 초
VLLM_HEALTH_POLL_INTERVAL = 5  # 초

# ── 사용자 대행 에이전트 ─────────────────────────────────────────────────────

USER_AGENT_MODEL = "gpt-4o-mini"
USER_AGENT_TEMPERATURE = 0       # 재현성 보장
USER_AGENT_SEED = 42             # 고정 시드

# ── 실험 타임아웃 ───────────────────────────────────────────────────────────

SINGLE_EXPERIMENT_TIMEOUT = 900  # 초 (15분)
