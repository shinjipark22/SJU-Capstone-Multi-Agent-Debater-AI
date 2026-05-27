"""
evaluation/llm.py — 평가 모듈용 LLM 호출 래퍼.

기존 토론 시스템과 동일한 vLLM(Qwen2.5-32B-Instruct-AWQ) 인스턴스를 재사용한다.
- src/graph/llm.py 의 langchain ChatOpenAI 클라이언트를 그대로 호출.
- 별도 모델 로드 / 추가 GPU 메모리 / bitsandbytes 의존성 없음.
- 평가 프롬프트는 결정론적 채점이 필요하므로 temperature=0 고정.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
_LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "fake")

# 평가 전용 LLM — 결정론적 (temperature=0) 채점.
_eval_llm: ChatOpenAI | None = None


def _get_eval_llm(max_new_tokens: int) -> ChatOpenAI:
    global _eval_llm
    if _eval_llm is None or _eval_llm.max_tokens != max_new_tokens:
        _eval_llm = ChatOpenAI(
            model=_LLM_MODEL,
            base_url=_VLLM_BASE_URL,
            api_key=_LLM_API_KEY,
            temperature=0.0,
            max_tokens=max_new_tokens,
            top_p=1.0,
            timeout=180,
        )
    return _eval_llm


def qwen_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 700) -> str:
    """
    기존 evaluator.py 가 호출하는 시그니처를 유지한 vLLM 래퍼.
    """
    llm = _get_eval_llm(max_new_tokens)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return (response.content or "").strip()
