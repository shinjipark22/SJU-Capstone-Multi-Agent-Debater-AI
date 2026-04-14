"""judge용 LLM 호출 — 기존 vLLM 엔드포인트 재사용 (추가 GPU 로드 없음).

원본 ymj-judge는 transformers + bitsandbytes 4bit 로컬 로드였으나,
우리 프로젝트는 이미 vLLM으로 32B AWQ를 서빙 중이므로 HTTP API 재사용한다.
load_model(), load_model_colab() 같은 함수는 noop로 남겨 호환성 유지.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from .config import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_TIMEOUT,
    VLLM_BASE_URL,
)

logger = logging.getLogger(__name__)

_client: Optional[ChatOpenAI] = None
_MAX_RETRIES = 3
_RETRY_DELAY = 3


def _get_client() -> ChatOpenAI:
    """judge 전용 ChatOpenAI 클라이언트 (lazy singleton)."""
    global _client
    if _client is None:
        _client = ChatOpenAI(
            model=JUDGE_MODEL,
            base_url=VLLM_BASE_URL,
            api_key=os.environ.get("LLM_API_KEY", "fake"),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            timeout=JUDGE_TIMEOUT,
        )
        logger.info("[judge/inference] vLLM 클라이언트 초기화: %s (%s)", JUDGE_MODEL, VLLM_BASE_URL)
    return _client


def load_model(*_args, **_kwargs):
    """legacy noop — vLLM이 이미 떠있어야 함."""
    logger.debug("[judge/inference] load_model() noop — vLLM HTTP 사용")


def load_model_colab(*_args, **_kwargs):
    """legacy noop — Colab 전용 로더는 제거됨."""
    logger.debug("[judge/inference] load_model_colab() noop")


def qwen_chat(
    system_prompt: str,
    user_msg: str,
    max_new_tokens: Optional[int] = None,
    assistant_prefix: str = "",
) -> str:
    """vLLM 경유 단일 턴 추론. assistant_prefix는 user msg 끝에 힌트로 append."""
    client = _get_client()
    if max_new_tokens is not None and max_new_tokens != JUDGE_MAX_TOKENS:
        client = client.bind(max_tokens=int(max_new_tokens))

    user_payload = user_msg
    if assistant_prefix:
        user_payload = f"{user_msg}\n\n[응답은 반드시 다음으로 시작]\n{assistant_prefix}"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_payload)]

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            text = content.strip()
            # assistant_prefix로 시작 유도했으면 그 접두어 제거
            if assistant_prefix and text.startswith(assistant_prefix):
                text = text[len(assistant_prefix):].strip()
            return text
        except (APITimeoutError, APIConnectionError) as e:
            logger.warning("[judge/qwen_chat] 연결 오류 시도 %d/%d: %s", attempt, _MAX_RETRIES, e)
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_RETRY_DELAY * attempt)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < _MAX_RETRIES:
                logger.warning("[judge/qwen_chat] 서버 %d 재시도 %d/%d", e.status_code, attempt, _MAX_RETRIES)
                time.sleep(_RETRY_DELAY * attempt)
            else:
                raise
    return ""


def parse_json(raw: str) -> dict:
    """LLM 출력에서 JSON 추출. 코드블록·주변 텍스트 자동 제거."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1)
    m2 = re.search(r"(\{[\s\S]+\})", raw)
    if m2:
        raw = m2.group(1)
    return json.loads(raw)
