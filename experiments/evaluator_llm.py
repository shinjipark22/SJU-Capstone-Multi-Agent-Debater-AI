"""
evaluator_llm.py -- Claude 3.5 Sonnet LLM-as-a-Judge 평가

각 토론 로그를 Claude에게 블라인드로 보내고, 7개 항목(1~5점)으로 평가받는다.
캐시, 재시도, JSON 파싱 안전장치를 포함한다.

사용법:
    python -m experiments.evaluator_llm --input data/logs/Qwen-2.5-32B-Instruct/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

from experiments.config import (
    EVALS_DIR,
    JUDGE_API_KEY_ENV,
    JUDGE_CONCURRENCY,
    JUDGE_MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_RETRY_DELAY,
    LLM_EVAL_CRITERIA,
    LOGS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Claude 클라이언트 ───────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    """Anthropic 클라이언트를 생성한다."""
    api_key = os.environ.get(JUDGE_API_KEY_ENV, "")
    if not api_key:
        raise ValueError(f"환경변수 {JUDGE_API_KEY_ENV}가 설정되지 않았습니다.")
    return anthropic.Anthropic(api_key=api_key)


# ── JSON 안전 파싱 ──────────────────────────────────────────────────────────

def clean_json(text: str) -> str:
    """마크다운 코드 블록으로 감싸진 JSON을 추출한다."""
    # ```json ... ``` 블록 추출
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # { ... } 블록 추출
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return text.strip()


def safe_parse_json(text: str) -> Optional[Dict]:
    """Claude 응답에서 JSON을 안전하게 파싱한다."""
    try:
        cleaned = clean_json(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("JSON 파싱 실패: %s", e)
        return None


# ── 프롬프트 구성 ───────────────────────────────────────────────────────────

def _format_debate_for_judge(log_data: Dict) -> str:
    """토론 로그를 judge용 텍스트로 포맷팅한다. 모델명은 제거(블라인드)."""
    lines = [
        f"[토론 주제] {log_data.get('topic', '')}",
        f"[카테고리] {log_data.get('category', '')}",
        f"[포맷] {log_data.get('debate_format', '')}",
        f"[사용자 입장] {log_data.get('user_stance', '')}",
        "",
        "=" * 60,
        "",
    ]

    for turn in log_data.get("turns", []):
        speaker = turn.get("speaker", "")
        side = turn.get("side", "")
        phase = turn.get("phase", "")
        text = turn.get("text", "")

        # 사용자 턴은 맥락용으로만 짧게 표시 (평가 대상 아님)
        if speaker == "user":
            lines.append(f"[{phase}] 사용자 (발언 생략 — 평가 대상 아님)")
            lines.append("")
            continue

        side_kr = "찬성" if side == "PRO" else "반대"
        lines.append(f"[{phase}] {speaker} ({side_kr})")
        lines.append(text)

        # 검색 도구 사용 여부
        for tc in turn.get("tool_calls", []):
            if tc.get("name") == "search_web":
                lines.append(f"  (검색: {tc.get('query', '')})")

        lines.append("")

    return "\n".join(lines)


JUDGE_SYSTEM_PROMPT = """당신은 AI 토론 시스템의 전문 심판입니다.
주어진 토론 로그에서 **AI 에이전트의 발언만** 평가하세요. 사용자 발언은 평가 대상이 아닙니다.
아래 7개 항목을 각각 1~5점으로 평가하세요.
각 항목에 대해 score(정수)와 reason(한국어, 1~2문장)을 반드시 포함하세요.

평가 항목:
1. self_repetition: 자기 이전 발언과 동일한 논거·근거·사례를 반복하지 않고 새로운 논거를 추가했는가? 같은 주제에 대해 이야기하는 것은 반복이 아니다. 같은 근거나 사례를 다시 쓰는 것만 반복으로 판단하라. (5=매우 다양, 1=같은 논거 반복)
2. team_repetition: 같은 진영 에이전트가 이미 사용한 논거·근거·사례와 다른 것을 제시했는가? 같은 입장(찬성/반대)을 공유하는 것은 반복이 아니다. 같은 근거나 사례를 되풀이하는 것만 반복으로 판단하라. (5=완전히 독립적, 1=같은 논거 되풀이)
3. role_consistency: 각 단계 내에서 자신의 입장을 일관되게 유지했는가? 단, role_reversal(역할반전) 단계에서 입장을 바꾸는 것은 토론 규칙에 의한 의도적 전환이므로 감점하지 마라. (5=완벽한 일관성, 1=비의도적 입장 이탈)
4. persona_tone_toxicity: 전문적 논조를 유지했는가? 욕설/인신공격이 없는가? (5=완벽, 1=심각한 위반)
5. web_search_tool_use: 통계/사건이 필요한 시점에 검색 도구를 적절히 사용했는가? (5=탁월한 활용, 1=전혀 미사용)
6. faithfulness_hallucination_control: 검색 결과를 왜곡 없이 반영했는가? 없는 수치를 만들어냈는가? (5=왜곡 없음, 1=심각한 할루시네이션)
7. logic_evidence_synthesis: 주장-근거-추론(CER) 구조로 논리적으로 엮었는가? (5=탄탄한 논증, 1=사실 나열)
8. korean_language_compliance: 한국어로 자연스럽게 작성했는가? 영어 문장, 외국어 섞임, CoT 유출(<think> 블록), 깨진 문자가 없는가? 고유명사(기관명, 인명)는 영어 허용. (5=완벽한 한국어, 1=외국어 대량 섞임)

반드시 아래 JSON 형식으로만 응답하세요:
{
  "self_repetition": {"score": N, "reason": "..."},
  "team_repetition": {"score": N, "reason": "..."},
  "role_consistency": {"score": N, "reason": "..."},
  "persona_tone_toxicity": {"score": N, "reason": "..."},
  "web_search_tool_use": {"score": N, "reason": "..."},
  "faithfulness_hallucination_control": {"score": N, "reason": "..."},
  "logic_evidence_synthesis": {"score": N, "reason": "..."},
  "korean_language_compliance": {"score": N, "reason": "..."}
}"""


# ── 캐시 ────────────────────────────────────────────────────────────────────

_CACHE_DIR = EVALS_DIR / "llm_cache"


def _cache_key(log_data: Dict) -> str:
    """로그 데이터의 해시를 캐시 키로 사용한다."""
    content = json.dumps(log_data.get("turns", []), ensure_ascii=False, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached(key: str) -> Optional[Dict]:
    """캐시된 평가 결과를 반환한다."""
    cache_path = _CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _set_cache(key: str, result: Dict):
    """평가 결과를 캐시에 저장한다."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{key}.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ── 평가 실행 ───────────────────────────────────────────────────────────────

def evaluate_single_log(client: anthropic.Anthropic, log_data: Dict) -> Optional[Dict]:
    """단일 토론 로그를 Claude로 평가한다. 캐시 + 재시도."""
    cache_key = _cache_key(log_data)
    cached = _get_cached(cache_key)
    if cached:
        logger.info("캐시 히트: %s", log_data.get("topic_id", ""))
        return cached

    debate_text = _format_debate_for_judge(log_data)

    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=2048,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": debate_text}],
            )

            raw_text = response.content[0].text
            parsed = safe_parse_json(raw_text)

            if parsed is None:
                logger.warning("JSON 파싱 실패 (시도 %d/%d)", attempt, JUDGE_MAX_RETRIES)
                continue

            # 필수 항목 검증
            missing = [c for c in LLM_EVAL_CRITERIA if c not in parsed]
            if missing:
                logger.warning("누락 항목 %s (시도 %d/%d)", missing, attempt, JUDGE_MAX_RETRIES)
                continue

            # score 검증 (1~5)
            valid = True
            for c in LLM_EVAL_CRITERIA:
                s = parsed[c].get("score")
                if not isinstance(s, (int, float)) or s < 1 or s > 5:
                    logger.warning("잘못된 점수 %s=%s (시도 %d/%d)", c, s, attempt, JUDGE_MAX_RETRIES)
                    valid = False
                    break
            if not valid:
                continue

            _set_cache(cache_key, parsed)
            return parsed

        except anthropic.RateLimitError:
            wait = JUDGE_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning("Rate limit, %d초 대기", wait)
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.error("API 오류: %s (시도 %d/%d)", e, attempt, JUDGE_MAX_RETRIES)
            time.sleep(JUDGE_RETRY_DELAY * attempt)

    logger.error("최종 실패: %s", log_data.get("topic_id", ""))
    return None


def evaluate_model_logs(model_dir: Path) -> List[Dict]:
    """모델 디렉토리의 모든 로그를 평가한다."""
    client = _get_client()
    results = []

    for log_path in sorted(model_dir.glob("*.json")):
        try:
            with log_path.open(encoding="utf-8") as f:
                log_data = json.load(f)

            scores = evaluate_single_log(client, log_data)
            if scores is None:
                results.append({
                    "file": log_path.name,
                    "model_name": log_data.get("model_name", "unknown"),
                    "topic_id": log_data.get("topic_id", ""),
                    "error": "평가 실패",
                })
                continue

            # 평균 점수 계산
            mean_score = sum(scores[c]["score"] for c in LLM_EVAL_CRITERIA) / len(LLM_EVAL_CRITERIA)

            results.append({
                "file": log_path.name,
                "model_name": log_data.get("model_name", "unknown"),
                "topic_id": log_data.get("topic_id", ""),
                "user_stance": log_data.get("user_stance", ""),
                "debate_format": log_data.get("debate_format", ""),
                "scores": scores,
                "mean_llm_score": round(mean_score, 2),
            })
            logger.info("[%s] %s → 평균 %.2f", log_data.get("model_name"), log_path.name, mean_score)

        except Exception as e:
            logger.error("평가 오류: %s — %s", log_path.name, e)
            results.append({"file": log_path.name, "error": str(e)})

    return results


def _save_results(model_id: str, results: List[Dict]):
    """평가 결과를 JSON으로 저장한다."""
    out_dir = EVALS_DIR / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_id}_llm.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s (%d건)", out_path, len(results))


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge 평가")
    parser.add_argument("--input", type=str, help="모델 로그 디렉토리 또는 'all'")
    args = parser.parse_args()

    if args.input == "all":
        for model_dir in sorted(LOGS_DIR.iterdir()):
            if model_dir.is_dir():
                results = evaluate_model_logs(model_dir)
                _save_results(model_dir.name, results)
    else:
        model_dir = Path(args.input)
        results = evaluate_model_logs(model_dir)
        _save_results(model_dir.name, results)


if __name__ == "__main__":
    main()
