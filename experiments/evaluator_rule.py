"""
evaluator_rule.py -- 정규식 기반 포맷 준수율 평가

각 토론 로그 JSON 전체를 대상으로 마크다운 헤더 존재 여부,
한국어 비율, 문장 끊김 등을 검사하여 0~100점을 산출한다.

사용법:
    python -m experiments.evaluator_rule --input data/logs/Qwen-2.5-32B-Instruct/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.config import EVALS_DIR, LOGS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 스테이지별 필수 헤더 ────────────────────────────────────────────────────

REQUIRED_HEADINGS = {
    "opening": {
        "agent": [
            r"###\s*자기소개",
            r"###\s*논거\s*\[?\s*1\s*\]?",
            r"###\s*논거\s*\[?\s*2\s*\]?",
            r"###\s*결론",
        ],
        "user": [
            r"###\s*논거\s*\[?\s*1\s*\]?",
            r"###\s*논거\s*\[?\s*2\s*\]?",
            r"###\s*결론",
        ],
    },
    "role_reversal": {
        "agent": [
            r"###\s*논거\s*\[?\s*1\s*\]?",
            r"###\s*논거\s*\[?\s*2\s*\]?",
            r"###\s*결론",
        ],
        "user": [],
    },
}


def _check_headings(text: str, patterns: List[str]) -> Tuple[int, int]:
    """필수 헤더 존재 여부. (존재 수, 전체 수) 반환."""
    if not patterns:
        return 0, 0
    found = sum(1 for p in patterns if re.search(p, text))
    return found, len(patterns)


def _check_template_copy(text: str) -> bool:
    """프롬프트 템플릿을 그대로 복사한 경우를 감지한다."""
    template_markers = [
        r'###\s*논거\s*\d+\s*:\s*소제목\s*$',  # "### 논거 1: 소제목" 그대로
        r'\(자기소개와 입장\)',
        r'\(논거\)',
        r'\(결론\)',
    ]
    for p in template_markers:
        if re.search(p, text, re.MULTILINE):
            return True
    return False


def _check_language_quality(text: str) -> Dict[str, float]:
    """한국어 비율, CoT 유출, 깨진 문자 등을 검사한다."""
    korean = len(re.findall(r'[가-힣]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    total = korean + english
    korean_ratio = korean / total if total > 0 else 0.0

    cot_leaked = bool(re.search(
        r'\b(?:First|Second|Let me|I need|In order to|Okay,?\s+so)\b',
        text, re.IGNORECASE,
    ))

    broken_chars = bool(re.search(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uff60。，]', text))

    # 마지막 줄 끊김 (숫자/영어/조사로 끝나면 끊긴 것)
    lines = text.rstrip().split('\n')
    last = lines[-1].strip() if lines else ""
    truncated = (
        bool(last) and not last.startswith('###') and len(last) > 10
        and not re.search(r'[.?!다까요)\*"—]$', last)
    )

    return {
        "korean_ratio": round(korean_ratio, 3),
        "cot_leaked": cot_leaked,
        "broken_chars": broken_chars,
        "truncated": truncated,
    }


def evaluate_single_log(log_path: Path) -> Dict:
    """단일 토론 로그 JSON을 평가하여 점수를 반환한다."""
    with log_path.open(encoding="utf-8") as f:
        data = json.load(f)

    turns = data.get("turns", [])
    if not turns:
        return {"file": log_path.name, "format_score": 0, "details": "빈 로그"}

    total_heading_found = 0
    total_heading_required = 0
    language_issues = 0
    tool_usage_count = 0
    entry_details = []

    for turn in turns:
        phase = turn.get("phase", "")
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")

        # 전원 AI 모드: 모든 턴 평가

        # 헤더 검사
        patterns = REQUIRED_HEADINGS.get(phase, {}).get("agent", [])
        found, total = _check_headings(text, patterns)
        total_heading_found += found
        total_heading_required += total

        # 언어 품질
        lang = _check_language_quality(text)
        if lang["cot_leaked"] or lang["broken_chars"] or lang["truncated"]:
            language_issues += 1
        if lang["korean_ratio"] < 0.3:
            language_issues += 1

        # 템플릿 복사 감지 ("### 논거 1: 소제목" 그대로 쓴 경우)
        is_template = _check_template_copy(text)
        if is_template:
            language_issues += 2

        # 격식체 — Rule에서 제외 (프롬프트 한계, LLM judge에서 평가)
        informal_count = len(re.findall(
            r'거든요|잖아요|인데요|네요[.]|[가-힣]야\s|해요[.]|같아요|어요[.]|죠[.]',
            text
        ))

        # delimiter 유출 (### 답변 시작/끝, ### 반박 시작/끝이 출력에 남음)
        if re.search(r'답변\s*시작|답변\s*끝|반박\s*시작|반박\s*끝', text):
            language_issues += 1

        # 볼드 깨짐 (** 열고 안 닫음)
        if text.count('**') % 2 != 0:
            language_issues += 1

        # 소제목 번호 중복 (### 논거 1이 2번)
        heading_nums = re.findall(r'###\s*논거\s*(\d+)', text)
        if heading_nums and len(heading_nums) != len(set(heading_nums)):
            language_issues += 1

        # 도구 사용
        tc_count = len(turn.get("tool_calls", []))
        tool_usage_count += tc_count

        entry_details.append({
            "turn_id": turn.get("turn_id"),
            "phase": phase,
            "speaker": speaker,
            "heading_found": found,
            "heading_required": total,
            "korean_ratio": lang["korean_ratio"],
            "cot_leaked": lang["cot_leaked"],
            "broken_chars": lang["broken_chars"],
            "truncated": lang["truncated"],
            "template_copy": is_template,
            "informal_count": informal_count,
            "delimiter_leaked": bool(re.search(r'답변\s*시작|답변\s*끝|반박\s*시작|반박\s*끝', text)),
            "bold_broken": text.count('**') % 2 != 0,
            "heading_num_dup": len(heading_nums) != len(set(heading_nums)) if heading_nums else False,
            "tool_calls": tc_count,
            "char_count": len(text),
        })

    # 점수 계산
    # 1. 헤더 준수율 (0~50점)
    heading_score = (total_heading_found / total_heading_required * 50) if total_heading_required > 0 else 50

    # 2. 언어 품질 (0~30점) — 이슈 당 -5점
    lang_score = max(0, 30 - language_issues * 5)

    # 3. 구조 완성도 (0~10점) — 5개 phase 존재 여부
    phases_present = set(t.get("phase") for t in turns)
    expected_phases = {"opening", "chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"}
    structure_score = len(phases_present & expected_phases) / len(expected_phases) * 10

    # 4. 도구 활용 (0~10점) — 1회 이상 사용 시 기본 5점 + 3회 이상 10점
    tool_score = min(10, tool_usage_count * 3) if tool_usage_count > 0 else 0

    final_score = round(heading_score + lang_score + structure_score + tool_score, 1)
    final_score = min(100, max(0, final_score))

    return {
        "file": log_path.name,
        "model_name": data.get("model_name", "unknown"),
        "topic_id": data.get("topic_id", ""),
        "user_stance": data.get("user_stance", ""),
        "debate_format": data.get("debate_format", ""),
        "format_score": final_score,
        "heading_score": round(heading_score, 1),
        "language_score": round(lang_score, 1),
        "structure_score": round(structure_score, 1),
        "tool_score": round(tool_score, 1),
        "total_turns": len(turns),
        "tool_usage_count": tool_usage_count,
        "per_turn_details": entry_details,
    }


def evaluate_model_logs(model_dir: Path) -> List[Dict]:
    """모델 디렉토리의 모든 로그를 평가한다."""
    results = []
    for log_path in sorted(model_dir.glob("*.json")):
        try:
            result = evaluate_single_log(log_path)
            results.append(result)
            logger.info("[%s] %s → %.1f점", result["model_name"], result["file"], result["format_score"])
        except Exception as e:
            logger.error("평가 실패: %s — %s", log_path.name, e)
            results.append({"file": log_path.name, "format_score": 0, "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Rule-based 포맷 평가")
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


def _save_results(model_id: str, results: List[Dict]):
    """평가 결과를 JSON으로 저장한다."""
    out_dir = EVALS_DIR / "rule"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_id}_rule.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s (%d건)", out_path, len(results))


if __name__ == "__main__":
    main()
