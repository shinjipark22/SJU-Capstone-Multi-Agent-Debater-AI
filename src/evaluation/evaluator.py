import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .metrics import METRICS
from .prompts import EVALUATION_PROMPT


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("empty LLM output")

    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_text)
    if fenced:
        raw_text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw_text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise ValueError("JSON object not found")


def _coerce_score(value: Any) -> int:
    try:
        value = float(value)
    except Exception:
        return 0
    value = max(0.0, min(5.0, value))
    return int(round(value))


def _normalize_eval(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    scores = raw_result.get("scores", {}) if isinstance(raw_result, dict) else {}
    normalized_scores: Dict[str, Dict[str, Any]] = {}

    for metric in METRICS:
        metric_key = metric["key"]
        metric_payload = scores.get(metric_key, {}) if isinstance(scores, dict) else {}
        normalized_scores[metric_key] = {
            "label": metric["label"],
            "alias": metric["alias"],
            "score": _coerce_score(metric_payload.get("score", 0)),
            "reason": str(metric_payload.get("reason", "")).strip(),
        }

    average_5 = round(
        sum(item["score"] for item in normalized_scores.values()) / len(METRICS),
        2,
    )
    average_100 = round((average_5 / 5.0) * 100, 1)

    return {
        "scores": normalized_scores,
        "average_5": average_5,
        "average_100": average_100,
        "overall_summary": str(raw_result.get("overall_summary", "")).strip(),
    }


def _default_llm_call(system_prompt: str, user_prompt: str) -> str:
    from .llm import qwen_chat
    return qwen_chat(system_prompt, user_prompt, max_new_tokens=700)


def evaluate_single_answer(
    answer_text: str,
    *,
    topic: str,
    phase_label: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    llm_fn = llm_fn or _default_llm_call

    user_prompt = f"""
[주제]
{topic}

[평가 대상]
{phase_label}

[답변 원문]
{answer_text.strip()}
""".strip()

    raw_output = llm_fn(EVALUATION_PROMPT, user_prompt)
    parsed = _extract_json_object(raw_output)
    normalized = _normalize_eval(parsed)
    normalized["phase_label"] = phase_label
    normalized["raw_text"] = answer_text.strip()
    normalized["llm_raw_output"] = raw_output
    return normalized


def analyze_user_before_after(
    pre_pro: str,
    pre_con: str,
    post_pro: str,
    post_con: str,
    *,
    topic: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    pre_pro_eval  = evaluate_single_answer(pre_pro,  topic=topic, phase_label="토론 전 찬성 답변", llm_fn=llm_fn)
    pre_con_eval  = evaluate_single_answer(pre_con,  topic=topic, phase_label="토론 전 반대 답변", llm_fn=llm_fn)
    post_pro_eval = evaluate_single_answer(post_pro, topic=topic, phase_label="토론 후 찬성 답변", llm_fn=llm_fn)
    post_con_eval = evaluate_single_answer(post_con, topic=topic, phase_label="토론 후 반대 답변", llm_fn=llm_fn)

    return {
        "topic": topic,
        "metrics": METRICS,
        "pro": {
            "pre": pre_pro_eval,
            "post": post_pro_eval,
            "delta_100": round(post_pro_eval["average_100"] - pre_pro_eval["average_100"], 1),
        },
        "con": {
            "pre": pre_con_eval,
            "post": post_con_eval,
            "delta_100": round(post_con_eval["average_100"] - pre_con_eval["average_100"], 1),
        },
    }


def print_evaluation_report(result: Dict[str, Any]) -> None:
    print(f"주제: {result['topic']}")

    for side_key, side_label in [("pro", "찬성"), ("con", "반대")]:
        side = result[side_key]
        print(f"\n{'='*40}")
        print(f"[{side_label}]")
        for phase_key, phase_title in [("pre", "토론 전"), ("post", "토론 후")]:
            phase = side[phase_key]
            print(f"\n  [{phase_title}]")
            for metric in METRICS:
                item = phase["scores"][metric["key"]]
                print(f"  - {metric['label']}: {item['score']}점 | {item['reason']}")
            print(f"  - 평균(5점 만점): {phase['average_5']}")
            print(f"  - 환산 점수(100점 만점): {phase['average_100']}점")
            if phase["overall_summary"]:
                print(f"  - 요약: {phase['overall_summary']}")
        print(
            f"\n  [변화량] 토론 전 {side['pre']['average_100']}점 → "
            f"토론 후 {side['post']['average_100']}점 "
            f"({side['delta_100']:+.1f}점)"
        )


def save_result_json(
    result: Dict[str, Any],
    path: str = "/content/user_before_after_result.json",
) -> None:
    output_path = Path(path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {output_path}")
