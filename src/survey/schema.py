"""schema.py — 설문 문항 정의 로더.

문항 원본은 연구팀 구글폼 4종("토론/구성적 논쟁" × "사전/사후")에서 그대로 추출해
`schema.json` 에 저장했다. 폼이 tech_003 기준으로 작성돼 있어 주제 의존 문구는
플레이스홀더로 바꿔두고, 세션의 실제 주제·모드로 치환해서 내려준다.

    {topic}      → 세션 주제 (따옴표 포함)
    {issue}      → 주제를 가리키는 짧은 표현 ("이 주제")
    {mode_label} → "토론" | "구성적 논쟁"
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

Phase = Literal["pre", "post"]
Mode = Literal["debate", "constructive"]

PHASES: tuple = ("pre", "post")

MODE_LABELS: Dict[str, str] = {"debate": "토론", "constructive": "구성적 논쟁"}

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
_RAW: Dict[str, List[dict]] = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

# CSV 내보내기 컬럼 순서 = 스키마에 정의된 문항 순서
ANSWER_KEYS: Dict[str, List[str]] = {
    phase: [item["key"] for section in sections for item in section["items"]]
    for phase, sections in _RAW.items()
}


def answer_keys(phase: Phase) -> List[str]:
    """해당 단계 문항 key 목록 (스키마 정의 순서)."""
    return ANSWER_KEYS[phase]


def load_schema(phase: Phase, mode: Mode = "debate", topic: str = "") -> List[Dict[str, Any]]:
    """플레이스홀더를 세션 정보로 치환한 설문 스키마를 반환한다."""
    if phase not in _RAW:
        raise KeyError(f"알 수 없는 phase: {phase}")

    replacements = {
        "{topic}": f'"{topic}"' if topic else "이 주제",
        "{issue}": "이 주제",
        "{mode_label}": MODE_LABELS.get(mode, MODE_LABELS["debate"]),
    }

    def fill(text: str) -> str:
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        return text

    sections: List[Dict[str, Any]] = []
    for raw_section in _RAW[phase]:
        section = dict(raw_section)
        if section.get("section"):
            section["section"] = fill(section["section"])
        if section.get("description"):
            section["description"] = fill(section["description"])
        section["items"] = [{**item, "title": fill(item["title"])} for item in raw_section["items"]]
        sections.append(section)
    return sections
