"""
nodes.py — 5단계: 종합 및 재개념화(Synthesis & Reconceptualization) 노드

[동작 흐름]
    1. 전체 토론 히스토리(입론~역할반전)를 요약
    2. 모든 AI 에이전트가 교차 순서로 최적해 의견 제시 (회의 발언)
    3. 사용자와 AI가 최적해에 대해 회의 (자유논박과 유사한 토론)
    4. 사용자가 대표로 "우리의 최적해"를 최종 확정

[설계 노트]
    - 회의식 토론: AI가 먼저 의견을 제시하고, 사용자와 주고받으며 다듬음
    - 사용자가 최종 의사결정자 — "우리의 최적해: ~"로 확정
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.phase1.stage1_opening.nodes as _opening_mod
from src.phase1.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _extract_delimited_text,
    _is_valid_speech,
    _LLM_KWARGS,
)
from src.phase1.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _is_valid_rebuttal,
    build_agent_stance_nums,
)
from src.state import DebateEntry, DebateState

# ── 종합 전용 LLM ──────────────────────────────────────────────────────────
_syn_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 1024, "temperature": 0.7})

# 사용자 발언 의도 추출 전용 LLM (결정적·짧은 출력)
_intent_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 400, "temperature": 0.0})

# 추출 결과 캐시 — 같은 사용자 발언에 대해 여러 에이전트가 호출해도 1회만 LLM 호출
_USER_INTENT_CACHE: Dict[str, dict] = {}


_USER_INTENT_PROMPT = """다음은 토론 중 사용자가 한 발언이다. 이 발언에서 회의 참여자들이 응답해야 할 핵심 요소를 JSON 으로 추출해라.

[토론 주제] {topic}

[사용자 발언]
{user_speech}

[추출 형식 — JSON 만 출력, 다른 텍스트 X]
{{
  "questions": ["사용자가 던진 **명시적 질문** (?, 어떤, 어떻게 등) - 있을 때만, 원문 짧게 인용"],
  "implicit_issues": ["사용자가 명시 질문은 안 했지만 답·구체화를 원하는 implicit 쟁점·요구 (예: 사용자가 '~수준은 안 된다'고 하면 → '그러면 적정 수준은 어떻게 정하는지' 같은 implicit issue)"],
  "claims": ["사용자가 주장한 핵심 입장·주장 1~2개 (짧게)"],
  "acknowledgments": ["사용자가 상대 측에 인정·수긍한 부분 (있을 때만, 짧게)"]
}}

[규칙]
- 없는 항목은 빈 리스트 [].
- 각 항목 최대 60자.
- questions 는 발언에 명시된 것만 (?표·의문사 있는 문장).
- implicit_issues 는 발언에서 자연스럽게 도출되는 후속 질문. 사용자가 어떤 주장·우려를 표명했으면 그것을 해소하려면 무엇이 답해져야 하는지.
- 한국어로.
"""


def _extract_user_intent(user_speech: str, topic: str) -> dict:
    """사용자 발언에서 (명시 질문, implicit 쟁점, 주장, 인정) 을 LLM 으로 1회 추출.

    같은 발언에 대해 모든 에이전트가 같은 추출 결과를 공유 (캐시).
    추출 실패 시 빈 dict 반환 — 호출자는 inject 안 함.
    """
    empty = {"questions": [], "implicit_issues": [], "claims": [], "acknowledgments": []}
    if not user_speech or len(user_speech.strip()) < 10:
        return empty

    cache_key = hashlib.md5(user_speech.encode("utf-8")).hexdigest()
    if cache_key in _USER_INTENT_CACHE:
        return _USER_INTENT_CACHE[cache_key]

    try:
        prompt = _USER_INTENT_PROMPT.format(
            topic=topic, user_speech=user_speech[:1500]
        )
        response: AIMessage = _invoke_with_retry(
            _intent_llm, [HumanMessage(content=prompt)], label="user_intent_extract"
        )
        raw = (response.content or "").strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return empty
        intent = json.loads(match.group(0))
        for key in ("questions", "implicit_issues", "claims", "acknowledgments"):
            v = intent.get(key)
            intent[key] = v if isinstance(v, list) else []
        _USER_INTENT_CACHE[cache_key] = intent
        return intent
    except Exception as e:
        print(f"  [user_intent_extract 실패] {e}")
        return empty


def _format_user_intent_block(intent: dict) -> str:
    """추출된 user intent 를 prompt 에 박을 블록으로 포맷.

    빈 intent 면 빈 문자열 반환 (호출자가 그대로 박아도 무해).
    """
    if not intent:
        return ""
    q = intent.get("questions", [])
    ii = intent.get("implicit_issues", [])
    c = intent.get("claims", [])
    a = intent.get("acknowledgments", [])
    if not (q or ii or c or a):
        return ""

    lines = ["\n[사용자 발언 핵심 — 회의 응답에 반드시 반영]"]
    if q:
        lines.append("- 사용자가 던진 **명시적 질문/요청**:")
        for item in q[:3]:
            lines.append(f"  · {item}")
        lines.append("  → 위 질문은 **첫 문장에서 구체 직답**. 추상 합의로 회피 금지.")
    if ii:
        lines.append("- 사용자 발언이 **답을 요구하는 implicit 쟁점**:")
        for item in ii[:3]:
            lines.append(f"  · {item}")
        lines.append("  → 위 쟁점 중 하나에 자기 perspective 로 **구체 답·예시·조건**을 던져라. \"단계적 도입으로 균형\" 같은 추상 회피 금지.")
    if c:
        lines.append("- 사용자의 핵심 주장/입장:")
        for item in c[:3]:
            lines.append(f"  · {item}")
    if a:
        lines.append("- 사용자가 인정·수긍한 부분:")
        for item in a[:3]:
            lines.append(f"  · {item}")
    return "\n".join(lines) + "\n"


# ── 토론 히스토리 요약 ────────────────────────────────────────────────────

def _summarize_debate(
    history: List[DebateEntry],
    agents: List[Dict],
    speaking_order: List[str],
) -> str:
    """전체 토론 히스토리를 단계별로 요약한다.

    참여자 수에 따라 절단 길이를 자동 조절하여 컨텍스트 초과를 방지한다.
    """
    stance_nums = build_agent_stance_nums(agents, speaking_order)
    num_speakers = len(speaking_order)

    # 참여자 수에 따라 절단 길이 조절 (3:3이면 짧게)
    if num_speakers <= 4:  # 2:2
        len_long, len_short = 200, 150
    elif num_speakers <= 6:  # 3:3
        len_long, len_short = 120, 80
    else:
        len_long, len_short = 80, 60

    def _speaker_display(entry: DebateEntry) -> str:
        if entry["speaker_id"] == "user":
            return "사용자"
        s = entry["stance"]
        label = "찬성" if s == "PRO" else "반대"
        num = stance_nums.get(entry["speaker_id"], 1)
        return f"{label}{num}"

    sections = []

    for phase_name, phase_label in [
        ("opening", "입론"),
        ("chained_rebuttal", "연쇄논박"),
        ("free_rebuttal", "자유논박"),
        ("role_reversal", "역할반전"),
    ]:
        entries = [e for e in history if e["phase"] == phase_name]
        if not entries:
            continue
        max_len = len_long if phase_name in ("opening", "role_reversal") else len_short
        lines = [f"[{phase_label}]"]
        for e in entries:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            lines.append(f"- {_speaker_display(e)} ({stance_kr}): {e['content'][:max_len]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ── 에이전트별 고유 관점 (에코 방지) ───────────────────────────────────────
_AGENT_PERSPECTIVES = [
    "실현 가능성 관점: 이 해결책이 현실에서 실행 가능한지, 비용과 시간은 어떤지 따져라",
    "피해자/수혜자 관점: 이 문제로 누가 가장 피해를 보고, 해결책은 누구에게 이득인지 따져라",
    "장기적 영향 관점: 이 해결책이 5~10년 후에도 유효한지, 부작용은 없는지 따져라",
    "국제 비교 관점: 다른 나라에서 비슷한 문제를 어떻게 해결했는지 사례를 들어라",
    "구조적 원인 관점: 표면적 증상이 아닌 근본 원인이 무엇인지 짚어라",
]


# ── 에이전트별 회의 역할 (perspective 와 다른 축, 단조로움 차단) ──────────
# perspective 는 "어떤 렌즈" 인 반면, 역할은 "회의에서 어떤 기능".
# 두 축을 같이 적용하면 자기 진영 + perspective + 역할 조합으로 발언이 구조적으로 분기.
_AGENT_ROLES: Dict[str, Dict[str, str]] = {
    "bridge_builder": {
        "label": "공감·다리 잇기",
        "desc": (
            "사용자·다른 발언자 의견의 진짜 가치를 짚어주고 자기 진영 입장과 연결해라. "
            "단, 형식적 인정 ('좋은 의견입니다', '맞는 말입니다') 금지 — "
            "구체적으로 어느 부분이 왜 가치 있는지 짚고, 그것을 자기 입장의 어떤 측면과 어떻게 연결하는지 분명히 해라. "
            "**시작 어구 다양화**: '말씀하신 ~' 만 반복하지 말고 아래 풀에서 골라라: "
            "'그 시각은 ~한 점에서 흥미로운데', '~ 부분 짚어보면', '사용자께서 ~ 라고 보셨는데', "
            "'그 지적은 ~ 측면에서 일리가 있는데', '~ 라는 관점은 ~한 면에서 유효한데', "
            "'~ 라고 보시는 부분은 ~한 의미가 있는데'."
        ),
    },
    "implementer": {
        "label": "구체화·실행 제안",
        "desc": (
            "추상 합의·일반론을 구체 실행 방안으로 전환하는 역할이다. "
            "다른 발언자가 '균형이 중요' 라고 끝나면 너는 '그래서 누가·어떤 단계로·어떤 조건에서·어떤 이해관계자가 어떻게' 를 던져라. "
            "발언에 반드시 **단계·이해관계자·실행 조건·예외 시나리오** 중 하나 이상 포함. "
            "**구체 수치 (%, 년수, 비용 등) 는 사용 금지** — 종합 단계는 검색이 없어 수치를 발명하면 환각이다. "
            "이미 알려진 실명 사례 (예: EU AI Act, IBM, OpenAI) 로 구체화하라."
        ),
    },
    "devils_advocate": {
        "label": "반례·검증 도발",
        "desc": (
            "자기 진영을 옹호하되, **너무 쉬운 합의는 의심하라.** "
            "다른 발언자가 '균형'·'단계적 도입' 같은 안전한 결론으로 빠지면 그 가정의 약점을 직접 짚어라. "
            "반례·예외 사례·구현 실패 시나리오 1개를 던져 회의가 안일하게 수렴하는 것을 차단하라. "
            "단, 사용자 진영 자체는 옹호."
        ),
    },
    "brainstormer": {
        "label": "새 각도 브레인스토밍",
        "desc": (
            "지금까지 회의에 안 나온 **새 차원·이해관계자·시나리오**를 제기하는 역할이다. "
            "토픽과 합리적으로 연결되는 범위에서 (무리한 상상 금지) 신선한 angle 1개를 던져라. "
            "이미 다룬 차원 (비용·규제·신뢰 등) 재탕 금지."
        ),
    },
    "synthesizer": {
        "label": "종합·미해결 정리",
        "desc": (
            "지금까지 회의 흐름의 핵심을 짧게 요약 + **아직 답 못 한 쟁점·사용자 질문**을 명시하는 역할이다. "
            "단순 동의로 마무리 X — 미해결 지점을 가시화해 회의가 안일하게 끝나는 것을 막아라."
        ),
    },
}


_ROLE_ASSIGNMENT_BY_COUNT: Dict[int, List[str]] = {
    # 1 AI (1:1 포맷): 정리자 (혼자라 모든 역할 어려움 → 미해결 정리에 집중)
    1: ["synthesizer"],
    # 3 AI (2:2 포맷): 가장 충돌·생산성 좋은 3개 조합
    3: ["bridge_builder", "implementer", "devils_advocate"],
    # 5 AI (3:3 포맷): 5개 역할 다
    5: ["bridge_builder", "implementer", "devils_advocate", "brainstormer", "synthesizer"],
}


def _assign_roles_for_format(num_ai_agents: int) -> List[str]:
    """포맷별 (AI 수별) 역할 키 리스트 반환.

    매핑 누락 시 fallback: perspective 만큼 반복.
    """
    if num_ai_agents in _ROLE_ASSIGNMENT_BY_COUNT:
        return _ROLE_ASSIGNMENT_BY_COUNT[num_ai_agents]
    # fallback: 모든 역할 순환
    keys = list(_AGENT_ROLES.keys())
    return [keys[i % len(keys)] for i in range(num_ai_agents)]


def _build_role_block(role_key: str) -> str:
    """역할 키 → prompt 에 박을 블록 텍스트. 비어있으면 빈 문자열."""
    if not role_key or role_key not in _AGENT_ROLES:
        return ""
    role = _AGENT_ROLES[role_key]
    return (
        f"\n[너의 회의 역할 — {role['label']}]\n"
        f"{role['desc']}\n"
        f"(intensity 가 강하면 강하게, 약하면 부드럽게 — 역할 톤을 intensity 에 맞춰라.)\n"
    )


# ── perspective 별 발언 시작 어구 풀 (mode collapse 회피) ──────────────────
# LLM 이 "양측 의견을 종합하면" 같은 단조 시작에 수렴하는 문제 회피.
# 각 perspective 에 어울리는 다양한 starter 를 prompt 에 박아 발언 다양화 유도.
_PERSPECTIVE_STARTERS = {
    "실현 가능성": ["비용 측면에서 보면", "실행 절차상", "현실적으로", "도입 비용을 따져보면", "운영 단계에서"],
    "피해자/수혜자": ["가장 영향받는 집단은", "수혜자 관점에서는", "누가 부담을 지는지 보면", "이해관계자 입장에서는"],
    "장기적 영향": ["5~10년 후를 보면", "장기적으로", "지속 가능성 측면에서", "시간이 지나면", "후속 세대 입장에서"],
    "국제 비교": ["다른 나라 사례를 보면", "해외에서는", "EU 의 경우", "선진국 사례를 참고하면", "글로벌 추세는"],
    "구조적 원인": ["근본 원인을 짚자면", "구조적으로 보면", "표면이 아닌 본질은", "원인을 거슬러 올라가면", "체계 차원에서는"],
}


def _get_perspective_label(perspective: str) -> str:
    """perspective 문자열 ('실현 가능성 관점: ...') 에서 키 라벨 추출."""
    for key in _PERSPECTIVE_STARTERS:
        if key in perspective:
            return key
    return ""


def _get_starters_for(perspective: str) -> List[str]:
    """perspective 에 매핑된 starter 풀 반환."""
    label = _get_perspective_label(perspective)
    return _PERSPECTIVE_STARTERS.get(label, [])


# 일반 회의 starter (perspective 무관, 모든 perspective 가 사용 가능)
_GENERIC_STARTERS = [
    "정리하자면", "결국", "흥미로운 점은", "한 가지 더하자면",
    "다른 관점에서 보면", "조금 다른 측면에서", "그러게요", "사실",
]


def _extract_speech_starter(speech: str, n_chars: int = 12) -> str:
    """발언의 첫 어구 추출 (시작 어구 중복 회피용)."""
    if not speech:
        return ""
    # ### 헤더 제거 후 첫 줄 첫 n_chars
    lines = [ln.strip() for ln in speech.split("\n") if ln.strip() and not ln.strip().startswith("###")]
    if not lines:
        return ""
    return lines[0][:n_chars].rstrip(",.")


# ── perspective 별 회의 자세 (강경도와 별개 다양화 축) ──────────────────────
# 균일 강경도 (모두 3) 환경에서도 5명의 발언이 distinct 하도록 perspective 차원의
# "회의 자세 + 발언 type + tone" 을 명시. 강경도 의존도 줄여 통제된 실험에서도
# 다양성 보장.
_PERSPECTIVE_STYLES = {
    # 각 perspective 는 진영-agnostic (양 진영 모두 자기 stance 옹호 가능).
    # stance = 회의에서 발언의 접근 방식 (어떤 측면에서 분석하는지) — 한쪽 진영
    # 옹호가 아닌 분석 차원.
    "실현 가능성": {
        "stance": "자기 진영 입장의 실행 가능성·비용·시간·인력 측면을 짚고 단계적 도입 절차를 제안한다",
        "speech_type": "실행 조건 / 도입 절차 / 자원 요구사항 분석",
        "tone": "차분한 분석조 — 숫자·기간·자원 단위로 말한다",
    },
    "피해자/수혜자": {
        "stance": "자기 진영 입장이 누구에게 영향을 주는지 이해관계자·분배 효과를 짚는다 — 어느 쪽 진영이든 자기 입장 지지자가 이득을 본다는 식으로 활용",
        "speech_type": "이해관계자 명시 + 분배 효과 + 영향 분석",
        "tone": "분배 분석조 — '누가 부담하고 누가 이득보는가' 명시",
    },
    "장기적 영향": {
        "stance": "자기 진영 입장의 5~10년 후 영향·지속가능성을 시간 축으로 분석한다",
        "speech_type": "시간 축 분석 + 후속 세대 영향 + 장기 시뮬레이션",
        "tone": "장기 시각 조언조 — '지금은 ~이지만 10년 후엔 ~'",
    },
    "국제 비교": {
        "stance": "자기 진영 입장에 부합하는 해외 사례를 인용하고 한국 적용 가능성을 검토한다",
        "speech_type": "해외 사례 인용 + 비교 + 적용 가능성 평가",
        "tone": "외부 사례 참고조 — '~국가는 ~방식이었습니다'",
    },
    "구조적 원인": {
        "stance": "표면 현상이 아닌 근본 원인·체계 차원에서 자기 진영 입장을 옹호한다",
        "speech_type": "원인 분석 + 본질 지목 + 체계 차원 제안",
        "tone": "본질 추구조 — '문제의 뿌리는 ~', '구조 차원에서는 ~'",
    },
}


def _get_perspective_style(perspective: str) -> Dict[str, str]:
    """perspective 에 매핑된 회의 자세·발언 type·tone 반환."""
    label = _get_perspective_label(perspective)
    return _PERSPECTIVE_STYLES.get(label, {})


def _build_style_block(perspective: str) -> str:
    """[너의 회의 자세] 블록 — perspective 별 unique 자세·발언 type·tone 명시."""
    style = _get_perspective_style(perspective)
    if not style:
        return ""
    label = _get_perspective_label(perspective)
    return (
        f"\n[너의 회의 자세 ({label} 담당)]\n"
        f"- 회의 자세: {style['stance']}\n"
        f"- 발언 type: {style['speech_type']}\n"
        f"- tone: {style['tone']}\n"
        f"이 자세·type·tone 이 너를 다른 발언자와 구분짓는 핵심이다. "
        f"다른 perspective 의 자세·tone 그대로 따라하지 마라.\n"
    )


def _build_starters_block(perspective: str, used_starters: List[str]) -> str:
    """[발언 시작 어구 다양화] 블록 — perspective 별 풀 + 일반 풀 + 사용된 것 차단."""
    label = _get_perspective_label(perspective)
    persp_starters = _get_starters_for(perspective)
    persp_examples = ", ".join(f'"{s}..."' for s in persp_starters[:4]) if persp_starters else ""
    generic_examples = ", ".join(f'"{s}..."' for s in _GENERIC_STARTERS[:6])
    used_block = ""
    if used_starters:
        used_list = "\n".join(f"- \"{s}\"" for s in used_starters)
        used_block = (
            f"\n[이미 이번 회의에서 사용된 시작 어구 — 똑같이 따라하지 마라]\n{used_list}\n"
        )
    persp_line = (
        f"너의 perspective ({label}) 에 어울리는 시작 어구: {persp_examples}\n"
        if persp_examples else ""
    )
    return (
        f"\n[발언 시작 다양화 — 매 AI 가 같은 어구로 시작하면 단조]\n"
        f"{persp_line}"
        f"일반 회의 시작 어구: {generic_examples}\n"
        f"위 풀에서 자연스러운 하나로 시작하라. 직전 발언자와 같은 어구는 피하라."
        f"{used_block}"
    )

# ── 강경도별 타협 태도 (합의 속도 조절) ──────────────────────────────────────
_INTENSITY_NEGOTIATION = {
    1: "상대 의견에 열린 태도를 보이되, 자기 핵심 조건 1개는 반드시 지키면서 구체적 절충안을 제시하라",
    2: "양쪽 의견을 수용하되, 빠진 조건이나 실행 가능성에 대해 구체적으로 질문하라",
    3: "양쪽 장단점을 비교하며, 아직 논의되지 않은 새로운 쟁점을 제기하라",
    4: "자기 원래 입장의 핵심 조건이 빠지면 최적해가 실패한다고 경고하라. 절대 쉽게 동의하지 마라",
    5: "자기 원래 입장을 강하게 고수하며, 상대 제안의 약점을 구체적으로 지적하라. 동의하지 마라",
}


# ── 초기 의견 제시 프롬프트 (회의 오프너) ──────────────────────────────────

def _build_proposal_prompt(
    topic: str,
    original_stance: str,
    perspective: str = "",
    intensity: int = 3,
    prev_speaker: str = "",
    prev_speech: str = "",
    used_starters: Optional[List[str]] = None,
    role_key: str = "",
) -> str:
    """Round 1 회의 오프닝 — 자연 대화 톤으로 의견 제시.

    이전 'mechanical' 한 schema (수용+타협안 강제) 대신, 회의실에서 사람들이
    자연스럽게 발언하듯 직전 발언자에 직접 반응하는 흐름을 유도.
    role_key: 회의 역할 (perspective 와 다른 축). 단조로움 차단.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"

    perspective_block = f"\n[너의 관점] {perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])
    role_block = _build_role_block(role_key)

    prev_block = ""
    if prev_speaker and prev_speech:
        prev_block = (
            f"\n[방금 직전 발언자: {prev_speaker}]\n"
            f"\"{prev_speech[:400]}\"\n"
            f"→ 이 발언에 직접 응답하라. 단순 동의·반대만 말고 회의를 한 발 더 나아가게.\n"
        )

    style_block = _build_style_block(perspective)
    starters_block = _build_starters_block(perspective, used_starters or [])

    return f"""[5단계 — 최적해 도출 회의 (Round 1, 발산 단계)]
'{topic}' 토론이 끝났다. 지금은 회의의 **첫 라운드 — 자기 진영 입장을 자기 perspective 로 분명히 던지는 발산 단계**다.
{prev_block}{perspective_block}{role_block}[너의 협상 태도] {negotiation}
[원래 입장] {stance_kr}
{style_block}{starters_block}

[Round 1 모드 — 자기 입장 주장 (발산)]
- 이번 라운드는 회의의 첫 발언이다. 자기 perspective 의 차원에서 **자기 진영 ({stance_kr}) 입장을 강하게 옹호**하라.
- 상대 진영 의견을 인정하지 마라. 자기 입장의 핵심 측면 1개를 분명히 던져라.
- 양보·합의 표현 금지 ("그래도 ~", "양측 모두 ~" 등). 합의는 다음 라운드에서 다룬다.
- 본인 perspective 의 분석 측면에서 자기 진영을 옹호하는 구체적 한 가지 (조건·우려·사례·이유) 를 명확히 제시.

[수치·통계 금지 — 매우 중요]
- 종합 단계는 검색을 호출하지 않는다. **구체 수치 (%, 년수, 비용·금액, 통계, 인원수) 인용 절대 금지** — 모두 환각이 된다.
- "약 2년", "20% 추가", "30% 절감" 같은 표현 금지. 다른 발언자가 그런 수치를 던졌어도 그대로 차용 금지.
- 대신 **논리·인과·이름 있는 실명 사례 (EU AI Act, IBM, OpenAI 등) ·조건적 추론** 으로 구체화하라.
- 정량적 표현이 꼭 필요하면 "상당한", "장기적으로", "일부", "다수" 같은 일반화 어휘만 허용.

[실명 사례 (회사·기관·법안) 재인용 금지 — entity echo 차단]
- 위 [이미 한 말] / 메시지 체인에서 **다른 발언자가 이미 인용한 실명 사례** (회사·기관·법안·제품명) 는 **재인용 금지**.
- 예: 직전 발언자가 IBM 사례 던졌으면 너는 IBM 다시 쓰지 마라. 다른 entity (Microsoft, Anthropic, OpenAI 등) 또는 자기 perspective 의 새 angle 로 가라.
- 같은 entity 재사용은 인용 echo 라 회의가 단조로워진다.
- 단, 토론 주제 자체에 박힌 핵심 entity (예: 토픽이 'EU AI Act' 라면 EU AI Act) 는 예외.

[사용자 대화 layer — 발산 모드 안에서도 인터랙티브하게]
- 위 메시지 체인의 이전 발언 중 **사용자가 던진 질문·요청·궁금증**이 보이면, 자기 입장 주장 안에 **그 질문을 짚으면서 자기 perspective 로 답하는 흐름**으로 풀어라.
  예: "사용자께서 ~를 물으셨는데, 제 perspective 에서 보면 ~한 측면이 핵심입니다."
- 질문이 없으면 무시. 억지로 만들지 마라. 보통의 입장 주장으로 진행.
- 사용자 발언을 인용할 때는 정확히 그 부분만 짧게 (10~20자) 짚어라. 길게 요약하지 마라.

[발언 가이드]
- 회의실에서 사람들이 말하는 것처럼 자연스럽게. 보고서 작성이 아니라 토론자들과의 대화다.
- 합니다체 유지하되 딱딱한 문어체 X, 회의장 토론 톤.
- 2~4문장. 짧아도 OK, 자연스러우면 풀어도 OK.
- 직전 라운드에서 다른 에이전트가 이미 말한 측면 그대로 반복하지 마라. 새 각도·심화·구체화는 OK.

반드시 아래 형식으로만 출력:

### 반박 시작
(자연스러운 회의 발언 2~4문장)
### 반박 끝"""


# ── 최적해 선언 프롬프트 (Round 3 전원 공통) ──────────────────────────────

def _build_finalize_prompt(
    topic: str,
    original_stance: str,
    perspective: str = "",
    intensity: int = 3,
    prev_speaker: str = "",
    prev_speech: str = "",
    used_starters: Optional[List[str]] = None,
    role_key: str = "",
) -> str:
    """Round 3 — 지금까지 흐름을 종합한 자기 결론.

    이전엔 "제가 생각하는 최적해는 ~입니다" 강제로 5명 모두 같은 시작 → 단조.
    이제 결론 어조는 자유롭게 (단정·제안·요약 형식 다양). 본질은 같음.
    role_key: 회의 역할 (perspective 와 다른 축). 단조로움 차단.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"
    perspective_block = f"\n[너의 관점] {perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])
    role_block = _build_role_block(role_key)

    prev_block = ""
    if prev_speaker and prev_speech:
        prev_block = (
            f"\n[직전 발언자: {prev_speaker}]\n\"{prev_speech[:400]}\"\n"
        )

    style_block = _build_style_block(perspective)
    starters_block = _build_starters_block(perspective, used_starters or [])

    return f"""[5단계 — Round 3: 종결 단계, 자기 perspective 의 최종 결론]
'{topic}' 에 대한 2라운드 동안의 회의 논의를 종합해 **본인 perspective 차원의 최종 결론**을 짧게 정리하라.
{prev_block}{perspective_block}{role_block}[원래 입장] {stance_kr}
[너의 협상 태도] {negotiation}
{style_block}{starters_block}

[Round 3 모드 — perspective-locked 결론]
- Round 1·2 흐름을 종합하되, **본인 perspective 차원의 핵심 메시지**를 명확히 박아라.
- 결론 마지막 문장은 반드시 **본인 perspective 의 angle 로 마무리**:
  · 실현 가능성 → "비용·일정·실행 단계로 볼 때 ~"
  · 피해자/수혜자 → "~ 집단에 미치는 영향이 ~"
  · 장기적 영향 → "5~10년 후를 보면 ~"
  · 국제 비교 → "다른 나라 사례를 보면 ~"
  · 구조적 원인 → "근본 원인은 ~"
- 새 쟁점 꺼내지 마라.

[금지 — 단조로움 차단]
- "균형이 중요" / "양쪽의 균형을 맞추는 것이 가장 이상적" / "조화롭게 발전" / "성능과 안전성 모두 고려" 같은 **추상 합의 결론 금지.**
- "결국 ~이 가장 효과적일 것입니다" 같은 모든 perspective 가 같이 쓸 만한 일반 결론 금지.
- 위 [이미 한 말] 의 결론 어구를 그대로 또 쓰면 실격.

[수치·통계 금지 — 매우 중요]
- 종합 단계는 검색을 호출하지 않는다. **구체 수치 (%, 년수, 비용·금액, 통계, 인원수) 인용 절대 금지** — 모두 환각이 된다.
- 다른 발언자가 그런 수치를 던졌어도 그대로 차용 금지.
- 대신 **논리·인과·이름 있는 실명 사례·조건적 추론** 으로 결론을 박아라.
- "상당한", "장기적으로", "일부", "다수" 같은 일반화 어휘만 허용.

[실명 사례 (회사·기관·법안) 재인용 금지 — entity echo 차단]
- 위 [이미 한 말] / 메시지 체인에서 **다른 발언자가 이미 인용한 실명 사례** (회사·기관·법안·제품명) 는 **재인용 금지**.
- 예: 직전 발언자가 IBM 사례 던졌으면 너는 IBM 다시 쓰지 마라. 다른 entity 또는 자기 perspective 의 새 angle 로 가라.
- 토픽 자체에 박힌 핵심 entity 는 예외.

[결론 작성 규칙 — 매우 중요]
- 마지막 문장은 본인 입장이 어느 쪽인지 명확해야 한다 ({stance_kr} 진영의 자기 perspective 카드).
- 양쪽 다 인정하고 끝내는 합의형 마무리 금지. 자기 입장의 핵심 조건·우선순위 1개를 분명히 박아라.

[발언 가이드]
- 2~4문장. 핵심 결론에 **강조** 하나만.
- 합니다체. 회의 마무리 발언 톤 (보고서 X).

반드시 아래 형식으로만 출력:

### 반박 시작
(2~4문장 결론, perspective-locked 마지막 문장)
### 반박 끝"""


def _build_discuss_prompt(
    topic: str,
    original_stance: str,
    user_latest: str,
    perspective: str = "",
    intensity: int = 3,
    already_said: str = "",
    used_starters: Optional[List[str]] = None,
    user_intent_block: str = "",
    role_key: str = "",
) -> str:
    """Round 2 — 사용자 의견에 자연 대화 톤으로 응답.

    수렴 강제·합의 템플릿 제거. 사용자 발언에 회의 참여자처럼 자연스럽게
    반응하고, intensity 별 협상 태도가 다양성을 만들도록 둔다.
    질문이면 직답, 의견이면 반응 — LLM 이 자연스럽게 판단하게 한다.
    user_intent_block: 미리 추출된 사용자 핵심 (질문/주장/인정) 블록.
    role_key: 회의 역할 (perspective 와 다른 축). 단조로움 차단.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"
    perspective_block = f"\n[너의 관점] {perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])
    role_block = _build_role_block(role_key)

    user_block = ""
    if user_latest:
        user_block = (
            f"\n[방금 사용자 발언 — 직접 응답하라]\n\"{user_latest[:500]}\"\n"
            f"{user_intent_block}"
        )

    style_block = _build_style_block(perspective)
    starters_block = _build_starters_block(perspective, used_starters or [])

    return f"""[5단계 — Round 2: 수렴 단계]
'{topic}' 최적해 회의가 진행 중이다. 방금 사용자가 의견을 던졌다.
{user_block}{perspective_block}{role_block}[원래 입장] {stance_kr}
[너의 협상 태도] {negotiation}
{already_said}
{style_block}{starters_block}
[Round 2 모드 — 수렴 진행 (단, 양보는 협상 태도에 따라)]
- Round 1 에서 자기 입장을 던졌다. 이번 라운드는 **수렴 단계** — 사용자·다른 발언자 의견 중 받아들일 부분을 짚고, 자기 perspective 의 보완·조건을 함께 던져 합의 가능한 접점을 모색하라.
- **단, 협상 태도** ({negotiation}) **를 그대로 반영하라.** 강경하면 강하게 자기 핵심 조건 사수, 부드러우면 더 열린 절충. 모두 같은 톤으로 수렴 금지.

[사용자 대화 layer — 인터랙티브가 핵심]
- 위 [방금 사용자 발언] 에 **질문·요청·궁금증**이 있으면 **첫 문장에서 그 질문에 구체적으로 직답**하라. 직답에 자기 perspective 의 구체 예시·조건·실행 방법을 박아라.
- 질문이 없고 단순 의견이면, 자기 협상 태도 그대로 받아쳐라 (인정·이견·구체화·반례·조건 모두 자유).
- 사용자 발언의 어느 부분에 응답하는지 짧게 짚어주면 인터랙티브함이 산다. 짚는 표현 다양화 — "말씀하신 ~", "그 지적은 ~", "~ 부분은 ~", "그 시각은 ~" 중 한 형태로 자연스럽게.

[perspective-locked 답 — 매우 중요, 단조로움 차단]
- 같은 사용자 질문에 대해 모두 같은 답 하면 회의가 헛돈다. **너의 perspective 차원에서만 답하라.**
  · 실현 가능성 관점 → 비용·일정·실행 단계로 답
  · 피해자/수혜자 관점 → 누가 부담·누가 이득·이해관계자별 영향
  · 장기적 영향 관점 → 5~10년 후의 부작용·성과
  · 국제 비교 관점 → 다른 나라 사례·국제 비교
  · 구조적 원인 관점 → 근본 원인·구조적 조건
- 위 [이미 한 말] 의 다른 perspective 답을 그대로 따라쓰지 마라. 같은 결론·같은 숫자·같은 어구 반복 금지.
- 직전 발언자가 이미 답한 차원 (예: "2년 내 10% 예산") 을 또 반복하면 실격. 너의 perspective 차원으로 다른 답을 던져라.

[피해야 할 패턴 — 회의가 헛도는 원인]
- "단계적 도입으로 균형" 같은 **추상 합의 어구**를 다른 발언자가 이미 썼는데 또 반복.
- "맞는 점은 ~인데 다만 ~" 같은 templated 합의 문장 반복.
- 사용자 질문을 무시하고 자기 입장만 또 늘어놓는 것.
- 구체 예시 없이 "균형이 중요" 류 일반론만 던지는 것.
- 직전 발언자가 제시한 구체 답 (수치·일정·방법) 을 그대로 또 인용하는 것.

[수치·통계 금지 — 매우 중요]
- 종합 단계는 검색을 호출하지 않는다. **구체 수치 (%, 년수, 비용·금액, 통계, 인원수) 인용 절대 금지** — 모두 환각이 된다.
- "약 2년", "20% 추가", "30% 절감" 같은 표현 금지. 다른 발언자가 그런 수치를 던졌어도 **그대로 차용 금지**.
- 대신 **논리·인과·이름 있는 실명 사례 (EU AI Act, IBM, OpenAI 등) ·조건적 추론** 으로 발언을 채워라.
- "상당한", "장기적으로", "일부", "다수" 같은 일반화 어휘만 허용.

[실명 사례 (회사·기관·법안) 재인용 금지 — entity echo 차단]
- 위 [이미 한 말] / 메시지 체인에서 **다른 발언자가 이미 인용한 실명 사례** (회사·기관·법안·제품명) 는 **재인용 금지**.
- 예: 직전 발언자가 IBM 사례 던졌으면 너는 IBM 다시 쓰지 마라. 다른 entity 또는 자기 perspective 의 새 angle 로 가라.
- 토픽 자체에 박힌 핵심 entity 는 예외.

[발언 가이드]
- 회의 토론자처럼 자연스럽게. 2~4문장. 합니다체.
- 위 [이미 한 말] 과 **다른 perspective 차원의 새 답** 으로 가라.

반드시 아래 형식으로만 출력:

### 반박 시작
(자연스러운 회의 응답 2~4문장)
### 반박 끝"""


# ── 멀티턴 메시지 체인 구축 (종합 회의) ───────────────────────────────────

def _build_synthesis_chain(
    agent: Dict,
    history: List,
    speaker_id: str,
    perspective: str = "",
) -> List:
    """종합 회의 히스토리에서 멀티턴 메시지 체인을 구축한다.

    해당 에이전트 발언 → AIMessage, 그 외(사용자+다른 에이전트) → HumanMessage.
    perspective 가 주어지면 system 메시지에 정체성으로 박아 LLM 행동을 강하게 유도.
    """
    persp_label = _get_perspective_label(perspective)
    perspective_identity = ""
    if perspective:
        perspective_identity = (
            f"\n[너의 회의 관점 — 매 발언이 이 관점에서 출발해야 한다]\n"
            f"{perspective}\n"
            f"이 관점이 너를 다른 발언자와 구분짓는 핵심이다. "
            f"다른 관점 (다른 라벨) 의 어구·접근 따라하지 마라."
        )

    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 최적해를 찾는 회의에서 토론자로 참여한다. "
        f"자기 관점을 유지하면서 앞 사람 발언에 자연스럽게 반응하라. "
        f"이전 발언자와 같은 시작 어구·같은 결론 표현 반복 금지. "
        f"2~4문장. 한국어. 합니다체. 회의 톤 (보고체 X)."
        f"{perspective_identity}"
    )
    messages = [SystemMessage(content=system)]

    syn_entries = [e for e in history if e["phase"] == "synthesis"]
    for entry in syn_entries:
        if entry["speaker_id"] == speaker_id:
            messages.append(AIMessage(content=entry["content"]))
        else:
            label = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
            messages.append(HumanMessage(content=f"[{label}] {entry['content']}"))

    return messages


_FALLBACK_MARKER = "이 문제의 핵심을 다시 짚어볼 필요가 있습니다"
_FALLBACK_RESPONSES = [
    "이 해결책의 실현 가능성을 따져보면, 비용과 시간 측면에서 단계적 접근이 필요합니다.",
    "이 문제로 가장 피해를 보는 계층을 우선 고려한 방안이 되어야 합니다.",
    "장기적 관점에서 이 방안이 5년 후에도 유효한지 검토가 필요합니다.",
    "유사한 문제를 해결한 다른 국가의 사례를 참고하면 도움이 될 것입니다.",
    "표면적 증상이 아닌 구조적 원인에 집중한 해결책이 필요합니다.",
]
_fallback_idx = 0


def _generate_with_synthesis_chain(
    agent: Dict,
    prompt: str,
    chain: List = None,
    topic: str = "",
) -> Tuple[str, str, List]:
    """멀티턴 체인에 새 프롬프트를 추가하고 생성한다. 빈 응답 시 최대 2회 재시도."""
    messages = list(chain) if chain else []
    messages.append(HumanMessage(content=prompt))

    for attempt in range(3):
        response: AIMessage = _invoke_with_retry(_syn_llm, messages, label=f"synthesis_attempt{attempt}")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

        # 유효하고 fallback 문장이 아니면 사용
        if _is_valid_rebuttal(speech) and _FALLBACK_MARKER not in speech:
            return speech, raw, []

        logger.warning("[synthesis] speech 무효 또는 fallback, 재시도 %d/3", attempt + 1)
        if attempt < 2:
            messages.append(HumanMessage(content="이전 응답이 부적절합니다. 한국어로 1~2문장, 구체적인 의견을 말하세요."))

    if not _is_valid_rebuttal(speech):
        global _fallback_idx
        speech = _FALLBACK_RESPONSES[_fallback_idx % len(_FALLBACK_RESPONSES)]
        _fallback_idx += 1

    return speech, raw, []


# ── 메인 노드: 초기 의견 제시 ──────────────────────────────────────────────

def _ai_speakers(state: DebateState) -> List[str]:
    """speaking_order 에서 AI 만 (사용자 제외)."""
    return [s for s in state["speaking_order"] if s != "user"]


def _current_synthesis_round_speeches(state: DebateState) -> List[str]:
    """현재 진행 중인 synthesis 라운드에서 이미 말한 AI 발화 (앞 80자) 리스트.

    라운드 = 사용자 1턴 + AI N턴. 마지막 사용자 발언 이후의 AI 발언만 수집.
    """
    history = state.get("debate_history", [])
    last_user_idx = -1
    for i, e in enumerate(history):
        if e["phase"] == "synthesis" and e["speaker_id"] == "user":
            last_user_idx = i
    speeches = []
    for e in history[last_user_idx + 1:]:
        if e["phase"] == "synthesis" and e["speaker_id"] != "user":
            speeches.append(e["content"][:80])
    return speeches


# ── 결론 어구 반복 차단 (Round 2/3 단조로움 가드) ─────────────────────────────


def _extract_overused_phrases(speeches: List[str], min_count: int = 2, n: int = 4) -> List[str]:
    """직전 발언들에서 **여러 발언에 겹쳐서 등장한** 한국어 n-gram 어구를 추출.

    하드코딩된 키워드 없이 텍스트만 보고 자주 등장한 substring 을 찾는다.
    prompt 에 ban list 로 박아 모델이 같은 표현으로 결론 짓지 않도록 한다.

    speeches: 같은 라운드 내 다른 에이전트의 발언 (앞 80자 정도)
    n: 윈도우 크기 (글자 수). 6글자 정도면 의미 있는 어구 (예: "단계적 도입")
    min_count: 이 값 이상 등장해야 ban list 포함
    """
    if len(speeches) < 2:
        return []
    # 각 speech 마다 등장한 n-gram set 을 계산하고, 등장한 speech 수를 count
    appears_in: Dict[str, int] = {}
    for s in speeches:
        # 한국어 자모/공백 정규화 후 n-gram
        text = re.sub(r"[^가-힣ㄱ-ㆎ ]", "", s)
        seen = set()
        for i in range(len(text) - n + 1):
            chunk = text[i : i + n].strip()
            # 의미 있는 어구만 (공백 너무 많은 것 제외)
            if len(chunk.replace(" ", "")) < n - 1:
                continue
            seen.add(chunk)
        for chunk in seen:
            appears_in[chunk] = appears_in.get(chunk, 0) + 1
    # min_count 이상 발언에 등장한 어구
    overused = [c for c, cnt in appears_in.items() if cnt >= min_count]
    # 길이 정렬, 중복 substring 정리 (긴 것 우선)
    overused.sort(key=len, reverse=True)
    dedup: List[str] = []
    for p in overused:
        if not any(p in d for d in dedup):
            dedup.append(p)
        if len(dedup) >= 6:
            break
    return dedup


def synthesis_propose_one_node(state: DebateState) -> DebateState:
    """초기 의견 제시 — 한 AI 에이전트만 발언. synthesis_propose_idx 카운터.

    노드 분리 이유: 각 에이전트 발화가 별도 LangGraph chunk → SSE 즉시 push.
    """
    speakers = _ai_speakers(state)
    idx = state.get("synthesis_propose_idx", 0)
    if idx >= len(speakers):
        return DebateState(**{**state, "phase": "synthesis"})

    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])

    speaker_id = speakers[idx]
    agent = agent_map[speaker_id]
    slabel = "찬성" if agent["stance"] == "PRO" else "반대"
    snum = stance_nums.get(speaker_id, 1)
    display = f"{slabel}{snum}"

    perspective = _AGENT_PERSPECTIVES[idx % len(_AGENT_PERSPECTIVES)]
    role_keys = _assign_roles_for_format(len(speakers))
    role_key = role_keys[idx % len(role_keys)] if role_keys else ""
    role_label = _AGENT_ROLES.get(role_key, {}).get("label", "")
    intensity = agent.get("intensity", 3)

    print(f"  [{display}] 의견 제시 중... (관점: {perspective[:20]}, 역할: {role_label}, 강경도: {intensity})")

    # 직전 발언자 (회의 흐름의 자연성) — synthesis phase 의 history 에서 마지막 비-self 발언
    prev_speaker_id = ""
    prev_speech_text = ""
    for e in reversed(history):
        if e.get("phase") != "synthesis":
            continue
        if e.get("speaker_id") != speaker_id:
            prev_speaker_id = e.get("speaker_id", "")
            prev_speech_text = e.get("content", "")
            break

    # 이미 사용된 시작 어구 (synthesis 전체에서) — 다양화 강제용
    used_starters = [
        _extract_speech_starter(e["content"])
        for e in history
        if e.get("phase") == "synthesis" and e.get("content")
    ]
    used_starters = [s for s in used_starters if s]

    prompt = _build_proposal_prompt(
        topic=topic, original_stance=agent["stance"],
        perspective=perspective, intensity=intensity,
        prev_speaker=prev_speaker_id, prev_speech=prev_speech_text,
        used_starters=used_starters, role_key=role_key,
    )
    # 같은 라운드에서 다른 에이전트가 이미 한 말 (history 에서 추출)
    this_round = _current_synthesis_round_speeches(state)
    if this_round:
        already_said = (
            "\n[이번 라운드에서 이미 나온 의견 — 그대로 반복 X, 새 측면으로 가라]\n"
            + "\n".join(f"- {s}" for s in this_round)
            + "\n"
        )
        prompt += f"\n{already_said}"

    from src.graph.llm import build_debate_chain
    debate_chain = build_debate_chain(history, speaker_id)
    chain = _build_synthesis_chain(agent, history, speaker_id, perspective=perspective)
    full_chain = debate_chain + [m for m in chain if m not in debate_chain]
    speech, raw, _logs = _generate_with_synthesis_chain(agent, prompt, full_chain, topic)

    entry = DebateEntry(
        turn=current_turn,
        speaker_id=speaker_id,
        stance=agent["stance"],
        phase="synthesis",
        content=speech,
        target_id=None,
        tool_calls_log=[],
        json_raw=raw,
    )
    history.append(entry)
    current_turn += 1
    print(f"  [{display}] {speech[:80]}...\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "synthesis_propose_idx": idx + 1,
        "phase": "synthesis",
        "is_finished": False,
    })


def synthesis_node(state: DebateState) -> DebateState:
    """[legacy] 모든 AI 가 한 번에 초기 의견 제시. 분리 전 호출자 호환용.

    회의 시작: 각 에이전트가 2~3문장으로 의견. 이후는 synthesis_discuss_node 로.
    """
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    print(f"\n[5단계: 최적해 회의] 초기 의견 제시\n")

    # 역할 배정 — AI 수 기반 (user 제외)
    num_ai = sum(1 for sid in speaking_order if sid != "user")
    role_keys = _assign_roles_for_format(num_ai)

    agent_idx = 0
    this_round_speeches: List[str] = []  # 이번 라운드에서 다른 에이전트가 한 말 수집
    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        perspective = _AGENT_PERSPECTIVES[agent_idx % len(_AGENT_PERSPECTIVES)]
        role_key = role_keys[agent_idx % len(role_keys)] if role_keys else ""
        role_label = _AGENT_ROLES.get(role_key, {}).get("label", "")
        agent_idx += 1
        intensity = agent.get("intensity", 3)

        print(f"  [{display}] 의견 제시 중... (관점: {perspective[:20]}, 역할: {role_label}, 강경도: {intensity})")

        prompt = _build_proposal_prompt(
            topic=topic,
            original_stance=agent["stance"],
            perspective=perspective,
            intensity=intensity,
            role_key=role_key,
        )
        # 이전 에이전트가 이미 한 말 추가
        if this_round_speeches:
            already_said = (
                "\n[다른 참여자가 이미 한 말 — 같은 내용 반복 금지]\n"
                + "\n".join(f"- {s}" for s in this_round_speeches)
                + "\n위와 완전히 다른 관점에서 발언하라.\n"
            )
            prompt += f"\n{already_said}"

        # 전체 토론 히스토리를 메시지 체인으로 전달 (_summarize_debate 대체)
        from src.graph.llm import build_debate_chain
        debate_chain = build_debate_chain(history, speaker_id)
        chain = _build_synthesis_chain(agent, history, speaker_id, perspective=perspective)
        # debate_chain(1~4단계) + synthesis_chain(5단계) 합치기
        full_chain = debate_chain + [m for m in chain if m not in debate_chain]
        speech, raw, _logs = _generate_with_synthesis_chain(agent, prompt, full_chain, topic)
        this_round_speeches.append(speech[:80])  # 다음 에이전트가 참고

        entry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="synthesis",
            content=speech,
            target_id=None,
            tool_calls_log=[],
            json_raw=raw,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{display}] {speech[:80]}...\n")

    print(f"[5단계: 최적해 회의] 초기 의견 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "synthesis",
        "is_finished": False,
    })


# ── 회의 응답 노드 (사용자 발언 후 AI 반응) ──────────────────────────────

def synthesis_discuss_one_node(state: DebateState) -> DebateState:
    """사용자 발언에 대한 AI 응답 — 한 에이전트만. synthesis_discuss_idx 카운터.

    노드 분리 이유: 발화 단위로 LangGraph chunk yield → SSE 즉시 push.
    """
    speakers = _ai_speakers(state)
    idx = state.get("synthesis_discuss_idx", 0)
    if idx >= len(speakers):
        return DebateState(**{**state, "phase": "synthesis", "is_finished": False})

    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])

    is_final_round = state.get("synthesis_user_turns", 0) >= 2
    user_messages = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "synthesis"]
    user_latest = user_messages[-1]["content"] if user_messages else ""

    speaker_id = speakers[idx]
    agent = agent_map[speaker_id]
    slabel = "찬성" if agent["stance"] == "PRO" else "반대"
    snum = stance_nums.get(speaker_id, 1)
    display = f"{slabel}{snum}"
    perspective = _AGENT_PERSPECTIVES[idx % len(_AGENT_PERSPECTIVES)]
    role_keys = _assign_roles_for_format(len(speakers))
    role_key = role_keys[idx % len(role_keys)] if role_keys else ""
    role_label = _AGENT_ROLES.get(role_key, {}).get("label", "")
    intensity = agent.get("intensity", 3)
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

    print(f"  [{display}] 응답 중... (역할: {role_label}, 강경도: {intensity})")

    # 같은 라운드에서 이미 한 말 (history 에서 추출) + 반복 어구 ban list
    this_round = _current_synthesis_round_speeches(state)
    already_said = ""
    if this_round:
        already_said = (
            "\n[이번 라운드에서 다른 참여자가 이미 한 말 — 같은 내용·표현 반복 금지]\n"
            + "\n".join(f"- {s}" for s in this_round)
            + "\n"
        )

    # ban list — synthesis 전체 발언 (라운드 간 누적) 대상으로 추출.
    # 같은 라운드만 보면 라운드 1에서 쓴 어구가 라운드 3에서 또 나옴.
    all_synthesis_speeches = [
        e["content"][:200]
        for e in history
        if e.get("phase") == "synthesis" and e.get("speaker_id") != "user"
    ]
    overused = _extract_overused_phrases(all_synthesis_speeches)
    if overused:
        already_said += (
            "\n[종합 회의 누적 — 여러 발언에 이미 등장한 어구. 그대로 쓰면 실격]\n"
            + ", ".join(f'"{p}"' for p in overused)
            + "\n→ 위 어구로 결론 짓거나 같은 합의 톤으로 마무리하지 마라. 다른 단어·다른 각도로 가라.\n"
        )

    from src.graph.llm import build_debate_chain
    debate_chain = build_debate_chain(history, speaker_id)

    # 직전 발언자 (Round 3 finalize 에서 직전 발언 비추기용)
    prev_speaker_id = ""
    prev_speech_text = ""
    for e in reversed(history):
        if e.get("phase") != "synthesis":
            continue
        if e.get("speaker_id") != speaker_id:
            prev_speaker_id = e.get("speaker_id", "")
            prev_speech_text = e.get("content", "")
            break

    # 이미 사용된 시작 어구 (synthesis 전체) — 다양화 강제
    used_starters = [
        _extract_speech_starter(e["content"])
        for e in history
        if e.get("phase") == "synthesis" and e.get("content")
    ]
    used_starters = [s for s in used_starters if s]

    # 사용자 발언에서 (질문/주장/인정) 추출 — 같은 user_latest 에 대해 캐시되므로
    # 이 노드가 5번 호출되어도 LLM 호출은 1번뿐.
    user_intent = _extract_user_intent(user_latest, topic) if user_latest else {}
    user_intent_block = _format_user_intent_block(user_intent)

    if is_final_round:
        prompt = _build_finalize_prompt(
            topic, agent["stance"], perspective=perspective, intensity=intensity,
            prev_speaker=prev_speaker_id, prev_speech=prev_speech_text,
            used_starters=used_starters, role_key=role_key,
        )
        if user_intent_block:
            prompt += "\n" + user_intent_block
        if already_said:
            prompt += already_said
    else:
        prompt = _build_discuss_prompt(
            topic=topic, original_stance=agent["stance"],
            user_latest=user_latest, perspective=perspective, intensity=intensity,
            already_said=already_said, used_starters=used_starters,
            user_intent_block=user_intent_block, role_key=role_key,
        )
    speech, raw, _logs = _generate_with_synthesis_chain(agent, prompt, debate_chain, topic)

    entry = DebateEntry(
        turn=current_turn,
        speaker_id=speaker_id,
        stance=agent["stance"],
        phase="synthesis",
        content=speech,
        target_id="user",
        tool_calls_log=[],
        json_raw=raw,
    )
    history.append(entry)
    current_turn += 1
    print(f"  [{display}] {speech[:80]}...\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "synthesis_discuss_idx": idx + 1,
        "phase": "synthesis",
        "is_finished": False,
    })


def synthesis_discuss_node(state: DebateState) -> DebateState:
    """[legacy] 모든 AI 가 한 번에 응답 — 분리 전 호출자 호환용.

    Round 3(synthesis_user_turns >= 2)에는 모든 에이전트가 최적해를 선언한다.
    """
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    # Round 3 여부: user가 2턴 완료한 상태 → 이번 AI 발언이 마지막 라운드
    is_final_round = state.get("synthesis_user_turns", 0) >= 2

    # 사용자 최근 발언
    user_messages = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "synthesis"]
    user_latest = user_messages[-1]["content"] if user_messages else ""

    phase_label = "최적해 선언 (Round 3)" if is_final_round else "AI 응답 생성"
    print(f"\n[5단계: 최적해 회의] {phase_label} 중...\n")

    # 역할 배정 — AI 수 기반 (user 제외)
    num_ai = sum(1 for sid in speaking_order if sid != "user")
    role_keys = _assign_roles_for_format(num_ai)

    # 사용자 발언 의도 추출 (라운드 시작 시 1회, 캐시됨)
    user_intent = _extract_user_intent(user_latest, topic) if user_latest else {}
    user_intent_block = _format_user_intent_block(user_intent)

    # synthesis 전체 발언 (라운드 누적 ban list 용)
    all_synthesis_speeches = [
        e["content"][:200]
        for e in history
        if e.get("phase") == "synthesis" and e.get("speaker_id") != "user"
    ]
    overused = _extract_overused_phrases(all_synthesis_speeches)

    agent_idx = 0
    this_round_speeches: List[str] = []  # 이번 라운드에서 다른 에이전트가 한 말 수집
    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        perspective = _AGENT_PERSPECTIVES[agent_idx % len(_AGENT_PERSPECTIVES)]
        role_key = role_keys[agent_idx % len(role_keys)] if role_keys else ""
        role_label = _AGENT_ROLES.get(role_key, {}).get("label", "")
        agent_idx += 1
        intensity = agent.get("intensity", 3)

        print(f"  [{display}] 응답 중... (역할: {role_label}, 강경도: {intensity})")

        # 이번 라운드에서 다른 에이전트가 이미 말한 내용을 프롬프트에 포함
        already_said = ""
        if this_round_speeches:
            already_said = (
                "\n[이번 라운드에서 다른 참여자가 이미 한 말 — 같은 내용 반복 금지]\n"
                + "\n".join(f"- {s}" for s in this_round_speeches)
                + "\n"
            )
        if overused:
            already_said += (
                "\n[종합 회의 누적 — 여러 발언에 이미 등장한 어구. 그대로 쓰면 실격]\n"
                + ", ".join(f'"{p}"' for p in overused)
                + "\n→ 위 어구로 결론 짓거나 같은 합의 톤으로 마무리하지 마라.\n"
            )

        # 전체 토론 히스토리 + 종합 회의 체인
        from src.graph.llm import build_debate_chain
        debate_chain = build_debate_chain(history, speaker_id)

        if is_final_round:
            prompt = _build_finalize_prompt(
                topic, agent["stance"], perspective=perspective, intensity=intensity,
                role_key=role_key,
            )
            if user_intent_block:
                prompt += "\n" + user_intent_block
            if already_said:
                prompt += already_said
        else:
            prompt = _build_discuss_prompt(
                topic=topic, original_stance=agent["stance"],
                user_latest=user_latest, perspective=perspective, intensity=intensity,
                already_said=already_said,
                user_intent_block=user_intent_block, role_key=role_key,
            )
        speech, raw, _logs = _generate_with_synthesis_chain(agent, prompt, debate_chain, topic)
        this_round_speeches.append(speech[:80])  # 다음 에이전트가 참고할 수 있도록 수집

        entry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="synthesis",
            content=speech,
            target_id="user",
            tool_calls_log=[],
            json_raw=raw,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{display}] {speech[:80]}...\n")

    print(f"[5단계: 최적해 회의] AI 응답 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "synthesis",
        "is_finished": False,
    })


# ── 턴 라우터 ──────────────────────────────────────────────────────────────

def should_end_synthesis(state: DebateState) -> bool:
    """종합 회의 종료 조건: 사용자 3턴 완료."""
    return state.get("synthesis_user_turns", 0) >= 3
