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
) -> str:
    """Round 1 회의 오프닝 — 자연 대화 톤으로 의견 제시.

    이전 'mechanical' 한 schema (수용+타협안 강제) 대신, 회의실에서 사람들이
    자연스럽게 발언하듯 직전 발언자에 직접 반응하는 흐름을 유도.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"

    perspective_block = f"\n[너의 관점] {perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

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
{prev_block}{perspective_block}[너의 협상 태도] {negotiation}
[원래 입장] {stance_kr}
{style_block}{starters_block}

[Round 1 모드 — 자기 입장 주장]
- 이번 라운드는 회의의 첫 발언이다. 자기 perspective 의 차원에서 **자기 진영 ({stance_kr}) 입장을 강하게 옹호**하라.
- 상대 진영 의견을 인정하지 마라. 자기 입장의 핵심 측면 1개를 분명히 던져라.
- 양보·합의 표현 금지 ("그래도 ~", "양측 모두 ~" 등). 합의는 다음 라운드에서 다룬다.
- 본인 perspective 의 분석 측면에서 자기 진영을 옹호하는 구체적 한 가지 (조건·우려·사례·이유) 를 명확히 제시.

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
) -> str:
    """Round 3 — 지금까지 흐름을 종합한 자기 결론.

    이전엔 "제가 생각하는 최적해는 ~입니다" 강제로 5명 모두 같은 시작 → 단조.
    이제 결론 어조는 자유롭게 (단정·제안·요약 형식 다양). 본질은 같음.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"
    perspective_block = f"\n[너의 관점] {perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

    prev_block = ""
    if prev_speaker and prev_speech:
        prev_block = (
            f"\n[직전 발언자: {prev_speaker}]\n\"{prev_speech[:400]}\"\n"
        )

    style_block = _build_style_block(perspective)
    starters_block = _build_starters_block(perspective, used_starters or [])

    return f"""[5단계 — Round 3: 종결 단계, 종합 + 자기 결론]
'{topic}' 에 대한 2라운드 동안의 회의 논의를 종합해 **본인의 최종 결론**을 짧게 정리하라.
{prev_block}{perspective_block}[원래 입장] {stance_kr}
[너의 협상 태도] {negotiation}
{style_block}{starters_block}

[Round 3 모드 — 종결 정리]
- Round 1 (자기 입장 주장) + Round 2 (상대 일부 인정) 흐름을 거쳤다. 이제 본인 관점에서 **합의된 부분 + 끝까지 유지한 본인 입장** 을 정리하라.
- 새 쟁점 꺼내지 마라. 지금까지 회의에서 다룬 내용을 자기 perspective 의 어조로 종합.
- 결론 형식 자유: "정리하자면..." / "결국..." / "저는 이렇게 정리합니다" 등 자연스럽게 시작.
- 본인 perspective 의 tone 유지.

[발언 가이드]
- 2~4문장. 핵심 결론에 **강조** 하나만.
- 합니다체. 회의 마무리 발언 톤 (보고서 X).
- 직전 발언자 결론과의 차이·접점을 자연스럽게 비춰도 좋다.

반드시 아래 형식으로만 출력:

### 반박 시작
(2~4문장 결론)
### 반박 끝"""


def _build_discuss_prompt(
    topic: str,
    original_stance: str,
    user_latest: str,
    perspective: str = "",
    intensity: int = 3,
    already_said: str = "",
    used_starters: Optional[List[str]] = None,
) -> str:
    """Round 2 — 사용자 의견에 자연 대화 톤으로 응답.

    이전엔 inline prompt 로 'mechanical' 했음. 분리 + 자연 회의 톤으로 reframe.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"
    perspective_block = f"\n[너의 관점] {perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

    user_block = ""
    if user_latest:
        user_block = (
            f"\n[방금 사용자 발언 — 직접 응답하라]\n\"{user_latest[:500]}\"\n"
        )

    style_block = _build_style_block(perspective)
    starters_block = _build_starters_block(perspective, used_starters or [])

    return f"""[5단계 — Round 2: 수렴 단계, 상대 의견 일부 인정 + 합의 시도]
'{topic}' 최적해 회의가 진행 중이다. 방금 사용자가 의견을 던졌다 — 직접 응답하라.
{user_block}{perspective_block}[원래 입장] {stance_kr}
[너의 협상 태도] {negotiation}
{already_said}
{style_block}{starters_block}
[Round 2 모드 — 수렴 시작]
- 이번 라운드는 회의의 **수렴 단계**다. Round 1 에서 자기 입장을 주장했으니, 이제는 사용자·다른 발언자 의견 중 **인정할 수 있는 부분을 명시**하고 합의를 시도하라.
- "맞는 점은 ~인데, 다만 자기 perspective 에서는 ~" 식의 흐름.
- 그러나 자기 진영 ({stance_kr}) 입장을 완전히 양보하지 마라. **합의 가능한 접점 1개 + 자기 perspective 의 유지·보완 1개** 를 함께 던져라.
- 단순 "동의합니다" 만으로 끝내지 마라.

[발언 가이드]
- 사용자 발언에 자연스럽게 응답하라. 회의 토론자가 말하는 톤으로.
- 인정·이견·질문·구체화·제안 자유. "그 점은 ~수긍하는데, 다만 ~", "그런 측면도 있죠. 그러면 ~" 등.
- 2~4문장. 합니다체. 보고체 X.
- 위 [이미 한 말] 과 다른 새 측면으로 가라.

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
    intensity = agent.get("intensity", 3)

    print(f"  [{display}] 의견 제시 중... (관점: {perspective[:20]}, 강경도: {intensity})")

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
        used_starters=used_starters,
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
        agent_idx += 1
        intensity = agent.get("intensity", 3)

        print(f"  [{display}] 의견 제시 중... (관점: {perspective[:20]}, 강경도: {intensity})")

        prompt = _build_proposal_prompt(
            topic=topic,
            original_stance=agent["stance"],
            perspective=perspective,
            intensity=intensity,
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
    intensity = agent.get("intensity", 3)
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

    print(f"  [{display}] 응답 중... (강경도: {intensity})")

    # 같은 라운드에서 이미 한 말 (history 에서 추출)
    this_round = _current_synthesis_round_speeches(state)
    already_said = ""
    if this_round:
        already_said = (
            "\n[이번 라운드에서 다른 참여자가 이미 한 말 — 같은 내용 반복 금지]\n"
            + "\n".join(f"- {s}" for s in this_round)
            + "\n"
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

    if is_final_round:
        prompt = _build_finalize_prompt(
            topic, agent["stance"], perspective=perspective, intensity=intensity,
            prev_speaker=prev_speaker_id, prev_speech=prev_speech_text,
            used_starters=used_starters,
        )
        if already_said:
            prompt += already_said
    else:
        prompt = _build_discuss_prompt(
            topic=topic, original_stance=agent["stance"],
            user_latest=user_latest, perspective=perspective, intensity=intensity,
            already_said=already_said, used_starters=used_starters,
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
        agent_idx += 1
        intensity = agent.get("intensity", 3)
        negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

        print(f"  [{display}] 응답 중... (강경도: {intensity})")

        # 이번 라운드에서 다른 에이전트가 이미 말한 내용을 프롬프트에 포함
        already_said = ""
        if this_round_speeches:
            already_said = (
                "\n[이번 라운드에서 다른 참여자가 이미 한 말 — 같은 내용 반복 금지]\n"
                + "\n".join(f"- {s}" for s in this_round_speeches)
                + "\n"
            )

        # 전체 토론 히스토리 + 종합 회의 체인
        from src.graph.llm import build_debate_chain
        debate_chain = build_debate_chain(history, speaker_id)

        if is_final_round:
            # Round 3: 최적해 선언 (user와 동일 프롬프트)
            prompt = _build_finalize_prompt(topic, agent["stance"], perspective=perspective, intensity=intensity)
            if already_said:
                prompt += already_said
        else:
            prompt = (
                f"[너의 고유 관점 — 반드시 이 관점에서만 발언하라] {perspective}\n\n"
                f"[너의 협상 태도] {negotiation}\n"
                f"{already_said}\n"
                f"사용자가 방금 '{user_latest[:100]}...'라고 말했다.\n\n"
                f"위에서 이미 언급된 내용과 완전히 다른 관점에서 구체적 조건이나 미해결 쟁점을 제기하라. "
                f"'동의합니다'/'좋은 의견입니다'/'좋은 출발점'으로 시작하지 마라. "
                f"대화하듯이 자연스럽게. 1~2문장.\n\n"
                f"### 반박 시작\n### 반박 끝"
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
