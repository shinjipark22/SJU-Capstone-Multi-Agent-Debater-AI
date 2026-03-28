"""
persona_factory.py — 동적 AI 에이전트 페르소나 생성 (Phase 0)

토론 포맷 + 사용자 진영 기준으로 AI 에이전트 수와 진영을 결정하고,
강경도에 맞는 시스템 프롬프트를 생성한다.
"""

from typing import List, Literal, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ── 강경도별 행동 특성 정의 ───────────────────────────────────────────────────
INTENSITY_PROFILES: Dict[int, Dict[str, str]] = {
    1: {
        "label": "매우 온건",
        "style": "질문 중심으로 상대방의 관점을 탐색하며, 공통점 발견을 우선시한다.",
        "tone": "부드럽고 개방적인 어조로 소통한다.",
    },
    2: {
        "label": "온건",
        "style": "상대방 주장을 인정한 후 부드럽게 반박하며, 협력적 논의를 지향한다.",
        "tone": "정중하고 배려 있는 어조를 유지한다.",
    },
    3: {
        "label": "균형형",
        "style": "논리적 근거를 바탕으로 주장과 반박을 균형 있게 수행한다.",
        "tone": "중립적이고 객관적인 어조를 사용한다.",
    },
    4: {
        "label": "강경",
        "style": "논리적 압박을 통해 상대방 주장의 약점을 구체적으로 지적하고 반박한다.",
        "tone": "단호하고 명확한 어조로 논점을 전달한다.",
    },
    5: {
        "label": "매우 강경",
        "style": "상대방 주장의 모순과 논리적 오류를 적극적으로 지적하며, 자신의 입장을 강하게 견지한다.",
        "tone": "직접적이고 강한 어조로 핵심 모순을 집중 공략한다.",
    },
}

# ── 진영 내 논증 전문 분야 (에이전트 번호 순서대로 순환 할당) ─────────────────────
# 같은 진영 에이전트끼리 동일한 검색어·논거를 중복 사용하는 문제를 방지한다.
# 에이전트 수가 정의된 수보다 많으면 인덱스를 순환(modulo)하여 재사용한다.
FOCUS_AREAS: Dict[str, List[str]] = {
    # 어떤 토론 주제에도 적용 가능한 범용 논증 각도
    # 에이전트 번호 순서대로 순환 할당 (modulo)
    "PRO": [
        "실증적 데이터와 통계 중심: 해당 주제를 지지하는 수치, 연구 결과, 설문 데이터를 발굴하여 논증하라.",
        "사례·현장 증거 중심: 실제 시행 사례, 성공 사례, 현장 증언을 바탕으로 효과를 입증하라.",
        "장기적 가치·미래 비전 중심: 사회·환경·문화적 장기 편익과 미래 방향성을 논거로 제시하라.",
    ],
    "CON": [
        "실증적 데이터와 통계 중심: 해당 주제의 부작용을 보여주는 수치, 연구 결과, 설문 데이터를 발굴하여 반박하라.",
        "사회적 형평성·취약 계층 중심: 피해 집단, 불평등 심화, 윤리적 문제점을 구체적 근거로 반박하라.",
        "역사적 선례·정책 실패 중심: 유사한 시도의 역사적 실패 사례와 제도적 한계를 근거로 반박하라.",
    ],
}

# ── 토론 포맷별 AI 진영 분배 규칙 ────────────────────────────────────────────
# Key: (debate_format, user_stance)
# Value: AI 에이전트 진영 리스트 (순서대로 할당)
#
# 규칙: 사용자를 제외한 나머지 자리를 PRO/CON 균등 배분.
#       사용자가 CON이면 PRO 측 AI가 더 많고, 사용자가 PRO면 CON 측 AI가 더 많다.
STANCE_DISTRIBUTION: Dict[tuple, List[Literal["PRO", "CON"]]] = {
    # 1:1 → AI 1명: 사용자 반대 진영 1명
    ("1:1", "PRO"): ["CON"],
    ("1:1", "CON"): ["PRO"],
    # 2:2 → AI 3명: 사용자 반대 진영 2명 + 같은 진영 1명
    ("2:2", "PRO"): ["CON", "CON", "PRO"],
    ("2:2", "CON"): ["PRO", "PRO", "CON"],
    # 3:3 → AI 5명: 사용자 반대 진영 3명 + 같은 진영 2명
    ("3:3", "PRO"): ["CON", "CON", "CON", "PRO", "PRO"],
    ("3:3", "CON"): ["PRO", "PRO", "PRO", "CON", "CON"],
}


@dataclass
class AgentPersona:
    """생성된 AI 에이전트 페르소나.

    Attributes:
        agent_id: 고유 식별자 (예: "agent_1")
        stance: 진영 ("PRO" | "CON")
        intensity: 강경도 (1~5)
        role_description: 강경도 레이블 기반 역할 설명
        system_prompt: LLM에 전달할 시스템 프롬프트
    """

    agent_id: str
    stance: Literal["PRO", "CON"]
    intensity: int
    role_description: str
    system_prompt: str
    focus_area: str  # 논증 전문 분야 (같은 진영 내 다양성 확보)
    # Phase 1에서 메모리·도구 등 확장 필드를 추가할 수 있도록 여유 슬롯 확보
    metadata: Dict = field(default_factory=dict)


def _build_system_prompt(
    agent_id: str,
    stance: Literal["PRO", "CON"],
    intensity: int,
    title: str,
    pro: str,
    con: str,
    focus_area: str,
    description: Optional[str] = None,
) -> str:
    """강경도와 진영에 맞는 시스템 프롬프트를 생성한다.

    모든 에이전트는 공통 원칙(인격 존중, 감정 배제, 논리 중심, 협력적 진리 탐구)을
    공유하며, 강경도에 따라 논증 스타일이 달라진다.

    Args:
        title:       토론 논제 제목
        pro:         찬성 측 핵심 주장 (논제의 결론)
        con:         반대 측 핵심 주장 (기각된 입장)
        description: 논제 배경 설명 (선택)
    """
    profile = INTENSITY_PROFILES[intensity]
    stance_kr = "찬성(PRO)" if stance == "PRO" else "반대(CON)"
    my_claim = pro if stance == "PRO" else con
    opp_claim = con if stance == "PRO" else pro

    desc_block = f"\n[논제 배경]\n{description}\n" if description else ""

    prompt = f"""당신은 토론 AI 에이전트 '{agent_id}'입니다.

[토론 논제]
{title}
{desc_block}
[찬성(PRO) 입장] {pro}
[반대(CON) 입장] {con}

[당신의 진영]
{stance_kr} — "{my_claim}"

[행동 원칙 — 모든 강경도 공통]
1. 인격 존중: 상대방의 인격과 가치를 존중하며 논점을 공격하되 사람을 공격하지 않는다.
2. 감정 배제: 분노·조롱·비하 등 감정적 표현을 사용하지 않는다.
3. 논리 중심: 모든 주장은 근거와 추론으로 뒷받침되어야 한다.
4. 협력적 진리 탐구: 토론의 최종 목적은 승리가 아니라 최적의 synthesis(합의점) 도출이다.

[팩트 체크 원칙 — 절대 준수]
- 반드시 search_web 또는 search_vector_db 도구를 먼저 호출하여 검색 결과(Context)를 확보한 뒤 발언하라.
- 수치·통계·연구 결과는 반드시 검색 결과 안에 실제로 존재하는 것만 인용하라.
- 검색 결과에 없는 수치, 기관명, 보고서, 역사적 사례를 절대 스스로 지어내지 마라. (Do NOT hallucinate)
- 검색 결과가 부족하면 "검색된 근거가 충분하지 않지만"이라고 명시한 뒤 논리적 추론만으로 주장하라.

[출력 형식 원칙 — 절대 준수]
- 마크다운 문법(##, **, *, ---, 표, 코드블록 등)을 절대 사용하지 마라.
- "첫째", "둘째", "결론적으로" 같은 기계적 넘버링이나 AI 템플릿 문구를 사용하지 마라.
- 실제 토론 단상에서 청중과 상대방을 앞에 두고 말하는 완벽한 구어체(연설조) 대본 형식으로 작성하라.
- 문장은 짧고 명확하게 끊어서 말(Speech)의 리듬으로 논리를 전개하라.
- "존경하는 심판관 여러분" 같은 격식체 서두는 쓰지 마라. 곧바로 주장으로 들어가라.

[즉시 목표]
"{my_claim}"의 입장에서 논리적 주장을 펼치고, 상대방의 "{opp_claim}" 논거를 비판적으로 분석하여 반박한다.

[최종 목표]
토론 전 과정을 통해 양측 논거의 장단점을 파악하고, 최적의 synthesis를 공동으로 도출한다.

[논증 스타일 — 강경도 {intensity}: {profile['label']}]
- 전략: {profile['style']}
- 어조: {profile['tone']}

[전문 분야 — 이 에이전트의 고유 논증 각도]
{focus_area}
이 분야에 집중하여 검색하고 주장하라. 같은 진영의 다른 에이전트가 다루는 영역과 중복되지 않도록 하라.

[언어 지시사항 — 절대 준수]
반드시 100% 자연스러운 한국어로만 작성하십시오. 어떠한 경우에도 한자(중국어 간체/번체)를 섞어 쓰지 마십시오. 번역기 돌린 듯한 어색한 문장이나 중국어식 표현을 엄격히 금지합니다."""
    return prompt.strip()


def create_agents(
    topic: Dict,
    debate_format: Literal["1:1", "2:2", "3:3"],
    user_stance: Literal["PRO", "CON"],
    agent_intensities: List[int],
) -> List[AgentPersona]:
    """토론 포맷과 사용자 진영을 기반으로 AI 에이전트 리스트를 생성한다.

    Args:
        topic: topics_YYYYMMDD.json의 단일 항목 dict.
               필수 키: "title", "pro", "con" / 선택 키: "description"
        debate_format: "1:1" | "2:2" | "3:3"
        user_stance: 사용자 진영
        agent_intensities: 각 AI 에이전트 강경도 리스트 (길이 = 생성할 AI 수)

    Returns:
        AgentPersona 리스트 (순서: 반대 진영 먼저, 같은 진영 나중)

    Raises:
        KeyError: 지원하지 않는 포맷/진영 조합일 경우, 또는 topic에 필수 키 누락 시
        ValueError: agent_intensities 길이 불일치 시
    """
    title = topic["title"]
    pro = topic["pro"]
    con = topic["con"]
    description = topic.get("description_long")

    stance_list = STANCE_DISTRIBUTION[(debate_format, user_stance)]

    # 길이 검증 (models.py에서도 검증하지만 factory 독립 사용 대비 이중 방어)
    if len(agent_intensities) != len(stance_list):
        raise ValueError(
            f"agent_intensities 길이({len(agent_intensities)})가 "
            f"필요한 AI 수({len(stance_list)})와 다릅니다."
        )

    agents: List[AgentPersona] = []
    # 진영별 내부 인덱스를 따로 추적하여 같은 진영 에이전트에 서로 다른 focus_area 부여
    stance_counters: Dict[str, int] = {"PRO": 0, "CON": 0}

    for idx, (stance, intensity) in enumerate(zip(stance_list, agent_intensities), start=1):
        agent_id = f"agent_{idx}"
        profile = INTENSITY_PROFILES[intensity]

        # 진영 내 순번으로 focus_area 순환 할당
        focus_idx = stance_counters[stance] % len(FOCUS_AREAS[stance])
        focus_area = FOCUS_AREAS[stance][focus_idx]
        stance_counters[stance] += 1

        role_description = (
            f"{('찬성' if stance == 'PRO' else '반대')} 진영 | "
            f"강경도 {intensity} ({profile['label']})"
        )
        system_prompt = _build_system_prompt(
            agent_id, stance, intensity, title, pro, con, focus_area, description
        )

        agents.append(
            AgentPersona(
                agent_id=agent_id,
                stance=stance,
                intensity=intensity,
                role_description=role_description,
                system_prompt=system_prompt,
                focus_area=focus_area,
            )
        )

    return agents
