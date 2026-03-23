"""
persona_factory.py — 동적 AI 에이전트 페르소나 생성 (Phase 0)

토론 포맷 + 사용자 진영 기준으로 AI 에이전트 수와 진영을 결정하고,
강경도에 맞는 시스템 프롬프트를 생성한다.
"""

from typing import List, Literal, Dict, Optional
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
    # Phase 1에서 메모리·도구 등 확장 필드를 추가할 수 있도록 여유 슬롯 확보
    metadata: Dict = field(default_factory=dict)


def _build_system_prompt(
    agent_id: str,
    stance: Literal["PRO", "CON"],
    intensity: int,
    title: str,
    pro: str,
    con: str,
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

[즉시 목표]
"{my_claim}"의 입장에서 논리적 주장을 펼치고, 상대방의 "{opp_claim}" 논거를 비판적으로 분석하여 반박한다.

[최종 목표]
토론 전 과정을 통해 양측 논거의 장단점을 파악하고, 최적의 synthesis를 공동으로 도출한다.

[논증 스타일 — 강경도 {intensity}: {profile['label']}]
- 전략: {profile['style']}
- 어조: {profile['tone']}
"""
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
    description = topic.get("description")

    stance_list = STANCE_DISTRIBUTION[(debate_format, user_stance)]

    # 길이 검증 (models.py에서도 검증하지만 factory 독립 사용 대비 이중 방어)
    if len(agent_intensities) != len(stance_list):
        raise ValueError(
            f"agent_intensities 길이({len(agent_intensities)})가 "
            f"필요한 AI 수({len(stance_list)})와 다릅니다."
        )

    agents: List[AgentPersona] = []
    for idx, (stance, intensity) in enumerate(zip(stance_list, agent_intensities), start=1):
        agent_id = f"agent_{idx}"
        profile = INTENSITY_PROFILES[intensity]
        role_description = (
            f"{('찬성' if stance == 'PRO' else '반대')} 진영 | "
            f"강경도 {intensity} ({profile['label']})"
        )
        system_prompt = _build_system_prompt(
            agent_id, stance, intensity, title, pro, con, description
        )

        agents.append(
            AgentPersona(
                agent_id=agent_id,
                stance=stance,
                intensity=intensity,
                role_description=role_description,
                system_prompt=system_prompt,
            )
        )

    return agents
