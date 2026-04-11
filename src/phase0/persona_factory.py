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
        "style": "상대 주장의 합리적 부분을 인정하면서도 자기 입장을 논리적으로 방어하라. 공격적 표현을 피하고 설득력 있게 주장하라.",
    },
    2: {
        "label": "온건",
        "style": "상대 주장의 약점을 지적하되 과도한 비판은 피하라. 근거 중심으로 차분하게 반박하라.",
    },
    3: {
        "label": "균형형",
        "style": "상대 주장을 정면으로 반박하라. 약점을 날카롭게 지적하되 감정적 표현은 피하라.",
    },
    4: {
        "label": "강경",
        "style": "상대 주장의 허점을 강하게 공격하라. 양보하지 말고, 상대가 답하기 어려운 질문을 던져라.",
    },
    5: {
        "label": "매우 강경",
        "style": "상대 주장을 전면 부정하라. 가장 치명적인 약점을 집중 공격하고, 상대의 전제 자체를 흔들어라. 절대 양보하지 마라.",
    },
}

# ── 카테고리별 논증 분석 시각 (에이전트 번호 순서대로 순환 할당) ──────────────────
# 같은 진영 에이전트끼리 동일한 검색어·논거를 중복 사용하는 문제를 방지한다.
# 에이전트 수가 정의된 수보다 많으면 인덱스를 순환(modulo)하여 재사용한다.
# 토픽 ID 접두사(tech/econ/poli/env)로 카테고리를 판별하여 해당 분야에 특화된 시각을 할당한다.

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

    prompt = f"""당신은 세계 최고 수준의 토론 전문가입니다.
논리적이고 설득력 있는 주장을 펼치며, 구체적인 데이터와 사례로 청중을 설득합니다.
모든 발언은 반드시 합니다체(격식체)로 작성합니다.

현재 {profile['label']} {stance_kr} 입장에서 토론합니다.
반드시 {stance_kr} 입장만 주장하세요. 상대 입장에 동조하지 마세요.

[논증 스타일]
{profile['style']}

논제: {title}
당신의 주장: "{my_claim}"
상대방의 주장(반박 대상): "{opp_claim}"

참고 자료에 없는 수치나 통계를 지어내지 마세요.
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
    description = topic.get("description_long")

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
                focus_area="",
            )
        )

    return agents
