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

# ── 진영 내 논증 전문 분야 (에이전트 번호 순서대로 순환 할당) ─────────────────────
# 같은 진영 에이전트끼리 동일한 검색어·논거를 중복 사용하는 문제를 방지한다.
# 에이전트 수가 정의된 수보다 많으면 인덱스를 순환(modulo)하여 재사용한다.
# PRO/CON 구분 없이 에이전트 번호 순서대로 순환 할당되는 순수 '분석 시각' 리스트.
# 같은 진영이라도 서로 다른 관점에서 논거를 구성하도록 강제한다.
FOCUS_AREAS: List[str] = [
    # 1. 경제·산업적 시각
    "경제적 효용과 산업 파급력 관점: 비용 대비 편익, 거시 경제 지표(GDP, 고용률 등), 시장 경쟁력, "
    "자원 배분의 효율성 등 '자본과 산업'의 관점에서 사안을 분석하는 시각.",

    # 2. 사회·윤리적 시각
    "사회적 영향과 윤리적 타당성 관점: 대중의 삶의 질, 계층 간 형평성, 인간의 기본권, 대중의 수용성 및 "
    "사회적 갈등 등 '인간과 사회 구조'에 미치는 영향을 중심으로 분석하는 시각.",

    # 3. 제도·환경적 시각
    "거시적 지속가능성과 제도적 리스크 관점: 법적/정책적 실현 가능성, 생태계 및 환경적 파급력, "
    "장기적 지속가능성, 역사적 선례 등 '시스템과 거시적 환경'의 관점에서 분석하는 시각.",
]

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

    desc_block = f"\n[논제 배경]\n{description}\n\n" if description else "\n"

    prompt = f"""[언어] 반드시 100% 한국어로만 출력. 중국어 한자 금지. 영문 기관명(IMF, WEF 등)만 예외.

당신은 토론 에이전트 '{agent_id}'입니다.

[논제] {title}{desc_block}[찬성] {pro}
[반대] {con}

[당신의 임무]
당신은 {stance_kr}입니다. {"논제가 '참'임을 증명하라." if stance == "PRO" else "논제가 '거짓'임을 증명하라."}
당신의 주장: "{my_claim}"
상대방의 주장: "{opp_claim}"

[근거 수집]
발언 전 반드시 search_web, search_vector_db를 호출해 근거를 확보하라.
검색 결과에 있는 수치·사례만 인용하라. 없는 데이터는 절대 지어내지 마라.
전문 분야 "{focus_area}"에 집중해 검색하라.

[논증 스타일 — 강경도 {intensity}: {profile['label']}]
전략: {profile['style']}
어조: {profile['tone']}

[출력 규칙]
- ~입니다/~습니다 합쇼체로만 말하라.
- 인사말·역할 설명 없이 첫 문장부터 핵심 주장으로 시작하라.
- 전략을 서술하지 말고 바로 근거를 들어 주장하라.

[출력 형식 — 절대 준수]
당신의 응답은 반드시 아래 JSON 형식으로만 출력하세요. 마크다운 코드 블록이나 다른 부연 설명은 절대 추가하지 마세요.
{{"internal_planning": "검색 결과 분석 및 논리 전개 계획 (속마음과 분석 과정을 여기에 쓰세요)", "final_speech": "인사말 없이 즉시 시작하는 최종 토론 발언 (이 내용만 관중에게 전달됩니다)"}}\""""
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

        # 에이전트 전체 순번(0-based)으로 분석 시각을 순환 할당
        # PRO/CON 구분 없이 agent_1→시각0, agent_2→시각1, agent_3→시각2, ...
        focus_area = FOCUS_AREAS[(idx - 1) % len(FOCUS_AREAS)]

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
