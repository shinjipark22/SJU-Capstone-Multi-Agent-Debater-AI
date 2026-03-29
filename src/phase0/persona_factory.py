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

# ── 카테고리별 논증 분석 시각 (에이전트 번호 순서대로 순환 할당) ──────────────────
# 같은 진영 에이전트끼리 동일한 검색어·논거를 중복 사용하는 문제를 방지한다.
# 에이전트 수가 정의된 수보다 많으면 인덱스를 순환(modulo)하여 재사용한다.
# 토픽 ID 접두사(tech/econ/poli/env)로 카테고리를 판별하여 해당 분야에 특화된 시각을 할당한다.
FOCUS_AREAS: Dict[str, List[str]] = {
    "tech": [
        "기술적 실현 가능성과 혁신성 관점: 현재 기술 수준, 구현 난이도, 기술 성숙도, "
        "기존 기술 대비 차별점 등 '기술 자체의 역량과 한계'를 중심으로 분석하는 시각.",

        "산업 생태계와 경쟁력 관점: 시장 구조 변화, 기업·스타트업 생태계 파급력, 일자리 대체·창출, "
        "글로벌 기술 패권 경쟁 등 '산업과 경제'에 미치는 영향을 중심으로 분석하는 시각.",

        "윤리·사회적 수용성 관점: 개인정보·프라이버시, 알고리즘 편향, 디지털 격차, "
        "인간 자율성 침해 등 '기술이 인간과 사회에 미치는 부작용'을 중심으로 분석하는 시각.",
    ],
    "econ": [
        "거시경제 효과와 성장 관점: GDP·고용률·물가 등 거시 지표 변화, 경기 부양 vs 위축 효과, "
        "국가 재정 건전성 등 '국가 경제 전반'에 미치는 영향을 중심으로 분석하는 시각.",

        "시장 구조와 공정성 관점: 독과점·진입 장벽, 소비자 후생, 중소기업 영향, "
        "소득 불평등 등 '시장 참여자 간 이해관계'를 중심으로 분석하는 시각.",

        "국제 통상과 지정학적 관점: 무역 수지, 공급망 재편, 경제 제재·관세, "
        "국가 간 협력·갈등 등 '글로벌 경제 질서'에 미치는 영향을 중심으로 분석하는 시각.",
    ],
    "poli": [
        "민주주의와 기본권 관점: 표현의 자유, 참정권, 사법 독립, 권력 분립, "
        "소수자 권리 등 '민주적 가치와 시민 권리'를 중심으로 분석하는 시각.",

        "정책 실효성과 제도 관점: 입법·행정 실현 가능성, 정책 비용 대비 효과, "
        "기존 제도와의 정합성, 집행 역량 등 '정책의 현실적 작동 가능성'을 중심으로 분석하는 시각.",

        "사회 통합과 갈등 관점: 계층·세대·지역 간 갈등, 여론 양극화, 사회적 신뢰, "
        "공동체 결속력 등 '사회 구성원 간 관계'에 미치는 영향을 중심으로 분석하는 시각.",
    ],
    "env": [
        "생태계와 환경 영향 관점: 탄소 배출, 생물 다양성, 자원 고갈, 오염 수준, "
        "기후변화 기여도 등 '자연환경에 대한 직접적 영향'을 중심으로 분석하는 시각.",

        "과학적 근거와 기술적 대안 관점: 연구 데이터의 신뢰성, 과학적 합의 수준, "
        "대체 기술 존재 여부, 측정·검증 방법론 등 '과학적 사실과 기술적 해법'을 중심으로 분석하는 시각.",

        "제도·경제적 지속가능성 관점: 환경 규제 실효성, 녹색 산업 경쟁력, 전환 비용, "
        "국제 환경 협약 이행 등 '지속가능한 발전을 위한 제도와 경제 구조'를 중심으로 분석하는 시각.",
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

    desc_block = f"\n[논제 배경]\n{description}\n\n" if description else "\n"

    prompt = f"""[언어 규칙 — 최우선 원칙]
- 모든 출력은 반드시 100% 한국어(한글 + 숫자 + 마크다운 기호)로만 작성하세요.
- 한자(漢字), 일본어(ひらがな/カタカナ), 아랍 문자 등 외국 문자를 절대 사용하지 마세요.
- 영어는 고유명사(GDP, AI, IMF 등)만 허용합니다.
- 이 규칙을 어기면 출력 전체가 무효 처리됩니다.

당신은 {stance_kr} 토론자입니다. ~입니다/~습니다 합쇼체를 사용합니다.

논제: {title}{desc_block}
찬성 입장: {pro}
반대 입장: {con}

당신의 주장: "{my_claim}"
상대방의 주장: "{opp_claim}"
분석 시각: {focus_area}
강경도: {profile['label']} — {profile['style']}
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

    # 토픽 ID 접두사로 카테고리 판별 (예: "tech_001" → "tech")
    topic_id: str = topic.get("id", "")
    category = topic_id.split("_")[0] if "_" in topic_id else ""
    focus_list = FOCUS_AREAS.get(category, list(FOCUS_AREAS.values())[0])

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
        focus_area = focus_list[(idx - 1) % len(focus_list)]

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
