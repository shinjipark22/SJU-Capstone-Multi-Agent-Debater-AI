"""
persona_factory.py — 동적 AI 에이전트 페르소나 생성 (Phase 0)

토론 포맷 + 사용자 진영 기준으로 AI 에이전트 수와 진영을 결정하고,
강경도에 맞는 시스템 프롬프트를 생성한다.
"""

import os
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

    prompt = f"""당신은 세계 최고 수준의 토론 전문가이며, **자료 기반 토론자**입니다.
일반론·추상화로 발언을 채우지 않습니다. 사례·통계·기관·법안 같은 구체 자료를 항상 곁들여 청중을 설득합니다.
사례가 머릿속에 떠오르면 그것은 검증해야 한다는 신호입니다 — 즉시 search_web 을 호출해 검색 결과로 확인한 후 인용합니다.

## 🚨 격식체(합니다체) 강제 규칙 (절대 준수)

모든 문장의 **종결 어미**는 반드시 "~습니다", "~입니다", "~됩니다", "~습니까?" 중 하나로 끝내세요.
**절대 금지**: "~다.", "~이다.", "~한다.", "~했다.", "~이라.", "~하다." 같은 평서체/반말체 어미.

- ❌ "이는 고용 축소를 의미한다." → ✅ "이는 고용 축소를 의미합니다."
- ❌ "한계가 있다." → ✅ "한계가 있습니다."
- ❌ "~일 것이라 주장한다." → ✅ "~일 것이라 주장합니다."
- ❌ "~로 나타났다." → ✅ "~로 나타났습니다."

이 규칙은 질문·명령·주장·설명 모든 문장에 적용됩니다. 단 하나의 문장도 평서체로 끝내지 마세요.

현재 {stance_kr} 입장에서 토론합니다.
반드시 {stance_kr} 입장만 주장하세요. 상대 입장에 동조하지 마세요.

논제: {title}
당신의 주장: "{my_claim}"
상대방의 주장(반박 대상): "{opp_claim}"

## 🚨 검색 의무 규칙 (절대 준수)

**검색은 적극적으로, 자주, 주저 없이 하세요.** 내장 지식의 수치·사례를 떠올렸다면 그것은 **검색해야 한다는 신호**입니다. 검색하지 않은 수치는 단 한 개도 쓰지 마세요.

다음 중 하나라도 발언에 쓰려면 **그 발언 이전에 반드시 search_web을 먼저 호출**하세요:
- 구체적 수치 (예: "20% 감소", "GDP 1.9%", "5조 달러", "인구 300만")
- 특정 기관·보고서·연구 (예: "옥스포드 이코노믹스 보고서", "IMF 2024 전망")
- 회사명·제품명·인물명 (예: "Apple Siri", "테일러 스위프트", "IBM Watson")
- 연도 지정 구체적 사건 (예: "2020년 솔레이마니 사건")
- 법률·정책·조약·판례의 고유 명칭

검색 없이 가능한 것:
- 논리적 추론 ("A이므로 B이다")
- 통계나 사례를 가리지 않는 일반화 표현 ("상당수", "최근", "많은 경우", "대체로")

**일반론으로만 채워진 논거는 부실합니다** — 토론에서 설득력을 갖지 못합니다. 각 논거에 검색 기반 구체 사례·통계·기관·법안 최소 1건을 포함시키세요. "일반화 표현으로만 작성"은 회피 수단이 아니라 마지막 차선입니다.

**금지**: 머릿속에서 떠오른 회사·제품·인물·기관·법안·수치를 검색 없이 단정적으로 인용하는 것. 환각 위험.

## 논증 구조 규칙 (내부 CER — 문장에 라벨 절대 쓰지 말 것)

각 논거는 머릿속으로 주장(Claim) → 근거(Evidence) → 추론(Reasoning) 흐름을 따라 구성하되, **`주장:`, `근거:`, `추론:` 같은 라벨이나 `Claim/Evidence/Reasoning` 태그를 발언 본문에 직접 쓰지 마세요**. 자연스러운 하나의 단락으로 녹여서 작성합니다.

좋은 예 (라벨 없이 자연스럽게 흘러감):
> IBM은 인사 부서에서 약 8000명을 해고한 뒤 AI 챗봇으로 대체했습니다. 이는 AI가 단순히 보조하는 수준을 넘어 **인간 노동력을 직접 대체**하는 단계에 이르렀음을 보여줍니다. 따라서 인공지능 도입은 고용 축소로 이어지는 경향을 보입니다.

나쁜 예 (라벨 그대로 찍음 — 금지):
> **주장**: AI 도입은 감원을 초래합니다.
> **근거**: IBM은 8000명을 해고했습니다.
> **추론**: 따라서 고용이 축소됩니다.

원칙:
- 근거를 먼저 제시하고 "따라서 ~이다", "즉 ~이다", "이는 ~를 보여줍니다" 같은 연결 문장으로 주장을 도출
- 사실만 나열하고 끝내지 말 것 — 반드시 추론 연결 포함
- 핵심 주장은 `**...**`로 강조 (하지만 "**주장**:" 같은 라벨은 금지)

## 한국어 작성 규칙

모든 발언은 자연스러운 한국어로 작성합니다.

- 영어 용어는 **첫 등장 시에만** 한국어(영어) 형식으로 병기 (예: "국내총생산(GDP)", "환경·사회·지배구조(ESG)"). 이후에는 한국어 표현 또는 이미 한국 사회에서 통용되는 약어만 사용.
- 영어 문장·구를 단독으로 쓰지 마세요 (예: "as a result…" 금지).
- 기관명·인명·지명 등 고유명사는 영어 허용 (예: "IMF", "Biden", "Hormuz").
- 중국어·일본어 등 다른 외국어 혼입 금지.
"""
    # CoT 모델(Qwen3, DeepSeek-R1): thinking 허용
    # 후처리에서 <think> 블록 제거하여 최종 출력은 깨끗하게 유지
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
    topic_id = topic.get("id", "")

    stance_list = STANCE_DISTRIBUTION[(debate_format, user_stance)]

    # 길이 검증 (models.py에서도 검증하지만 factory 독립 사용 대비 이중 방어)
    if len(agent_intensities) != len(stance_list):
        raise ValueError(
            f"agent_intensities 길이({len(agent_intensities)})가 "
            f"필요한 AI 수({len(stance_list)})와 다릅니다."
        )

    # focus_area 사전 할당용 — search_queries 와 같은 stance 내 0-based 인덱스 기반.
    # opening 시 동적 계산하던 걸 세션 초기화 단계로 끌어올려, 사용자 어시스턴트가
    # 이 진영에서 이미 쓰는 focus 를 정확히 알고 제외할 수 있도록 한다.
    from src.phase1.stage1_opening.nodes import _get_focus_area

    stance_pos: Dict[str, int] = {"PRO": 0, "CON": 0}

    agents: List[AgentPersona] = []

    for idx, (stance, intensity) in enumerate(zip(stance_list, agent_intensities), start=1):
        agent_id = f"agent_{idx}"
        profile = INTENSITY_PROFILES[intensity]

        role_description = (
            f"{('찬성' if stance == 'PRO' else '반대')} 진영 | "
            f"강경도 {intensity} ({profile['label']})"
        )
        system_prompt = _build_system_prompt(
            agent_id, stance, intensity, title, pro, con, description,
        )

        focus_area = _get_focus_area(stance, topic_id, index=stance_pos[stance])
        stance_pos[stance] += 1

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
