"""
user_agent.py -- GPT-4o-mini 기반 사용자 대행 에이전트

모든 실험에서 사용자 역할을 동적으로 수행한다.
temperature=0, seed=42로 재현성을 보장하면서 실제 핑퐁 토론을 생성한다.

핵심 설계:
- 토픽별 고정 공격 페르소나 (공격 수위·논리 궤적 통일)
- 대화 히스토리 누적 (멀티턴 맥락 유지)
- 모든 모델에 동일한 사용자 조건 적용
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 사용자 대행 LLM 설정 ────────────────────────────────────────────────────

USER_AGENT_MODEL = "gpt-4o-mini"
USER_AGENT_TEMPERATURE = 0
USER_AGENT_SEED = 42

# ── 토픽별 공격 페르소나 ────────────────────────────────────────────────────
# 각 토픽에 대해 사용자의 공격 관점·전략·핵심 질문을 고정한다.
# 이를 통해 모든 모델에 동일한 "사용자 압박 수준"이 적용된다.

TOPIC_PERSONAS: Dict[str, Dict] = {
    "tech_001": {
        "perspective": "경제적 불평등과 구조적 실업",
        "strategy": "AI가 만드는 일자리는 고학력·고기술 직종에 집중되어 저숙련 노동자의 이동이 현실적으로 불가능하다는 점을 집요하게 파고들어라.",
        "key_questions": [
            "새 일자리의 진입 장벽이 기존 실직자에게 현실적인가?",
            "전환 속도와 재교육 속도 사이의 격차를 어떻게 해결할 것인가?",
        ],
    },
    "tech_002": {
        "perspective": "환경 비용과 에너지 소비",
        "strategy": "데이터센터의 전력 소비와 탄소 배출이 기술 혁신의 이점을 상쇄한다는 점��� 수치로 공격하라.",
        "key_questions": [
            "데이터센터 전력 소비 증가율 대비 에너지 효율 개선율은?",
            "자율 규제가 실패한 산업 사례는?",
        ],
    },
    "tech_003": {
        "perspective": "AI 안전성과 실제 사고 사례",
        "strategy": "AI 편향과 안전 사고 사례를 구체적으로 들어 윤리적 안전장치의 필요성을 ��장하라.",
        "key_questions": [
            "자율주행 사고, 채용 AI 편향 등 실제 피해 사례는?",
            "사후 규제로 충분한가, 사전 안전장치가 필수인가?",
        ],
    },
    "econ_001": {
        "perspective": "지정학적 역사와 외교적 맥락",
        "strategy": "미국의 JCPOA 탈퇴, 이라크 침공 등 구체적 역사 사건을 들어 미국의 정책 실패를 공격하라.",
        "key_questions": [
            "JCPOA 탈퇴 후 이란 우라늄 농축이 가속된 사실을 어떻게 설명하는가?",
            "미국 중동 개입의 총 비용 대비 안정화 성과는?",
        ],
    },
    "econ_002": {
        "perspective": "거시경제 데이터와 소비자 부담",
        "strategy": "관세가 소비자 물가에 전가되는 메커니즘을 통계로 공격하고, 무역적자가 실제로 줄었는지 팩트체크하라.",
        "key_questions": [
            "관세 부과 후 미국 가구당 추가 비용은 얼마인가?",
            "무역적자가 관세 부과 후 실제로 감소했는가?",
        ],
    },
    "econ_003": {
        "perspective": "시장 효율성과 소비자 후생",
        "strategy": "플랫폼 독점이 소비자 후생을 저해하는 구체적 사례를 들어 규제 필요성을 주장하라.",
        "key_questions": [
            "독과점이 가격, 혁신, 소비자 선택에 미치는 영향은?",
            "자율 규제 vs 정부 규제 중 어느 쪽이 효과적인가?",
        ],
    },
    "poli_001": {
        "perspective": "국가 안보와 군사 전략",
        "strategy": "보안과 투명성의 균형이 필요하며, 무제한 접근은 작전 실패로 이어질 수 있다는 점을 군사 사례로 공격하라.",
        "key_questions": [
            "베트남전 언론 보도가 작전에 미친 영향은?",
            "보안 제한 없이 언론 접근을 허용한 나라의 결과는?",
        ],
    },
    "poli_002": {
        "perspective": "헌법과 법치주의",
        "strategy": "표현의 자유와 공공 안전 사이의 법적 균형을 판례와 헌법 원칙으로 공격하라.",
        "key_questions": [
            "표현의 자유에 대한 헌법적 제한의 근거는?",
            "다른 민주주의 국가들의 관련 판례는?",
        ],
    },
    "poli_003": {
        "perspective": "식량 안보와 과학적 근거",
        "strategy": "GM 작물의 생산성 향상 데이터를 들어 식량 안보 기여를 주장하고, 생태계 위험은 관리 가능하다고 반박하라.",
        "key_questions": [
            "GM 작물 도입 후 수확량 변화 데이터는?",
            "GM 작물의 생태계 위험이 실증적으로 입증된 사례는?",
        ],
    },
    "env_001": {
        "perspective": "사회경제적 요인과 거버넌스",
        "strategy": "난민 발생의 주원인이 내전·빈곤·거버넌스 실패이며, 기후는 촉매에 불과하다는 점을 사례로 공격하라.",
        "key_questions": [
            "같은 기후 조건에서 난민이 발생하는 나라와 안 하는 나라의 차이는?",
            "시리아 내전에서 가뭄의 역할은 얼마나 결정적이었는가?",
        ],
    },
    "env_002": {
        "perspective": "에너지 정책과 공급망",
        "strategy": "인플레이션의 주원인이 에너지 정책 실패와 공급망 교란이며, 기후변화는 간접 요인이라고 공격하라.",
        "key_questions": [
            "독일의 에너지 전환 실패가 전기 가격에 미친 영향은?",
            "프랑스 원전 비중과 전기료의 관계는?",
        ],
    },
    "env_003": {
        "perspective": "정치적 행동과 제도 개혁",
        "strategy": "과학적 교육보다 정치적 참여와 제도 개혁이 기후위기 대응에 더 효과적이라고 공격하라.",
        "key_questions": [
            "환경 교육 수준과 실제 탄소 배출 감소 사이의 상관관계는?",
            "정치적 압력으로 기후 정책이 바뀐 사례는?",
        ],
    },
}


def _build_system_prompt(topic_id: str, topic_title: str, stance: str, pro_claim: str, con_claim: str) -> str:
    """사용자 대행 에이전트의 시스템 프롬프트를 생성한다."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    my_claim = pro_claim if stance == "PRO" else con_claim
    opp_claim = con_claim if stance == "PRO" else pro_claim

    persona = TOPIC_PERSONAS.get(topic_id, {})
    perspective = persona.get("perspective", "논리적으로 반박")
    strategy = persona.get("strategy", "상대 주장의 약점을 찾아 반박하라.")
    key_questions = persona.get("key_questions", [])
    questions_block = "\n".join(f"  - {q}" for q in key_questions) if key_questions else ""

    return f"""너는 구성적 논쟁(Constructive Controversy) 토론의 참가자다.
{stance_kr} 입장에서 토론하며, 최종적으로는 최적해를 함께 찾는 것이 목표다.

[논제] {topic_title}
[너의 주장] {my_claim}
[상대 주장(반박 대상)] {opp_claim}

[너의 공격 관점] {perspective}
[너의 전략] {strategy}
[핵심 질문]
{questions_block}

[형식 규칙]
- 반드시 한국어 합니다체(격식체)로 작성하라
- 모든 문장을 "~합니다", "~입니다", "~됩니다"로 끝내라
- 핵심 주장에 **강조** 표시하라
- 상대 AI의 직전 발언에 직접 반응하라. 벽보고 얘기하지 마라
- 구체적 근거(수치, 사례, 국가 비교)를 들어 반박하라
- 감정적 표현, 인신공격 없이 논리적으로 주장하라
- 주어진 공격 관점에서 일관되게 공격하라

[금지]
- 상대 의견에 쉽게 동의하지 마라
- "좋은 지적입니다" 같은 빈 말 금지
- 추상적 일반론 금지. 구체적으로 공격하라"""


class UserAgent:
    """GPT-4o-mini 기반 사용자 대행 에이전트.

    모든 실험에서 동일한 모델(GPT-4o-mini), 시드(42), 온도(0)로
    사용자 역할을 수행한다. 대화 히스토리를 누적하여 멀티턴 맥락을 유지한다.
    """

    def __init__(self, topic_id: str, topic_title: str, stance: str, pro_claim: str = "", con_claim: str = ""):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.topic_id = topic_id
        self.stance = stance
        self.system_prompt = _build_system_prompt(topic_id, topic_title, stance, pro_claim, con_claim)
        self.conversation: List[Dict[str, str]] = []  # 전체 대화 히스토리

    def _call(self, prompt: str, max_tokens: int = 1024) -> str:
        """GPT-4o-mini 호출. 대화 히스토리에 누적."""
        self.conversation.append({"role": "user", "content": prompt})

        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation

        response = self.client.chat.completions.create(
            model=USER_AGENT_MODEL,
            messages=messages,
            temperature=USER_AGENT_TEMPERATURE,
            seed=USER_AGENT_SEED,
            max_tokens=max_tokens,
        )

        reply = response.choices[0].message.content.strip()
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def add_ai_context(self, ai_speech: str, speaker_label: str = "상대 AI"):
        """AI 에이전트의 발언을 대화 히스토리에 추가한다 (맥락 유지용)."""
        self.conversation.append({
            "role": "user",
            "content": f"[{speaker_label}의 발언]\n{ai_speech}",
        })

    # ── 단계별 생성 메서드 ──────────────────────────────────────────────────

    def generate_opening(self) -> str:
        """1단계 입론을 생성한다."""
        stance_kr = "찬성" if self.stance == "PRO" else "반대"
        return self._call(f"""'{self.stance}' 입장에서 입론을 작성하라.

반드시 아래 구조로:

### 자기소개와 입장 표명
저는 이 논제에 {stance_kr}합니다. (1~2문장으로 입장과 이유)

### 논거 1: (소제목)
(3~5줄. 구체적 사례·수치·국가 비교 포함)

### 논거 2: (소제목)
(3~5줄. 구체적 사례·수치·국가 비교 포���)

### 결론
(1~2문장)""")

    def generate_rebuttal(self, ai_speech: str) -> str:
        """2단계 연쇄논박을 생성한다."""
        self.add_ai_context(ai_speech, "상대 AI")
        return self._call(
            "위 상대 AI의 발언에서 가장 약한 논거 1개를 골라 3~4문장으로 반박하라. "
            "너의 공격 관점에서 구체적 근거를 들어 공격하라.",
            max_tokens=512,
        )

    def generate_free_rebuttal_defense(self, ai_attack: str) -> str:
        """3단계 자유논박 방어를 생성한다."""
        self.add_ai_context(ai_attack, "상대 AI 공격")
        return self._call(
            "위 상대 AI의 공격에 1~2���장으로 방어하라. 질문이 있으면 먼저 답하라.",
            max_tokens=256,
        )

    def generate_free_rebuttal_attack(self) -> str:
        """3단계 자유논박 공격을 생성한다."""
        return self._call(
            "이제 너의 공격 관점에서 아직 공격하지 않은 약점을 1~2문장으로 공격하라. "
            "핵심 질문 중 아직 사용하지 않은 것을 활용하라.",
            max_tokens=256,
        )

    def generate_role_reversal(self) -> str:
        """4단계 역할반전 발언을 생성한다."""
        reversed_kr = "반대" if self.stance == "PRO" else "찬성"
        return self._call(f"""[역할 반전] 이제 {reversed_kr} 입장에서 주장하라.
원래 입장을 완전히 버리고, 상대편의 논리로 진심으로 설득하라.

반드시 아래 구조로:
### 논거 1: (소제목)
(3~4문장)
### 논거 2: (소제목)
(3~4문장)
### 결론
(1~2문장)""", max_tokens=512)

    def generate_synthesis(self, ai_opinions: str) -> str:
        """5단계 종합 회의 발언을 생성한다."""
        self.add_ai_context(ai_opinions, "AI 에이전트들")
        return self._call(
            "위 AI 에이전트들의 의견에 반응하면서, 양쪽 주장을 절충한 최적해를 향한 제안을 1~2문장으로 하라.",
            max_tokens=256,
        )

    def generate_synthesis_final(self) -> str:
        """최적해 최종 확정을 생성한다."""
        return self._call(
            "지금까지의 토론을 종합하여 '우리의 최적해'를 2~3문장으로 확정하라. "
            "양쪽 주장의 타당한 부분을 반영한 균형 잡힌 결론을 내려라.",
            max_tokens=256,
        )
