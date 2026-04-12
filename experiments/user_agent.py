"""
user_agent.py -- GPT-4o-mini 기반 사용자 대행 에이전트

모든 실험에서 사용자 역할을 수행한다.
temperature=0, seed=42로 재현성을 보장하면서 동적 대화를 생성한다.

토픽별 고정된 공격 페르소나를 통해 모든 모델에 공평한 조건을 적용한다.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 사용자 대행 LLM 설정 ───��───────────────────────────────────────────────

USER_AGENT_MODEL = "gpt-4o-mini"
USER_AGENT_TEMPERATURE = 0
USER_AGENT_SEED = 42

# ── 토픽별 공격 페��소나 ───────────────────────────────────────────────────

TOPIC_PERSONAS: Dict[str, str] = {
    # 기술
    "tech_001": "경제적 관점에서 집요하게 반박하라. 실업률, GDP 영향, 소득 불평등 데이터를 근거로 공격하라.",
    "tech_002": "환경 비용과 에너지 소비 관점에서 반박하라. 탄소 배출, 전력 소비 통계를 근거로 공격하라.",
    "tech_003": "안전성과 사고 사례 관점에서 반박하라. 실제 AI 편향/사고 사례를 들어 공격하라.",
    # 경제
    "econ_001": "지정학적 역사와 외교 관점에서 반박하라. 구체적 역사 사건과 조약을 근거�� 공격하라.",
    "econ_002": "거시경제 데이터 관점에서 반박하라. 무역수지, 물가지수, 고용 통계를 근거로 공격하라.",
    "econ_003": "시장 효율성과 경쟁 ���점에서 반박하라. 산업 독과점, 소비자 후생 데이터를 근거로 공격하라.",
    # 정치
    "poli_001": "국가 안보와 군사 전략 ��점에서 반박하라. 군사 작전 사례와 전략적 이익을 근거로 공격하라.",
    "poli_002": "헌법과 법치주의 관점에서 반박하라. 판례와 법적 원칙��� 근거로 공격하라.",
    "poli_003": "식량 안보와 과학적 근거 관점에서 반박하라. 식량 생산성 데이터와 과학 연구를 근거로 공격하라.",
    # 환경
    "env_001": "사회경제적 요인 관점에서 반박하라. 빈곤, 내전, 거버넌스 실패가 더 큰 원인이라고 공격하라.",
    "env_002": "에너지 정책과 공급망 관점에서 반박하라. 에너지 믹스, 정책 실패 사례를 근거로 공격하라.",
    "env_003": "정치적 행동과 제도 개혁 관점에서 반박하라. 교육보다 정치적 참여가 효과적이라고 공격하라.",
}


def _build_system_prompt(topic_id: str, topic_title: str, stance: str) -> str:
    """사용자 대행 에이전트의 시스템 프롬프트를 생성한다."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    persona = TOPIC_PERSONAS.get(topic_id, "논리적으로 반박하라.")

    return f"""너는 토론 참가자(사용자)다. {stance_kr} 입장에서 토론한다.
반드시 한국어 합니다체로 작성하라.

논제: {topic_title}

[너의 공격 전략]
{persona}

[규칙]
- 상대 AI의 직전 발언에 직접 반응하라
- 구체적 근거와 사례를 들어 반박하라
- 감정적 표현 없이 논리적으로 주장하라
- 주어진 관점에서 일관되게 공격하라"""


class UserAgent:
    """GPT-4o-mini 기반 사용자 대행 에이전트.

    모든 실험에서 동일한 모델·시드·페르소나로 사용자 역할을 ���행한다.
    temperature=0, seed=42로 재현성을 보장한다.
    """

    def __init__(self, topic_id: str, topic_title: str, stance: str):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니���.")

        self.client = OpenAI(api_key=api_key)
        self.topic_id = topic_id
        self.topic_title = topic_title
        self.stance = stance
        self.system_prompt = _build_system_prompt(topic_id, topic_title, stance)
        self.history: List[Dict[str, str]] = []

    def _call(self, user_prompt: str, max_tokens: int = 1024) -> str:
        """GPT-4o-mini를 호출��여 사용자 발언을 생성한다."""
        self.history.append({"role": "user", "content": user_prompt})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        response = self.client.chat.completions.create(
            model=USER_AGENT_MODEL,
            messages=messages,
            temperature=USER_AGENT_TEMPERATURE,
            seed=USER_AGENT_SEED,
            max_tokens=max_tokens,
        )

        reply = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def generate_opening(self) -> str:
        """입론을 생성한다."""
        stance_kr = "찬성" if self.stance == "PRO" else "반대"
        prompt = f"""'{self.topic_title}'에 대한 {stance_kr} 입론을 작성하라.

반드시 아래 형식:
### 자기소개와 입장 표명
(1~2문장)
### 논거 1
(3~4문장. 구체적 사례/수치 포함)
### 논거 2
(3~4문장. 구체적 사례/수치 포함)
### 결��
(1~2문장)"""
        return self._call(prompt)

    def generate_rebuttal(self, ai_speech: str) -> str:
        """연쇄논박 반박을 생성한다."""
        prompt = f"""상대 AI의 입론:
{ai_speech[:500]}

위 입론에서 가장 약한 논거 1개를 골라 3~4문장으로 반박하라."""
        return self._call(prompt, max_tokens=512)

    def generate_free_rebuttal_defense(self, ai_attack: str) -> str:
        """자유논박 방어를 생성한���."""
        prompt = f"""상�� AI의 공격:
{ai_attack[:300]}

위 공격에 1~2문장으로 방어하라."""
        return self._call(prompt, max_tokens=256)

    def generate_free_rebuttal_attack(self, ai_defense: str = "") -> str:
        """자유논박 공격을 생성한다."""
        context = f"\n상대 AI의 방어:\n{ai_defense[:300]}\n" if ai_defense else ""
        prompt = f"""{context}상대의 논거에서 아직 공격하지 않은 약점을 1~2문장으로 공격하라."""
        return self._call(prompt, max_tokens=256)

    def generate_role_reversal(self) -> str:
        """역할��전 발언을 생성한다."""
        reversed_stance = "반대" if self.stance == "PRO" else "찬성"
        prompt = f"""[역할 반전] 이제 {reversed_stance} 입장에서 주장하라.

반드시 아래 형식:
### 논거 1
(3~4문��)
### 논거 2
(3~4문장)
### 결론
(1~2문장)"""
        return self._call(prompt, max_tokens=512)

    def generate_synthesis(self, ai_opinions: str) -> str:
        """종합 회의 발언을 생성한다."""
        prompt = f"""AI 에이전트들의 의견:
{ai_opinions[:500]}

위 의견에 반응하면서 최적해를 향한 제안을 1~2문장으로 하라."""
        return self._call(prompt, max_tokens=256)

    def generate_synthesis_final(self) -> str:
        """최적해 최종 확정을 생���한다."""
        prompt = """지���까지의 토론을 종합하여 "우리의 최적해"를 2~3문장으로 확정하라.
양쪽 주장의 타당한 부분을 반영한 균형 잡힌 결론을 내려라."""
        return self._call(prompt, max_tokens=256)
