"""
scoring.py — McClelland (2014) 기반 실시간 토론 채점 엔진

[핵심 원칙] 턴 완전 독립 측정
    매 발언 쌍은 이전 턴의 p, v, o 상태를 전혀 이어받지 않는다.
    각 턴마다 p=0, o_prev=0 에서 새로 시작하여
    그 턴에서 추출한 r, g 만으로 수치를 산출한다.

[수식] McClelland (2014)  d=0
    s      : 0.0025  (slowing factor)
    r      : 0~40    입장 강도/논리성  ← Qwen 7B 추출
    g      : 5~50    공격성/타격력     ← Qwen 7B 추출
    p_kt   : 0       (매 턴 리셋 — 이전 v 미사용)
    o_kt   : 0 + s * { g * (r_kt - 0) - 0 } = s * g * r
    v_t    : Σ o_it  (d=0)

    → 각 에이전트 o = 0.0025 × g × r

[범위]
    o  : 0 ~ 5.0   (g_min×r_min=0, g_max×r_max=50×40×0.0025=5.0)
    v  : 0 ~ N×5.0  (N = 발언 에이전트 수)
    Dom: -(N/2×5.0) ~ +(N/2×5.0)

[실시간 출력]
    매 발언 쌍 완료 시:
        [에이전트명] r:XX, g:XX, o:X.XXXX
        종합 대립 지수 (v): X.XXXX
        Dominance: [찬성/반대] +X.XXXX
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

from .extractor import extract_rg

logger = logging.getLogger(__name__)

# ── 수식 상수 ─────────────────────────────────────────────────────────────────
_SLOWING_FACTOR: float = 0.0025  # McClelland (2014)


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class SpeechScore:
    """단일 발언의 추출·계산 결과."""
    agent_id: str
    stance: Literal["PRO", "CON"]
    r: int
    g: int
    o: float


@dataclass
class TurnResult:
    """발언 한 쌍이 완료된 후 산출되는 실시간 결과.

    Attributes:
        turn_index  : 쌍 번호 (0부터 시작)
        phase       : 토론 단계명
        speeches    : 이 쌍에서 발언한 에이전트들의 SpeechScore 리스트
        v           : 종합 대립 지수 (모든 에이전트 o 합산)
        pro_sum     : PRO 에이전트 o 합계
        con_sum     : CON 에이전트 o 합계
        dominance   : 우세 진영 ("PRO" 또는 "CON")
        dominance_gap: 우세 진영 점수 차이 (절댓값)
        log_lines   : 화면 출력용 문자열 리스트
    """
    turn_index: int
    phase: str
    speeches: List[SpeechScore]
    v: float
    pro_sum: float
    con_sum: float
    dominance: Literal["PRO", "CON"]
    dominance_gap: float
    log_lines: List[str] = field(default_factory=list)


# ── 채점 엔진 ─────────────────────────────────────────────────────────────────

class DebateScorer:
    """실시간 턴제 독립 채점 엔진.

    사용 흐름:
        1. 토론 초기화 시 DebateScorer(agents) 생성
        2. 발언 쌍이 완료될 때마다 process_pair() 호출
        3. 반환된 TurnResult를 UI/로그에 즉시 출력

    지원 형식:
        1:1 → 에이전트 2명 (PRO 1, CON 1)
        2:2 → 에이전트 4명 (PRO 2, CON 2)
        3:3 → 에이전트 6명 (PRO 3, CON 3)
        (사용자 포함)

    Example:
        >>> agents = [
        ...     {"agent_id": "user",    "stance": "PRO"},
        ...     {"agent_id": "agent_1", "stance": "CON"},
        ... ]
        >>> scorer = DebateScorer(agents)
        >>> result = scorer.process_pair(
        ...     phase="opening",
        ...     first_speech=("user",    "저는 찬성합니다. 왜냐하면..."),
        ...     second_speech=("agent_1", "반박합니다. 근거가 없습니다."),
        ... )
        >>> for line in result.log_lines:
        ...     print(line)
    """

    def __init__(self, agents: List[Dict]) -> None:
        """
        Args:
            agents: AgentSnapshot 또는 동일 구조 dict 리스트.
                    필수 키: "agent_id", "stance"
        """
        self._stances: Dict[str, Literal["PRO", "CON"]] = {
            a["agent_id"]: a["stance"] for a in agents
        }
        self._turn_index: int = 0  # 처리한 쌍 번호

    # ── 공개 메서드 ───────────────────────────────────────────────────────────

    def process_pair(
        self,
        phase: str,
        first_speech: tuple[str, str],
        second_speech: tuple[str, str],
    ) -> TurnResult:
        """발언 한 쌍을 처리하고 실시간 채점 결과를 반환한다.

        매 턴 p=0, o_prev=0 으로 리셋 후 계산 → 이전 턴 영향 없음.

        계산 흐름:
            r, g  ← Qwen 7B 추출
            o     = 0.0025 × g × r   (p=0, o_prev=0 대입)
            v     = Σ o (이 턴 발언자들의 합)
            Dom   = PRO_Σo − CON_Σo

        Args:
            phase         : 토론 단계명 (e.g. "opening", "free_rebuttal")
            first_speech  : (agent_id, 발언 텍스트)
            second_speech : (agent_id, 발언 텍스트)

        Returns:
            TurnResult — 즉시 출력 가능한 채점 결과
        """
        # p=0, o_prev=0 — 이전 턴 상태 완전 리셋
        speech_scores: List[SpeechScore] = []
        for agent_id, text in (first_speech, second_speech):
            ss = self._compute_speech(agent_id, text)
            speech_scores.append(ss)

        result = self._build_result(phase, speech_scores)
        self._turn_index += 1
        return result

    def process_phase_speeches(
        self,
        phase: str,
        speeches: List[tuple[str, str]],
        on_pair_complete: Optional[Callable[["TurnResult"], None]] = None,
    ) -> List["TurnResult"]:
        """발언 목록을 순서대로 2개씩 묶어 쌍마다 즉시 채점한다.

        입론·자유논박처럼 speaking_order 전체를 한 번에 넘길 때 사용한다.
        쌍이 완료될 때마다 on_pair_complete 콜백이 즉시 호출되므로
        UI에서 실시간 출력이 보장된다.

        예시 — 2:2 입론 (발언 4개 → 채점 2번):
            speeches = [
                ("agent_1", "PRO1 발언"),   # ┐ 쌍 1 완료 → 즉시 채점·출력
                ("agent_3", "CON1 발언"),   # ┘
                ("agent_2", "PRO2 발언"),   # ┐ 쌍 2 완료 → 즉시 채점·출력
                ("user",    "CON2 발언"),   # ┘
            ]
            results = scorer.process_phase_speeches(
                phase="opening",
                speeches=speeches,
                on_pair_complete=print_turn_result,  # 쌍 끝날 때마다 바로 출력
            )
            # results[0] = Turn 1 결과, results[1] = Turn 2 결과

        3:3 입론이면 발언 6개 → 채점 3번, 자유논박 질문-답변도 동일하게 처리된다.

        Args:
            phase            : 토론 단계명 (e.g. "opening", "free_rebuttal")
            speeches         : [(agent_id, 발언 텍스트), ...] — speaking_order 순서
            on_pair_complete : 쌍 채점 직후 호출할 콜백 (없으면 자동 출력 없음)

        Returns:
            TurnResult 리스트 (쌍 수만큼)
        """
        results: List[TurnResult] = []
        for i in range(0, len(speeches) - 1, 2):
            result = self.process_pair(phase, speeches[i], speeches[i + 1])
            if on_pair_complete is not None:
                on_pair_complete(result)
            results.append(result)
        return results

    def get_turn_index(self) -> int:
        """현재까지 처리한 쌍 수를 반환한다."""
        return self._turn_index

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _compute_speech(self, agent_id: str, speech_text: str) -> SpeechScore:
        """단일 발언의 r, g를 추출하고 o를 계산한다.

        p=0, o_prev=0 고정 (매 턴 리셋):
            o = 0 + s * { g * (r - 0) - 0 } = s * g * r

        Args:
            agent_id    : 발언자 ID
            speech_text : 발언 텍스트

        Returns:
            SpeechScore
        """
        if agent_id not in self._stances:
            logger.warning("미등록 에이전트 '%s' → 임시 PRO로 처리.", agent_id)
            self._stances[agent_id] = "PRO"

        stance = self._stances[agent_id]
        r, g = extract_rg(speech_text)

        # p=0, o_prev=0 → o = s * g * r
        o = _SLOWING_FACTOR * g * r

        return SpeechScore(agent_id=agent_id, stance=stance, r=r, g=g, o=o)

    def _build_result(
        self,
        phase: str,
        speeches: List[SpeechScore],
    ) -> TurnResult:
        """TurnResult를 생성하고 출력용 log_lines를 구성한다."""
        pro_sum = sum(ss.o for ss in speeches if ss.stance == "PRO")
        con_sum = sum(ss.o for ss in speeches if ss.stance == "CON")
        v = pro_sum + con_sum
        gap = pro_sum - con_sum

        dominance: Literal["PRO", "CON"] = "PRO" if gap >= 0 else "CON"
        dominance_gap = abs(gap)

        lines = _format_log(
            turn_index=self._turn_index,
            phase=phase,
            speeches=speeches,
            v=v,
            dominance=dominance,
            dominance_gap=dominance_gap,
        )

        return TurnResult(
            turn_index=self._turn_index,
            phase=phase,
            speeches=speeches,
            v=v,
            pro_sum=pro_sum,
            con_sum=con_sum,
            dominance=dominance,
            dominance_gap=dominance_gap,
            log_lines=lines,
        )


# ── 출력 포맷터 ───────────────────────────────────────────────────────────────

def _format_log(
    turn_index: int,
    phase: str,
    speeches: List[SpeechScore],
    v: float,
    dominance: Literal["PRO", "CON"],
    dominance_gap: float,
) -> List[str]:
    """실시간 출력용 문자열 리스트를 생성한다.

    출력 예시:
        ═══ Turn 3 │ 자유논박 ═══
        [user]    r:28, g:35, o: 0.147
        [agent_1] r:15, g:20, o:-0.052
        종합 대립 지수 (v): 0.095
        Dominance: 찬성 +0.199
    """
    phase_label = _PHASE_LABELS.get(phase, phase)
    lines: List[str] = [
        f"═══ Turn {turn_index + 1} │ {phase_label} ═══",
    ]
    for ss in speeches:
        stance_label = "찬성" if ss.stance == "PRO" else "반대"
        lines.append(
            f"  [{ss.agent_id}({stance_label})]  r:{ss.r:2d}, g:{ss.g:2d}, o:{ss.o:+.4f}"
        )
    lines.append(f"  종합 대립 지수 (v): {v:+.4f}")
    dom_label = "찬성" if dominance == "PRO" else "반대"
    lines.append(f"  Dominance: {dom_label} +{dominance_gap:.4f}")
    return lines


_PHASE_LABELS: Dict[str, str] = {
    "opening": "입론",
    "chained_rebuttal": "연쇄 논박",
    "free_rebuttal": "자유 논박",
    "role_reversal": "역할 반전",
    "synthesis": "종합",
}


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def print_turn_result(result: TurnResult) -> None:
    """TurnResult의 log_lines를 표준 출력으로 즉시 출력한다."""
    for line in result.log_lines:
        print(line)
