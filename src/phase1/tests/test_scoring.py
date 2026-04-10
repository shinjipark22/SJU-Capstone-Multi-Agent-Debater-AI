"""
test_scoring.py — DebateScorer 단위 테스트

각 턴이 완전히 독립적으로 계산되는지 검증한다.
  o = 0.0025 × g × r  (p=0, o_prev=0 리셋)
  v = Σ o (이 턴 발언자)
  Dom = PRO_Σo − CON_Σo

실행: python -m pytest src/phase1/tests/test_scoring.py -v
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.phase1.scoring import (
    DebateScorer, TurnResult, SpeechScore, _SLOWING_FACTOR, _SIM_STEPS, _simulate_pct
)

# ── 픽스처 ────────────────────────────────────────────────────────────────────
AGENTS_1v1 = [
    {"agent_id": "user",    "stance": "PRO"},
    {"agent_id": "agent_1", "stance": "CON"},
]
AGENTS_2v2 = [
    {"agent_id": "agent_1", "stance": "PRO"},
    {"agent_id": "agent_2", "stance": "PRO"},
    {"agent_id": "agent_3", "stance": "CON"},
    {"agent_id": "user",    "stance": "CON"},
]
AGENTS_3v3 = [
    {"agent_id": "agent_1", "stance": "PRO"},
    {"agent_id": "agent_2", "stance": "PRO"},
    {"agent_id": "agent_3", "stance": "PRO"},
    {"agent_id": "agent_4", "stance": "CON"},
    {"agent_id": "agent_5", "stance": "CON"},
    {"agent_id": "user",    "stance": "CON"},
]


# ── 수식 정확성 ───────────────────────────────────────────────────────────────

class TestFormula:

    def test_pro_reference_positive_con_negative(self):
        """PRO reference = +magnitude, CON reference = -magnitude."""
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair(
            phase="opening",
            first_speech=("user",    "찬성 발언"),
            second_speech=("agent_1", "반대 발언"),
        )
        pro_ss = next(ss for ss in result.speeches if ss.stance == "PRO")
        con_ss = next(ss for ss in result.speeches if ss.stance == "CON")
        assert pro_ss.reference >= 0, "PRO reference가 음수"
        assert con_ss.reference <= 0, "CON reference가 양수"
        assert pro_ss.reference == pro_ss.magnitude
        assert con_ss.reference == -con_ss.magnitude

    def test_simulate_pct_100_steps(self):
        """_simulate_pct 100스텝 수렴 값이 단순 s*g*r보다 훨씬 큰지 확인."""
        o1, o2 = _simulate_pct(ref1=30, g1=30, ref2=-25, g2=25)
        single_step = _SLOWING_FACTOR * 30 * 30  # 1스텝만 했을 때
        assert abs(o1) > single_step * 5, "100스텝 o1이 1스텝 값과 비슷함 → 시뮬 미작동"

    def test_o_converges_toward_reference(self):
        """CON o는 음수, PRO o는 양수 방향으로 수렴해야 한다."""
        o1, o2 = _simulate_pct(ref1=30, g1=30, ref2=-30, g2=30)
        assert o1 > 0, f"PRO o1={o1} 양수여야 함"
        assert o2 < 0, f"CON o2={o2} 음수여야 함"

    def test_v_equals_sum_of_turn_speeches_only(self):
        """v = 이 턴 발언자 o 합계 (비발언자 포함 안 함)."""
        scorer = DebateScorer(AGENTS_2v2)
        result = scorer.process_pair(
            phase="chained_rebuttal",
            first_speech=("agent_1", "찬성 발언"),
            second_speech=("agent_3", "반대 발언"),
        )
        expected_v = sum(ss.o for ss in result.speeches)
        assert abs(result.v - expected_v) < 1e-9

    def test_turns_are_independent(self):
        """턴 독립성: 동일한 발언을 두 번 넣으면 동일한 결과여야 한다."""
        scorer = DebateScorer(AGENTS_1v1)
        r1 = scorer.process_pair("opening", ("user", "같은 발언"), ("agent_1", "같은 발언"))
        r2 = scorer.process_pair("opening", ("user", "같은 발언"), ("agent_1", "같은 발언"))

        # o, v, dominance_gap 모두 동일해야 함 (턴 간 영향 없음)
        for ss1, ss2 in zip(r1.speeches, r2.speeches):
            assert abs(ss1.o - ss2.o) < 1e-9, "동일 발언인데 o가 다름 → 턴 간 상태 누적됨"
        assert abs(r1.v - r2.v) < 1e-9
        assert abs(r1.dominance_gap - r2.dominance_gap) < 1e-9

    def test_turn_index_increments(self):
        scorer = DebateScorer(AGENTS_1v1)
        for expected in range(3):
            result = scorer.process_pair("opening",
                ("user", "찬성"), ("agent_1", "반대"))
            assert result.turn_index == expected

    def test_o_finite_always(self):
        """o, v 가 항상 유한한지 확인."""
        scorer = DebateScorer(AGENTS_1v1)
        for _ in range(100):
            result = scorer.process_pair("free_rebuttal",
                ("user",    "강력한 반박입니다. 근거가 없습니다."),
                ("agent_1", "재반박합니다."),
            )
            assert math.isfinite(result.v)
            for ss in result.speeches:
                assert math.isfinite(ss.o)


# ── 범위 검증 ─────────────────────────────────────────────────────────────────

class TestRange:

    def test_o_range_after_simulation(self):
        """100스텝 후 o는 reference(-40~+40) 근방에 수렴해야 한다."""
        # PRO max, CON max 대칭 케이스
        o1, o2 = _simulate_pct(ref1=40, g1=50, ref2=-40, g2=50, steps=100)
        assert -45 <= o1 <= 45, f"o1={o1} 범위 초과"
        assert -45 <= o2 <= 45, f"o2={o2} 범위 초과"

    def test_v_finite(self):
        """v = o1+o2 가 항상 유한해야 한다."""
        o1, o2 = _simulate_pct(ref1=40, g1=50, ref2=-40, g2=50)
        assert math.isfinite(o1 + o2)


# ── Dominance 검증 ────────────────────────────────────────────────────────────

class TestDominance:

    def test_dominance_is_pro_or_con(self):
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair("opening", ("user", "찬성"), ("agent_1", "반대"))
        assert result.dominance in ("PRO", "CON")

    def test_dominance_gap_non_negative(self):
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair("opening", ("user", "찬성"), ("agent_1", "반대"))
        assert result.dominance_gap >= 0.0

    def test_dominance_equals_v(self):
        """dominance_gap = |v|, dominance는 v 부호로 결정."""
        scorer = DebateScorer(AGENTS_3v3)
        result = scorer.process_pair("free_rebuttal",
            ("agent_1", "강한 논거 제시"), ("agent_4", "반박"))
        assert abs(result.dominance_gap - abs(result.v)) < 1e-9
        if result.v >= 0:
            assert result.dominance == "PRO"
        else:
            assert result.dominance == "CON"


# ── 출력 포맷 ─────────────────────────────────────────────────────────────────

class TestLogLines:

    def test_log_has_enough_lines(self):
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair("opening", ("user", "찬성"), ("agent_1", "반대"))
        assert len(result.log_lines) >= 4

    def test_header_has_turn_and_phase(self):
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair("free_rebuttal", ("user", "질문"), ("agent_1", "답변"))
        assert "Turn 1" in result.log_lines[0]
        assert "자유 논박" in result.log_lines[0]

    def test_dominance_label_korean(self):
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair("opening", ("user", "찬성"), ("agent_1", "반대"))
        dom_line = next(l for l in result.log_lines if "Dominance:" in l)
        assert "찬성" in dom_line or "반대" in dom_line

    def test_judgment_line_exists(self):
        scorer = DebateScorer(AGENTS_1v1)
        result = scorer.process_pair("opening", ("user", "찬성"), ("agent_1", "반대"))
        assert result.dominance_judgment != ""
        assert any("Qwen 판정" in l for l in result.log_lines)


# ── 100번 반복 ────────────────────────────────────────────────────────────────

class TestHundredIterations:

    def test_100_turns_all_independent(self):
        """100번 반복 - 매 턴 독립, 결과 항상 유효."""
        scorer = DebateScorer(AGENTS_2v2)

        # 고정 발언 (텍스트 동일 → r,g 동일 → o,v 동일해야 함)
        FIXED_PRO = "찬성 측 발언입니다. 명확한 근거를 제시합니다."
        FIXED_CON = "반박합니다. 근거가 없습니다. 틀렸습니다."

        results = []
        for i in range(100):
            result = scorer.process_pair(
                phase="free_rebuttal",
                first_speech=("agent_1", FIXED_PRO),
                second_speech=("agent_3", FIXED_CON),
            )
            results.append(result)
            # 매 턴: 유한, 범위 내, 인덱스 순서
            assert math.isfinite(result.v), f"Turn {i}: v가 유한하지 않음"
            assert -80.0 <= result.v <= 80.0, f"Turn {i}: v={result.v} 범위 초과"
            assert result.turn_index == i

        # 동일 입력 → 동일 출력: 턴 간 상태 누적 없음을 증명
        first_v = results[0].v
        assert all(abs(r.v - first_v) < 1e-9 for r in results), (
            "동일 발언인데 턴마다 v가 달라짐 → 턴 간 상태가 누적되고 있음"
        )
