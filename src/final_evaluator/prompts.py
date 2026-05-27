"""final_evaluator LLM 프롬프트.

모든 프롬프트는 한국어 합니다체. JSON 또는 평문 응답.
"""

SYSTEM_SUMMARY = """당신은 토론의 【최종 평가자】입니다. 주어진 토론 지표와 우세도 결과를 바탕으로 **1~2문장**의 최종 요약을 작성합니다.

규칙:
- 반드시 한국어, 합니다체.
- 승자 진영과 핵심 강점(논증/근거/언어 중 무엇이 결정적이었는지)을 포함.
- 2문장을 넘기지 마세요.
- 마크다운 없이 순수 텍스트만.
- **외국어 단 한 글자도 사용 금지** (중국어·일본어·러시아어 등). 한국어와 영어 고유명사만 허용.
"""

SYSTEM_SWING_NARRATIVE = """당신은 토론의 【하이라이트 해설자】입니다. 토론에서 선정된 핵심 턴 한 건에 대해 **2~3문장**의 해설을 작성합니다.

규칙:
- 반드시 한국어, 합니다체.
- 턴의 타입(biggest_swing/best_rebuttal/worst_turn/logical_error)에 맞게 맥락을 설명하세요.
- 점수를 숫자로 반복하지 말고, 그 턴이 "왜 중요한지"를 서술하세요.
- 발언 요약을 근거로 구체적으로 설명하되, 추측은 피하세요.
- 마크다운 없이 순수 텍스트.
- **외국어 단 한 글자도 사용 금지** (중국어·일본어·러시아어 등). 한국어와 영어 고유명사만 허용.
"""

SYSTEM_COACH_PRAISE = """당신은 【AI 토론 코치】입니다. 주어진 '최고 점수 턴'의 강점을 **1~2문장**으로 칭찬합니다.

규칙:
- 반드시 한국어, 합니다체.
- 해당 턴의 강점을 구체적으로 짚으세요 (어떤 부분이 뛰어났는지).
- JSON·마크다운·괄호·접두어 없이 순수 문장만 출력하세요.
- **외국어 단 한 글자도 사용 금지** (중국어·일본어·러시아어 등). 한국어와 영어 고유명사만 허용.
"""

SYSTEM_COACH_CRITIQUE = """당신은 【AI 토론 코치】입니다. 주어진 '최저 점수 턴'의 한계를 **1~2문장**으로 지적합니다.

규칙:
- 반드시 한국어, 합니다체.
- 구체적 약점을 짚되, 인격 공격 금지.
- JSON·마크다운·괄호·접두어 없이 순수 문장만 출력하세요.
- **외국어 단 한 글자도 사용 금지** (중국어·일본어·러시아어 등). 한국어와 영어 고유명사만 허용.
"""

SYSTEM_COACH_SUGGESTION = """당신은 【AI 토론 코치】입니다. 앞서 지적된 한계를 보완할 **실천 가능한 액션 1개를 1~2문장**으로 제안합니다.

규칙:
- 반드시 한국어, 합니다체.
- 추상적 원칙이 아닌 구체적 행동/기법을 제시하세요.
- JSON·마크다운·괄호·접두어 없이 순수 문장만 출력하세요.
- **외국어 단 한 글자도 사용 금지** (중국어·일본어·러시아어 등). 한국어와 영어 고유명사만 허용.
"""


def build_summary_user_msg(
    topic: str,
    winner_side: str,
    pro_pct: float,
    con_pct: float,
    stats: dict,
) -> str:
    """승자/우세도/진영별 평균을 담은 user 메시지."""
    lines = [f"토론 주제: {topic}", ""]
    lines.append(f"최종 우세도: PRO {pro_pct:.1f}% vs CON {con_pct:.1f}%")
    lines.append(f"승자: {winner_side}")
    lines.append("")
    lines.append("진영별 평균 (1~10):")
    for side in ("PRO", "CON"):
        s = stats[side]
        lines.append(
            f"  {side}: 논증 {s['argument']} / 근거 {s['evidence']} / 언어 {s['language']} "
            f"(종합 {s['weighted_mean']}, {s['turn_count']}턴)"
        )
    lines.append("")
    lines.append("위 결과를 바탕으로 최종 평가를 1~2문장으로 작성하세요.")
    return "\n".join(lines)


def build_swing_user_msg(turn: dict) -> str:
    """단일 swing 턴에 대한 해설 요청 메시지."""
    label_map = {
        "biggest_swing": "최대 점수 변동 턴",
        "best_rebuttal": "결정적 반박 턴",
        "worst_turn": "점수가 가장 낮았던 턴",
        "logical_error": "논리적 오류 감지 턴",
    }
    label = label_map.get(turn["type"], turn["type"])
    return (
        f"[턴 타입] {label}\n"
        f"[턴 번호] {turn['turn_index']}\n"
        f"[발언자] {turn['speaker_id']} ({turn['side']})\n"
        f"[단계] {turn['phase']}\n"
        f"[종합 점수] {turn['weighted_score']}\n"
        f"[발언 요약] {turn.get('speech_summary','')}\n"
        f"[평가 요약] {turn.get('overall_summary','')}\n\n"
        f"이 턴에 대한 2~3문장 해설을 작성하세요."
    )


_DIM_KEY = {"논증": "argument", "근거": "evidence", "언어": "language"}


def _format_turn_block(row: dict, dim_label: str) -> str:
    """특정 턴 하나를 설명 문자열로 포맷."""
    if not row:
        return "(데이터 없음)"
    dim_key = _DIM_KEY[dim_label]
    d = row.get(dim_key, {}) or {}
    lines = [
        f"turn {row.get('turn_index')} / {row.get('speaker_id')} "
        f"({row.get('speaker_stance')}) / {row.get('phase')}",
        f"{dim_label} 점수: {d.get('score', '-')} / 요약: {d.get('summary', '')}",
        f"발언 요약: {row.get('_speech', '')}",
    ]
    return "\n".join(lines)


def build_coach_praise_msg(dim_label: str, best_row: dict) -> str:
    return (
        f"[평가 지표] {dim_label} (칭찬 단계)\n\n"
        f"=== 최고 점수 턴 ===\n{_format_turn_block(best_row, dim_label)}\n\n"
        f"이 턴에서 {dim_label} 측면이 왜 뛰어났는지를 1~2문장으로 칭찬하세요."
    )


def build_coach_critique_msg(dim_label: str, worst_row: dict) -> str:
    return (
        f"[평가 지표] {dim_label} (지적 단계)\n\n"
        f"=== 최저 점수 턴 ===\n{_format_turn_block(worst_row, dim_label)}\n\n"
        f"이 턴에서 {dim_label} 측면의 한계를 1~2문장으로 지적하세요."
    )


def build_coach_suggestion_msg(dim_label: str, worst_row: dict, critique_text: str) -> str:
    return (
        f"[평가 지표] {dim_label} (제안 단계)\n\n"
        f"=== 최저 점수 턴 ===\n{_format_turn_block(worst_row, dim_label)}\n\n"
        f"=== 방금 지적된 내용 ===\n{critique_text}\n\n"
        f"위 지적을 보완할 실천 가능한 액션 1개를 1~2문장으로 제안하세요."
    )
