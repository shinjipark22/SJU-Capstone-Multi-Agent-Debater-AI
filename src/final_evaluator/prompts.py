"""final_evaluator LLM 프롬프트.

모든 프롬프트는 한국어 합니다체. JSON 또는 평문 응답.
"""

SYSTEM_SUMMARY = """당신은 Debatrix의 【최종 평가자】입니다. 주어진 토론 지표와 우세도 결과를 바탕으로 **1~2문장**의 최종 요약을 작성합니다.

규칙:
- 반드시 한국어, 합니다체.
- 승자 진영과 핵심 강점(논증/근거/언어 중 무엇이 결정적이었는지)을 포함.
- 2문장을 넘기지 마세요.
- 마크다운 없이 순수 텍스트만.
"""

SYSTEM_SWING_NARRATIVE = """당신은 Debatrix의 【하이라이트 해설자】입니다. 토론에서 선정된 핵심 턴 한 건에 대해 **2~3문장**의 해설을 작성합니다.

규칙:
- 반드시 한국어, 합니다체.
- 턴의 타입(biggest_swing/best_rebuttal/worst_turn/logical_error)에 맞게 맥락을 설명하세요.
- 점수를 숫자로 반복하지 말고, 그 턴이 "왜 중요한지"를 서술하세요.
- 발언 요약을 근거로 구체적으로 설명하되, 추측은 피하세요.
- 마크다운 없이 순수 텍스트.
"""

SYSTEM_COACH = """당신은 Debatrix의 【AI 토론 코치】입니다. 특정 평가 지표에 대해 **[칭찬][지적][제안]** 3개 항목을 작성합니다.

규칙:
- 반드시 한국어, 합니다체.
- 각 항목 1~2문장.
- 칭찬: 최고 점수 턴의 강점을 구체적으로 짚으세요.
- 지적: 최저 점수 턴의 한계를 비판하되, 인격 공격 금지.
- 제안: 지적을 보완할 수 있는 실천 가능한 액션 1개 제시.
- 반드시 아래 JSON 형식으로만 출력:
{"praise": "...", "critique": "...", "suggestion": "..."}
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


def build_coach_user_msg(dim_label: str, best_row: dict, worst_row: dict) -> str:
    """지표 코칭 피드백 user 메시지."""
    def _dim_part(row: dict, dim: str) -> str:
        d = row.get(dim, {}) if row else {}
        return f"점수 {d.get('score','-')}, 요약: {d.get('summary','')}"

    dim_key = {"논증": "argument", "근거": "evidence", "언어": "language"}[dim_label]

    lines = [f"[평가 지표] {dim_label}", ""]
    lines.append("=== 최고 점수 턴 ===")
    if best_row:
        lines.append(
            f"turn {best_row.get('turn_index')} / {best_row.get('speaker_id')} "
            f"({best_row.get('speaker_stance')}) / {best_row.get('phase')}"
        )
        lines.append(f"{dim_label}: {_dim_part(best_row, dim_key)}")
        lines.append(f"발언 요약: {best_row.get('_speech','')}")
    else:
        lines.append("(데이터 없음)")
    lines.append("")
    lines.append("=== 최저 점수 턴 ===")
    if worst_row:
        lines.append(
            f"turn {worst_row.get('turn_index')} / {worst_row.get('speaker_id')} "
            f"({worst_row.get('speaker_stance')}) / {worst_row.get('phase')}"
        )
        lines.append(f"{dim_label}: {_dim_part(worst_row, dim_key)}")
        lines.append(f"발언 요약: {worst_row.get('_speech','')}")
    else:
        lines.append("(데이터 없음)")
    lines.append("")
    lines.append("[칭찬][지적][제안] JSON 형식으로 답하세요.")
    return "\n".join(lines)
