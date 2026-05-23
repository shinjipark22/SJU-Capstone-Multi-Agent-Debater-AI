"""실제 토론 transcript JSON 으로부터 토론 어시스턴트 5단계 발화를 생성.

사용:
    python tests/assistant_from_transcript.py
    python tests/assistant_from_transcript.py --file data/logs/Qwen-2.5-32B-Instruct/tech_001_2v2_balanced.json

각 페이즈마다 '사용자가 발언하기 직전' 상태의 GuideContext 를 transcript 에서
재구성해서 어시스턴트가 사용자에게 보여줄 안내문 생성.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()


from src.debate_assistant import (  # noqa: E402
    GuideContext,
    HistoryEntry,
    build_guide_message,
    get_user_slot_focus_area,
)


_PHASES = ["opening", "chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"]


def _load_topic_meta(topic_id: str) -> dict:
    """topics_*.json 에서 pro/con 라벨 조회."""
    topics_path = _ROOT / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        return {"pro": "찬성", "con": "반대"}
    with topics_path.open(encoding="utf-8") as f:
        topics = json.load(f)
    for cat in topics.get("categories", {}).values():
        for t in cat:
            if t["id"] == topic_id:
                return t
    return {"pro": "찬성", "con": "반대"}


def _stance_list_for(debate_format: str, user_stance: str):
    table = {
        ("2:2", "PRO"): ["CON", "CON", "PRO"],
        ("2:2", "CON"): ["PRO", "PRO", "CON"],
        ("3:3", "PRO"): ["CON", "CON", "CON", "PRO", "PRO"],
        ("3:3", "CON"): ["PRO", "PRO", "PRO", "CON", "CON"],
    }
    return table.get((debate_format, user_stance), [])


def _agent_stance_map(transcript: dict) -> dict:
    """agent_id → stance 매핑."""
    stances = _stance_list_for(transcript["debate_format"], transcript["user_stance"])
    return {f"agent_{i}": st for i, st in enumerate(stances, start=1)}


def _turns_before_user_in_phase(turns: list, phase: str) -> list:
    """해당 phase 에서 사용자가 처음 말하기 직전까지의 turn 들."""
    user_first_idx = None
    for i, t in enumerate(turns):
        if t.get("phase") == phase and t.get("speaker") == "user":
            user_first_idx = i
            break
    if user_first_idx is None:
        return turns
    return turns[:user_first_idx]


def _all_turns_before_phase_user(turns: list, phase: str) -> list:
    """해당 phase 사용자 첫 발언 직전까지 전체 history."""
    return _turns_before_user_in_phase(turns, phase)


def _opponent_speech(turns_so_far: list, phase: str, user_stance: str, agent_stance_map: dict) -> str:
    """현재 phase 의 마지막 비-user AI 발언 (사용자가 받아칠 대상).
    못 찾으면 phase 무관하게 마지막 비-user 발언."""
    for t in reversed(turns_so_far):
        if t.get("speaker") == "user":
            continue
        if t.get("phase") == phase:
            return t.get("text", "") or ""
    for t in reversed(turns_so_far):
        if t.get("speaker") != "user":
            return t.get("text", "") or ""
    return ""


def _to_history_entries(turns: list, agent_stance_map: dict, user_stance: str) -> list:
    """transcript turn list → HistoryEntry list (어시스턴트 input 포맷)."""
    out = []
    for t in turns:
        sp = t.get("speaker", "")
        if sp == "user":
            st = user_stance
        else:
            st = agent_stance_map.get(sp, t.get("side", "PRO"))
        out.append(HistoryEntry(
            speaker_id=sp,
            stance=st,
            phase=t.get("phase", ""),
            content=t.get("text", "") or "",
        ))
    return out


def build_ctx(transcript: dict, phase: str, topic_meta: dict, user_focus_area: str) -> GuideContext:
    agent_map = _agent_stance_map(transcript)
    user_stance = transcript["user_stance"]
    history_turns = _all_turns_before_phase_user(transcript["turns"], phase)
    history = _to_history_entries(history_turns, agent_map, user_stance)
    opp = _opponent_speech(history_turns, phase, user_stance, agent_map)

    return GuideContext(
        topic=transcript["topic"],
        user_stance=user_stance,
        pro_claim=topic_meta.get("pro", "찬성"),
        con_claim=topic_meta.get("con", "반대"),
        topic_id=transcript["topic_id"],
        user_focus_area=user_focus_area,
        assistant_name="비비드",
        opponent_speech=opp,
        history=history,
        links=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default=str(_ROOT / "data/logs/Qwen-2.5-32B-Instruct/tech_001_2v2_balanced.json"),
        help="transcript JSON 경로",
    )
    parser.add_argument(
        "--phases", nargs="*", default=_PHASES, choices=_PHASES,
        help="생성할 페이즈 (기본: 전체 5단계)",
    )
    args = parser.parse_args()

    fp = Path(args.file)
    if not fp.exists():
        raise SystemExit(f"파일 없음: {fp}")

    with fp.open(encoding="utf-8") as f:
        transcript = json.load(f)

    print(f"[transcript] {fp.name}")
    print(f"  토픽: {transcript['topic']}")
    print(f"  사용자 진영: {transcript['user_stance']}")
    print(f"  포맷: {transcript['debate_format']} | 강경도: {transcript['agent_intensities']}")
    print(f"  turns: {len(transcript['turns'])}")
    print()

    topic_meta = _load_topic_meta(transcript["topic_id"])

    # 사용자 진영의 AI 가 가져가는 focus 들 시뮬레이션 (운영 환경 재현).
    # transcript 에는 agent 별 focus 가 없어서 직접 search_queries 와 stance 분포로 추정:
    # 같은 진영 AI 가 keywords[0], [1], ... 순서대로 점유 → 사용자는 그 다음 자리.
    from src.phase1.stage1_opening.nodes import _get_focus_area
    agent_map = _agent_stance_map(transcript)
    same_stance_ai_count = sum(1 for st in agent_map.values() if st == transcript["user_stance"])
    used_focus = [
        _get_focus_area(transcript["user_stance"], topic_id=transcript["topic_id"], index=i)
        for i in range(same_stance_ai_count)
    ]
    used_focus = [f for f in used_focus if f]
    user_focus_area = get_user_slot_focus_area(
        transcript["topic_id"], transcript["user_stance"], excluded_focuses=used_focus
    )
    print(f"[focus_area] 사용자 슬롯에 배정될 focus: {user_focus_area!r}")
    print()

    out_dir = _ROOT / "test_results" / "assistant_guide"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = fp.stem
    md_path = out_dir / f"assistant_from_{stem}_{ts}.md"
    json_path = out_dir / f"assistant_from_{stem}_{ts}.json"

    results = {}
    for phase in args.phases:
        print(f"{'=' * 72}\n [{phase}]\n{'=' * 72}")
        try:
            ctx = build_ctx(transcript, phase, topic_meta, user_focus_area)
            text = build_guide_message(phase, ctx)
            results[phase] = {"status": "ok", "text": text}
            print(text)
            print()
        except Exception as e:
            results[phase] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # 발표자료용 markdown
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# 토론 어시스턴트 5단계 발화 — {transcript['topic_id']}\n\n")
        f.write(f"- **토픽**: {transcript['topic']}\n")
        f.write(f"- **사용자 진영**: {transcript['user_stance']}\n")
        f.write(f"- **소스 transcript**: `{fp.name}`\n")
        f.write(f"- **focus_area**: {user_focus_area}\n\n---\n\n")
        for phase in args.phases:
            r = results.get(phase, {})
            f.write(f"## {phase}\n\n")
            if r.get("status") == "ok":
                f.write(r["text"])
                f.write("\n\n---\n\n")
            else:
                f.write(f"_ERROR: {r.get('error', 'unknown')}_\n\n---\n\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "transcript": fp.name,
            "topic": transcript["topic"],
            "topic_id": transcript["topic_id"],
            "user_stance": transcript["user_stance"],
            "focus_area": user_focus_area,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n저장:\n  {md_path}\n  {json_path}")
    fail = sum(1 for r in results.values() if r.get("status") != "ok")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
