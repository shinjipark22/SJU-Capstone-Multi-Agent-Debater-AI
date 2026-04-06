"""
test_run_all.py — 모든 주제에 대해 1단계 입론 노드를 순회 실행

터미널에서 실행:
    python -m src.stage1_opening.test_run_all
"""

import json
import sys
import os
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node


# ── 설정 ─────────────────────────────────────────────────────────────────────

DEBATE_FORMAT = "3:3"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 5, 4, 1]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_results"


def _load_all_topics() -> list:
    """모든 주제를 로드하여 리스트로 반환한다."""
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    topics = []
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            topics.append(t)
    return topics


def run_single_topic(topic_dict: dict) -> dict:
    """단일 주제에 대해 입론 노드를 실행하고 결과를 반환한다."""
    personas = create_agents(
        topic=topic_dict,
        debate_format=DEBATE_FORMAT,
        user_stance=USER_STANCE,
        agent_intensities=AGENT_INTENSITIES,
    )

    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id,
            stance=p.stance,
            intensity=p.intensity,
            role_description=p.role_description,
            system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        )
        for p in personas
    ]

    state = build_initial_state(
        topic=topic_dict["title"],
        user_stance=USER_STANCE,
        user_intensity=USER_INTENSITY,
        agents=snapshots,
        topic_id=topic_dict["id"],
    )

    result_state = opening_arguments_node(state)
    return result_state


def format_result(result_state: dict) -> str:
    """결과를 포맷팅된 문자열로 반환한다."""
    lines = []
    _stance_cnt = {"PRO": 0, "CON": 0}

    for entry in result_state["debate_history"]:
        _stance_cnt[entry["stance"]] += 1
        _s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        lines.append("-" * 70)
        lines.append(f"  발언자  : {_s_label} 에이전트{_stance_cnt[entry['stance']]}")
        lines.append(f"  턴 번호 : {entry['turn']}")
        lines.append(f"  단계    : {entry['phase']}")

        tool_log = entry.get("tool_calls_log", [])
        if tool_log:
            lines.append(f"\n  [사용된 도구 — 총 {len(tool_log)}회]")
            for tc in tool_log:
                lines.append(f"    - 도구명: {tc['name']}")
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                lines.append(f"      인자  : {args_str}")
        else:
            lines.append(f"\n  [사용된 도구] 없음")

        lines.append(f"\n  입론 내용:\n")
        lines.append(entry["content"])
        lines.append("")

    return "\n".join(lines)


def main():
    topics = _load_all_topics()
    print(f"\n총 {len(topics)}개 주제를 순회합니다.\n")

    _OUTPUT_DIR.mkdir(exist_ok=True)

    summary = []

    for i, topic in enumerate(topics, 1):
        topic_id = topic["id"]
        title = topic["title"]
        print("=" * 70)
        print(f"  [{i}/{len(topics)}] {topic_id}: {title}")
        print("=" * 70)

        start = time.time()
        try:
            result_state = run_single_topic(topic)
            elapsed = time.time() - start

            # 결과 파일 저장
            output_path = _OUTPUT_DIR / f"opening_{topic_id}.txt"
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"주제: {title}\n")
                f.write(f"토픽 ID: {topic_id}\n")
                f.write(f"소요 시간: {elapsed:.1f}초\n\n")
                f.write(format_result(result_state))

            history = result_state["debate_history"]
            ai_count = len([sid for sid in result_state["speaking_order"] if sid != "user"])
            no_tool = sum(1 for e in history if not e.get("tool_calls_log"))

            status = "OK"
            if len(history) != ai_count:
                status = "FAIL(history)"
            elif no_tool > 0:
                status = f"WARN(도구미사용 {no_tool}명)"

            summary.append(f"  {status:20s} | {elapsed:6.1f}s | {topic_id}: {title}")
            print(f"  -> {status} ({elapsed:.1f}s) -> {output_path.name}\n")

        except Exception as e:
            elapsed = time.time() - start
            summary.append(f"  {'ERROR':20s} | {elapsed:6.1f}s | {topic_id}: {title} — {e}")
            print(f"  -> ERROR ({elapsed:.1f}s): {e}\n")
            traceback.print_exc()

    # 최종 요약
    print("\n" + "=" * 70)
    print("  전체 결과 요약")
    print("=" * 70)
    for line in summary:
        print(line)
    print("=" * 70)

    # 요약 파일 저장
    summary_path = _OUTPUT_DIR / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\n요약 저장: {summary_path}\n")


if __name__ == "__main__":
    main()
