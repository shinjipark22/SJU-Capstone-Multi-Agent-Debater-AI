"""
test_run_all.py — 모든 주제에 대해 2단계 연쇄논박을 순회 실행

입론 state가 캐시되어 있으면 불러오고, 없으면 입론부터 실행한다.

터미널에서 실행:
    python -m src.stage2_rebuttal.test_run_all
"""

import json
import sys
import os
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, DebateEntry, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node, _pre_search, _build_opening_prompt, _generate_opening
from src.stage2_rebuttal.nodes import chained_rebuttal_node


# ── 설정 ─────────────────────────────────────────────────────────────────────

DEBATE_FORMAT = "3:3"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 5, 4, 1]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_results"
_CACHE_DIR = _OUTPUT_DIR / "opening_cache"


def _load_all_topics() -> list:
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    topics = []
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            topics.append(t)
    return topics


def _build_opening_state(topic_dict: dict) -> dict:
    """입론 state를 생성한다. 캐시가 있으면 불러오고, 없으면 실행 후 캐시한다."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{topic_dict['id']}.json"

    if cache_path.exists():
        print(f"  [입론] 캐시 로드: {cache_path.name}")
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    print(f"  [입론] 캐시 없음, 입론 실행 중...")
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
    result_state = dict(result_state)

    # 사용자 대리 입론 생성
    user_turn = state["speaking_order"].index("user") if "user" in state["speaking_order"] else len(result_state["debate_history"])
    stance_kr = "찬성" if USER_STANCE == "PRO" else "반대"
    search_results, tool_log = _pre_search(
        state["topic"], USER_STANCE,
        topic_id=topic_dict["id"],
    )
    prompt = _build_opening_prompt(state["topic"], USER_STANCE, f"{stance_kr} 에이전트(사용자 대리)", search_results)
    user_agent = {
        "system_prompt": f"너는 {stance_kr} 토론자다. 한국어만 사용하라.",
        "stance": USER_STANCE,
    }
    speech, raw = _generate_opening(user_agent, prompt)
    if '### 자기소개' not in speech:
        speech = f"### 자기소개와 입장 표명\n{speech}"
    result_state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=USER_STANCE,
        phase="opening", content=speech, target_id=None,
        tool_calls_log=tool_log, json_raw=raw,
    ))
    result_state["debate_history"].sort(key=lambda e: e["turn"])
    result_state["phase"] = "chained_rebuttal"
    print(f"  [사용자 대리] 입론 생성 완료 (turn={user_turn})")

    # 캐시 저장
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=2)
    print(f"  [입론] 캐시 저장: {cache_path.name}")

    return result_state


def run_single_topic(topic_dict: dict) -> dict:
    """단일 주제에 대해 연쇄논박을 실행하고 결과를 반환한다."""
    opening_state = _build_opening_state(topic_dict)
    opening_state["phase"] = "chained_rebuttal"
    return chained_rebuttal_node(opening_state)


def format_result(result_state: dict) -> str:
    lines = []

    for phase_name, phase_label in [("opening", "입론"), ("chained_rebuttal", "연쇄논박")]:
        phase_entries = [e for e in result_state["debate_history"] if e["phase"] == phase_name]
        if not phase_entries:
            continue

        lines.append("=" * 70)
        lines.append(f"  [{phase_label}]")
        lines.append("=" * 70)

        for entry in phase_entries:
            s_label = "찬성" if entry["stance"] == "PRO" else "반대"
            speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]

            lines.append("-" * 70)
            lines.append(f"  발언자  : {speaker} ({s_label})")
            lines.append(f"  턴 번호 : {entry['turn']}")
            lines.append(f"  단계    : {entry['phase']}")
            if entry.get("target_id"):
                lines.append(f"  대상    : {entry['target_id']}")

            tool_log = entry.get("tool_calls_log", [])
            if tool_log:
                lines.append(f"\n  [사용된 도구 — 총 {len(tool_log)}회]")
                for tc in tool_log:
                    lines.append(f"    - 도구명: {tc['name']}")
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    lines.append(f"      인자  : {args_str}")
            else:
                lines.append(f"\n  [사용된 도구] 없음")

            lines.append(f"\n  내용:\n")
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

            output_path = _OUTPUT_DIR / f"rebuttal_{topic_id}.txt"
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"주제: {title}\n")
                f.write(f"토픽 ID: {topic_id}\n")
                f.write(f"소요 시간: {elapsed:.1f}초\n\n")
                f.write(format_result(result_state))

            rebuttal_count = len([e for e in result_state["debate_history"] if e["phase"] == "chained_rebuttal"])

            status = "OK"
            if rebuttal_count == 0:
                status = "FAIL(논박없음)"

            summary.append(f"  {status:20s} | {elapsed:6.1f}s | 논박 {rebuttal_count}건 | {topic_id}: {title}")
            print(f"  -> {status} ({elapsed:.1f}s) 논박 {rebuttal_count}건 -> {output_path.name}\n")

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

    summary_path = _OUTPUT_DIR / "rebuttal_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\n요약 저장: {summary_path}\n")


if __name__ == "__main__":
    main()
