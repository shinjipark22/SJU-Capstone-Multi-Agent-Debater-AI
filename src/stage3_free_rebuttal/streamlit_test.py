"""
streamlit_test.py — 1~3단계 대화형 토론 테스트 UI

실행:
    streamlit run src/stage3_free_rebuttal/streamlit_test.py
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, DebateEntry, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node
from src.stage2_rebuttal.nodes import chained_rebuttal_node
from src.stage3_free_rebuttal.nodes import free_rebuttal_node

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_results"

DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]


@st.cache_data
def load_all_topics():
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    topics = []
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            topics.append(t)
    return topics


def save_results():
    state = st.session_state.state
    topic_dict = st.session_state.topic_dict
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_id = topic_dict["id"]
    output_path = _OUTPUT_DIR / f"debate_{topic_id}_{timestamp}.txt"

    lines = [
        f"토픽: {topic_dict['title']}",
        f"토픽 ID: {topic_id}",
        f"포맷: {DEBATE_FORMAT} | 사용자: {USER_STANCE}",
        f"생성 시각: {timestamp}",
        "",
    ]
    for phase_name, phase_label in [
        ("opening", "1단계: 입론"),
        ("chained_rebuttal", "2단계: 연쇄논박"),
        ("free_rebuttal", "3단계: 자유논박"),
    ]:
        entries = [e for e in state["debate_history"] if e["phase"] == phase_name]
        lines.append("=" * 70)
        lines.append(f" {phase_label} ({len(entries)}건)")
        lines.append("=" * 70)
        for entry in entries:
            s_label = "찬성" if entry["stance"] == "PRO" else "반대"
            speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
            target = entry.get("target_id", "")
            target_str = f" → {target}" if target else ""
            lines.append("-" * 70)
            lines.append(f"[턴 {entry['turn']}] {speaker} ({s_label}){target_str}")
            lines.append(entry["content"])
            lines.append("")

    lines.append("=" * 70)
    lines.append(f"총 발언: {len(state['debate_history'])}건")
    lines.append("=" * 70)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def init_session():
    if "phase" not in st.session_state:
        st.session_state.phase = "topic_select"
        st.session_state.state = None
        st.session_state.topic_dict = None
        st.session_state.personas = None
        st.session_state.selected_opponent = None
        st.session_state.chat_history = []


def render_entry(entry, prefix=""):
    s_label = "찬성" if entry["stance"] == "PRO" else "반대"
    speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
    target = entry.get("target_id", "")
    target_str = f" → {target}" if target else ""
    is_user = entry["speaker_id"] == "user"

    with st.chat_message("user" if is_user else "assistant"):
        st.markdown(f"**{prefix}{speaker} ({s_label}){target_str}**")
        st.write(entry["content"])


def main():
    st.set_page_config(page_title="토론 테스트", page_icon="🎙️", layout="wide")
    init_session()

    st.title("🎙️ 멀티에이전트 토론 테스트")

    # ══════════════════════════════════════════════
    # 토픽 선택
    # ══════════════════════════════════════════════
    if st.session_state.phase == "topic_select":
        st.header("토픽 선택")
        topics = load_all_topics()
        topic_names = [f"{t['id']} | {t['title']}" for t in topics]
        selected_idx = st.selectbox("토픽을 선택하세요", range(len(topics)),
                                     format_func=lambda i: topic_names[i])
        if st.button("토픽 확정 → 입론 시작", type="primary"):
            topic_dict = topics[selected_idx]
            st.session_state.topic_dict = topic_dict
            personas = create_agents(
                topic=topic_dict, debate_format=DEBATE_FORMAT,
                user_stance=USER_STANCE, agent_intensities=AGENT_INTENSITIES,
            )
            st.session_state.personas = personas
            snapshots = [
                AgentSnapshot(
                    agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
                    role_description=p.role_description, system_prompt=p.system_prompt,
                    focus_area=p.focus_area,
                ) for p in personas
            ]
            state = build_initial_state(
                topic=topic_dict["title"], user_stance=USER_STANCE,
                user_intensity=USER_INTENSITY, agents=snapshots,
            )
            st.session_state.state = state
            st.session_state.phase = "opening_ai"
            st.rerun()

    # ══════════════════════════════════════════════
    # 1단계: 입론 — AI 생성
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "opening_ai":
        st.header("1단계: 입론")
        with st.spinner("AI 에이전트 입론 생성 중..."):
            state = opening_arguments_node(st.session_state.state)
            st.session_state.state = dict(state)
        st.session_state.phase = "opening_user"
        st.rerun()

    elif st.session_state.phase == "opening_user":
        st.header("1단계: 입론")
        topic = st.session_state.topic_dict["title"]
        st.info(f"**토픽:** {topic}\n\n**사용자 입장:** 찬성(PRO)")

        # AI 입론 표시
        st.subheader("AI 에이전트 입론")
        for entry in st.session_state.state["debate_history"]:
            if entry["phase"] == "opening":
                render_entry(entry)

        # 사용자 입론 입력
        st.subheader("사용자 입론 작성")
        user_opening = st.text_area("찬성 입장에서 입론을 작성하세요",
                                      height=200, key="opening_input")
        if st.button("입론 제출 → 연쇄논박", type="primary"):
            if user_opening.strip():
                state = st.session_state.state
                user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
                state["debate_history"].append(DebateEntry(
                    turn=user_turn, speaker_id="user", stance=USER_STANCE,
                    phase="opening", content=user_opening.strip(),
                    target_id=None, tool_calls_log=[], json_raw="",
                ))
                state["debate_history"].sort(key=lambda e: e["turn"])
                state["phase"] = "chained_rebuttal"
                st.session_state.state = state
                st.session_state.phase = "rebuttal_ai"
                st.rerun()
            else:
                st.warning("입론을 입력해주세요.")

    # ══════════════════════════════════════════════
    # 2단계: 연쇄논박 — AI 생성
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "rebuttal_ai":
        st.header("2단계: 연쇄논박")
        with st.spinner("AI 에이전트 연쇄논박 생성 중..."):
            state = chained_rebuttal_node(st.session_state.state)
            st.session_state.state = dict(state)
        st.session_state.phase = "rebuttal_user"
        st.rerun()

    elif st.session_state.phase == "rebuttal_user":
        st.header("2단계: 연쇄논박")

        # AI 연쇄논박 표시
        st.subheader("AI 에이전트 연쇄논박")
        for entry in st.session_state.state["debate_history"]:
            if entry["phase"] == "chained_rebuttal":
                render_entry(entry)

        # 사용자를 공격한 에이전트 찾기
        attacker = None
        for e in reversed(st.session_state.state["debate_history"]):
            if e["phase"] == "chained_rebuttal" and e["target_id"] == "user":
                attacker = e
                break

        if attacker:
            st.warning(f"**{attacker['speaker_id']}**가 사용자를 공격했습니다. 반박하세요!")
            rebuttal_target = attacker["speaker_id"]
        else:
            con_agents = [a["agent_id"] for a in st.session_state.state["agents"] if a["stance"] == "CON"]
            rebuttal_target = con_agents[0] if con_agents else "agent_1"

        st.subheader(f"사용자 연쇄논박 → {rebuttal_target}")
        user_rebuttal = st.text_area("반박을 작성하세요", height=150, key="rebuttal_input")
        if st.button("연쇄논박 제출 → 자유논박", type="primary"):
            if user_rebuttal.strip():
                state = st.session_state.state
                state["debate_history"].append(DebateEntry(
                    turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                    phase="chained_rebuttal", content=user_rebuttal.strip(),
                    target_id=rebuttal_target, tool_calls_log=[], json_raw="",
                ))
                state["current_turn"] += 1
                state["phase"] = "free_rebuttal"
                st.session_state.state = state
                st.session_state.phase = "free_select"
                st.rerun()
            else:
                st.warning("연쇄논박을 입력해주세요.")

    # ══════════════════════════════════════════════
    # 3단계: 자유논박 — 상대 선택
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "free_select":
        st.header("3단계: 자유논박 — 상대 선택")
        opposite = "CON" if USER_STANCE == "PRO" else "PRO"
        opponents = [p for p in st.session_state.personas if p.stance == opposite]
        opponent_names = [f"{p.agent_id} — {p.role_description[:60]}" for p in opponents]
        selected_idx = st.selectbox("상대 에이전트를 선택하세요", range(len(opponents)),
                                     format_func=lambda i: opponent_names[i])
        if st.button("상대 확정 → 자유논박 시작", type="primary"):
            selected = opponents[selected_idx]
            st.session_state.state["selected_opponent_id"] = selected.agent_id
            st.session_state.selected_opponent = selected
            st.session_state.chat_history = []
            st.session_state.phase = "free_rebuttal"
            st.rerun()

    # ══════════════════════════════════════════════
    # 3단계: 자유논박 — 채팅
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "free_rebuttal":
        selected = st.session_state.selected_opponent
        opponent_label = "반대" if selected.stance == "CON" else "찬성"
        topic = st.session_state.topic_dict["title"]

        st.header(f"자유논박: 사용자(찬성) ↔ {selected.agent_id}({opponent_label})")
        st.caption(f"토픽: {topic}")

        # 채팅 히스토리 표시
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 사용자 입력
        user_input = st.chat_input("발언을 입력하세요 (종료하려면 사이드바 버튼)")
        if user_input:
            # 사용자 발언 표시 + 기록
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            state = st.session_state.state
            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                phase="free_rebuttal", content=user_input, target_id=selected.agent_id,
                tool_calls_log=[], json_raw="",
            ))
            state["current_turn"] += 1

            # 에이전트 응답 생성
            with st.spinner(f"{selected.agent_id} 응답 생성 중..."):
                state = free_rebuttal_node(state)
                state = dict(state)
                st.session_state.state = state

            # 에이전트 최신 발언
            agent_entries = [
                e for e in state["debate_history"]
                if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"
            ]
            if agent_entries:
                latest = agent_entries[-1]["content"]
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**[{selected.agent_id}]** {latest}"
                })
            st.rerun()

        # 사이드바: 저장/종료
        with st.sidebar:
            st.subheader("토론 관리")
            st.write(f"**토픽:** {topic[:30]}...")
            st.write(f"**상대:** {selected.agent_id}")
            fr_count = len([e for e in st.session_state.state["debate_history"] if e["phase"] == "free_rebuttal"])
            st.write(f"**자유논박 턴:** {fr_count}건")

            if st.button("💾 결과 저장 (txt)"):
                path = save_results()
                st.success(f"저장: {path}")

            if st.button("🔄 다른 토픽으로"):
                st.session_state.phase = "topic_select"
                st.session_state.state = None
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
