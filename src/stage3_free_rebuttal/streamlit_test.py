"""
streamlit_test.py — 1~3단계 대화형 토론 테스트 UI (채팅 형태)

실행:
    streamlit run src/stage3_free_rebuttal/streamlit_test.py --server.port 8501
"""

import json
import sys
import os
import threading
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
    personas = st.session_state.personas
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_id = topic_dict["id"]
    output_path = _OUTPUT_DIR / f"debate_{topic_id}_{timestamp}.txt"

    lines = [
        f"토픽: {topic_dict['title']}",
        f"토픽 ID: {topic_id}",
        f"포맷: {DEBATE_FORMAT} | 사용자: {USER_STANCE}",
        f"에이전트: {len(personas)}명",
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
        st.session_state.messages = []
        st.session_state.opening_ready = False
        st.session_state.rebuttal_ready = False


def add_msg(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


_bg_result = {}  # 스레드 간 공유 딕셔너리


def _run_opening_bg(state_copy):
    """백그라운드에서 AI 입론 생성."""
    result = opening_arguments_node(state_copy)
    _bg_result["opening"] = dict(result)


def _run_rebuttal_bg(state_copy):
    """백그라운드에서 AI 연쇄논박 생성."""
    result = chained_rebuttal_node(state_copy)
    _bg_result["rebuttal"] = dict(result)


def main():
    st.set_page_config(page_title="토론 테스트", page_icon="🎙️", layout="wide")
    init_session()

    # ── 사이드바 (토픽 선택 이후)
    if st.session_state.phase != "topic_select":
        with st.sidebar:
            st.title("🎙️ 토론 테스트")
            if st.session_state.topic_dict:
                st.write(f"**토픽:** {st.session_state.topic_dict['title'][:40]}...")
                phase_labels = {
                    "opening_user": "1단계: 입론 (작성 중)",
                    "opening_done": "1단계: 입론 완료",
                    "rebuttal_user": "2단계: 연쇄논박 (작성 중)",
                    "rebuttal_done": "2단계: 연쇄논박 완료",
                    "free_select": "3단계: 상대 선택",
                    "free_rebuttal": "3단계: 자유논박",
                }
                st.write(f"**단계:** {phase_labels.get(st.session_state.phase, st.session_state.phase)}")
                total = len(st.session_state.state["debate_history"]) if st.session_state.state else 0
                st.write(f"**발언:** {total}건")
                st.divider()
                if st.button("💾 결과 저장 (txt)"):
                    path = save_results()
                    st.success(f"저장 완료!\n{path}")
                if st.button("🔄 처음부터 다시"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

    # ══════════════════════════════════════════════
    # 토픽 선택
    # ══════════════════════════════════════════════
    if st.session_state.phase == "topic_select":
        st.title("🎙️ 멀티에이전트 토론 테스트")
        st.header("토픽 선택")
        topics = load_all_topics()
        topic_names = [f"{t['id']} | {t['title']}" for t in topics]
        selected_idx = st.selectbox("토픽을 선택하세요", range(len(topics)),
                                     format_func=lambda i: topic_names[i])
        if st.button("토픽 확정 → 토론 시작", type="primary"):
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
            st.session_state.messages = []
            st.session_state.opening_ready = False

            # 백그라운드에서 AI 입론 생성 시작
            import copy
            state_copy = copy.deepcopy(state)
            thread = threading.Thread(target=_run_opening_bg, args=(state_copy,), daemon=True)
            thread.start()
            st.session_state._opening_thread = thread

            st.session_state.phase = "opening_user"
            st.rerun()
        return

    # ══════════════════════════════════════════════
    # 1단계: 입론 — 사용자 입력 (AI는 백그라운드 생성)
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "opening_user":
        topic = st.session_state.topic_dict["title"]
        st.header("1단계: 입론")
        st.info(f"**토픽:** {topic}\n\n**사용자 입장:** 찬성(PRO)")

        # 백그라운드 결과 체크 (스레드 종료 or _bg_result 도착)
        thread = st.session_state.get("_opening_thread")
        if not st.session_state.opening_ready:
            if "opening" in _bg_result:
                st.session_state.state = _bg_result.pop("opening")
                st.session_state.opening_ready = True
            elif thread and not thread.is_alive():
                # 스레드 끝났는데 결과가 없으면 동기로 재실행
                state = opening_arguments_node(st.session_state.state)
                st.session_state.state = dict(state)
                st.session_state.opening_ready = True

        if not st.session_state.opening_ready:
            st.warning("🤖 AI 에이전트들이 입론을 생성하고 있습니다... 사용자 입론을 먼저 작성하세요!")
        else:
            st.success("✅ AI 에이전트 입론 생성 완료!")

        with st.form("opening_form"):
            st.subheader("✍️ 사용자 입론")
            intro = st.text_area("자기소개와 입장 표명", placeholder="자신을 소개하고 찬성 입장을 밝히세요.", height=80)
            arg1 = st.text_area("논거 1", placeholder="첫 번째 논거를 구체적으로 작성하세요.", height=120)
            arg2 = st.text_area("논거 2", placeholder="두 번째 논거를 구체적으로 작성하세요.", height=120)
            conclusion = st.text_area("결론", placeholder="핵심 주장을 정리하세요.", height=80)
            submitted = st.form_submit_button("입론 제출", type="primary", use_container_width=True)

        if submitted:
            sections = []
            if intro.strip():
                sections.append(f"### 자기소개와 입장 표명\n{intro.strip()}")
            if arg1.strip():
                sections.append(f"### 논거 1\n{arg1.strip()}")
            if arg2.strip():
                sections.append(f"### 논거 2\n{arg2.strip()}")
            if conclusion.strip():
                sections.append(f"### 결론\n{conclusion.strip()}")

            if not sections:
                st.warning("최소 1개 섹션은 입력해주세요.")
            elif not st.session_state.opening_ready:
                st.warning("AI 에이전트 입론이 아직 생성 중입니다. 잠시 후 다시 시도하세요.")
            else:
                user_opening = "\n\n".join(sections)
                state = st.session_state.state
                user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
                state["debate_history"].append(DebateEntry(
                    turn=user_turn, speaker_id="user", stance=USER_STANCE,
                    phase="opening", content=user_opening,
                    target_id=None, tool_calls_log=[], json_raw="",
                ))
                state["debate_history"].sort(key=lambda e: e["turn"])
                state["phase"] = "chained_rebuttal"
                st.session_state.state = state

                # 입론 결과를 채팅으로 표시
                for entry in state["debate_history"]:
                    if entry["phase"] == "opening":
                        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
                        speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
                        role = "user" if entry["speaker_id"] == "user" else "assistant"
                        add_msg(role, f"**[{speaker} ({s_label}) 입론]**\n\n{entry['content']}")

                add_msg("assistant", "---\n## 2단계: 연쇄논박\nAI 에이전트들이 연쇄논박을 생성하고 있습니다...")

                # 백그라운드에서 AI 연쇄논박 생성 시작
                st.session_state.rebuttal_ready = False
                import copy
                state_copy = copy.deepcopy(st.session_state.state)
                thread = threading.Thread(target=_run_rebuttal_bg, args=(state_copy,), daemon=True)
                thread.start()
                st.session_state._rebuttal_thread = thread

                st.session_state.phase = "rebuttal_user"
                st.rerun()

        # AI 아직 생성 중이면 새로고침 버튼
        if not st.session_state.opening_ready:
            if st.button("🔄 AI 생성 상태 확인"):
                st.rerun()

    # ══════════════════════════════════════════════
    # 2단계: 연쇄논박 — 채팅 표시 + 사용자 입력 (AI는 백그라운드)
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "rebuttal_user":
        render_messages()

        # 백그라운드 결과 체크
        thread = st.session_state.get("_rebuttal_thread")
        if not st.session_state.rebuttal_ready:
            if "rebuttal" in _bg_result:
                st.session_state.state = _bg_result.pop("rebuttal")
                st.session_state.rebuttal_ready = True
            elif thread and not thread.is_alive():
                state = chained_rebuttal_node(st.session_state.state)
                st.session_state.state = dict(state)
                st.session_state.rebuttal_ready = True

        if not st.session_state.rebuttal_ready:
            st.warning("🤖 AI 에이전트들이 연쇄논박을 생성하고 있습니다... 반박을 미리 준비하세요!")
        else:
            # AI 연쇄논박 결과가 아직 채팅에 안 들어갔으면 추가
            if not st.session_state.get("rebuttal_msgs_added"):
                for entry in st.session_state.state["debate_history"]:
                    if entry["phase"] == "chained_rebuttal":
                        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
                        target = entry.get("target_id", "")
                        add_msg("assistant", f"**[{entry['speaker_id']} ({s_label}) → {target}]**\n\n{entry['content']}")

                attacker_id = None
                for e in reversed(st.session_state.state["debate_history"]):
                    if e["phase"] == "chained_rebuttal" and e["target_id"] == "user":
                        attacker_id = e["speaker_id"]
                        break
                st.session_state.rebuttal_target = attacker_id or "agent_1"
                add_msg("assistant", f"⚔️ **{st.session_state.rebuttal_target}**가 사용자를 공격했습니다! 반박해주세요.")
                st.session_state.rebuttal_msgs_added = True
                st.rerun()

            st.success("✅ AI 에이전트 연쇄논박 완료!")

        target = st.session_state.get("rebuttal_target", "agent_1")
        with st.form("rebuttal_form"):
            user_rebuttal = st.text_area(f"✍️ {target}에 대한 반박", height=150)
            submitted = st.form_submit_button("연쇄논박 제출 → 자유논박", type="primary", use_container_width=True)

        if submitted:
            if not user_rebuttal.strip():
                st.warning("연쇄논박을 입력해주세요.")
            elif not st.session_state.rebuttal_ready:
                st.warning("AI 에이전트 연쇄논박이 아직 생성 중입니다. 잠시 후 다시 시도하세요.")
            else:
                state = st.session_state.state
                state["debate_history"].append(DebateEntry(
                    turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                    phase="chained_rebuttal", content=user_rebuttal.strip(),
                    target_id=target, tool_calls_log=[], json_raw="",
                ))
                state["current_turn"] += 1
                state["phase"] = "free_rebuttal"
                st.session_state.state = state

                add_msg("user", f"**[사용자 (찬성) → {target}]**\n\n{user_rebuttal.strip()}")
                add_msg("assistant", "---\n## 3단계: 자유논박\n상대 에이전트를 선택해주세요.")
                st.session_state.phase = "free_select"
                st.rerun()

        # AI 아직 생성 중이면 새로고침 버튼
        if not st.session_state.rebuttal_ready:
            if st.button("🔄 AI 생성 상태 확인"):
                st.rerun()

    # ══════════════════════════════════════════════
    # 3단계: 자유논박 — 상대 선택
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "free_select":
        render_messages()

        opposite = "CON" if USER_STANCE == "PRO" else "PRO"
        opponents = [p for p in st.session_state.personas if p.stance == opposite]
        opponent_names = [f"{p.agent_id} — {p.role_description[:60]}" for p in opponents]
        selected_idx = st.selectbox("상대 에이전트를 선택하세요", range(len(opponents)),
                                     format_func=lambda i: opponent_names[i])
        if st.button("상대 확정 → 자유논박 시작", type="primary"):
            selected = opponents[selected_idx]
            st.session_state.state["selected_opponent_id"] = selected.agent_id
            st.session_state.selected_opponent = selected
            opponent_label = "반대" if selected.stance == "CON" else "찬성"
            add_msg("assistant", f"🥊 **자유논박: 사용자(찬성) ↔ {selected.agent_id}({opponent_label})**\n\n발언을 입력해주세요.")
            st.session_state.phase = "free_rebuttal"
            st.rerun()

    # ══════════════════════════════════════════════
    # 3단계: 자유논박 — 채팅
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "free_rebuttal":
        selected = st.session_state.selected_opponent
        render_messages()

        user_input = st.chat_input("발언을 입력하세요")
        if user_input:
            add_msg("user", user_input)
            state = st.session_state.state
            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                phase="free_rebuttal", content=user_input, target_id=selected.agent_id,
                tool_calls_log=[], json_raw="",
            ))
            state["current_turn"] += 1

            with st.spinner(f"{selected.agent_id} 응답 생성 중..."):
                state = free_rebuttal_node(state)
                state = dict(state)
                st.session_state.state = state

            agent_entries = [
                e for e in state["debate_history"]
                if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"
            ]
            if agent_entries:
                latest = agent_entries[-1]["content"]
                add_msg("assistant", f"**[{selected.agent_id}]** {latest}")
            st.rerun()


if __name__ == "__main__":
    main()
