"""
streamlit_app.py — 1~5단계 대화형 토론 UI (직접 함수 호출 방식)

LangGraph interrupt/resume는 FastAPI 전용.
Streamlit은 st.session_state로 상태 관리 + 노드 함수 직접 호출.

실행:
    streamlit run src/streamlit_app.py --server.port 8501
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, DebateEntry, build_initial_state
from src.phase1.stage1_opening.nodes import opening_arguments_node
from src.phase1.stage2_rebuttal.nodes import chained_rebuttal_node, build_agent_stance_nums
from src.phase1.stage3_free_rebuttal.nodes import free_rebuttal_node, should_end_free_rebuttal
from src.phase1.stage4_role_reversal.nodes import role_reversal_node
from src.phase1.stage5_synthesis.nodes import synthesis_node, synthesis_discuss_node, should_end_synthesis

_DATA_PATH = Path(__file__).parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent / "test_results"

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
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_id = st.session_state.topic_dict["id"]
    output_path = _OUTPUT_DIR / f"streamlit_{topic_id}_{timestamp}.txt"

    lines = [
        f"토픽: {st.session_state.topic_dict['title']}",
        f"포맷: {DEBATE_FORMAT} | 사용자: {USER_STANCE}",
        f"시각: {timestamp}", "",
    ]
    for phase_name, label in [
        ("opening", "1단계: 입론"), ("chained_rebuttal", "2단계: 연쇄논박"),
        ("free_rebuttal", "3단계: 자유논박"), ("role_reversal", "4단계: 역할반전"),
        ("synthesis", "5단계: 종합"),
    ]:
        entries = [e for e in state["debate_history"] if e["phase"] == phase_name]
        lines.append(f"{'='*60}\n {label} ({len(entries)}건)\n{'='*60}")
        for e in entries:
            s = "찬성" if e["stance"] == "PRO" else "반대"
            sp = "사용자" if e["speaker_id"] == "user" else e["speaker_id"]
            lines.append(f"[{sp} ({s})] {e['content']}\n")
    if state.get("synthesis_draft"):
        lines.append(f"{'='*60}\n우리의 최적해: {state['synthesis_draft']}\n{'='*60}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def add_msg(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def display_name(entry, state):
    if entry["speaker_id"] == "user":
        return "사용자"
    nums = build_agent_stance_nums(state["agents"], state["speaking_order"])
    s = "찬성" if entry["stance"] == "PRO" else "반대"
    return f"{s}{nums.get(entry['speaker_id'], 1)}"


def main():
    st.set_page_config(page_title="멀티에이전트 토론", page_icon="🎙️", layout="wide")

    if "phase" not in st.session_state:
        st.session_state.phase = "topic_select"
        st.session_state.state = None
        st.session_state.topic_dict = None
        st.session_state.messages = []

    # 사이드바
    if st.session_state.phase != "topic_select":
        with st.sidebar:
            st.title("🎙️ 토론")
            if st.session_state.topic_dict:
                st.write(f"**토픽:** {st.session_state.topic_dict['title'][:30]}...")
                st.write(f"**단계:** {st.session_state.phase}")
                if st.session_state.state:
                    st.write(f"**발언:** {len(st.session_state.state['debate_history'])}건")
            st.divider()
            if st.button("💾 결과 저장"):
                st.success(f"저장: {save_results()}")
            if st.button("🔄 처음부터"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    # ══════════════════════════════════════
    # 토픽 선택
    # ══════════════════════════════════════
    if st.session_state.phase == "topic_select":
        st.title("🎙️ 멀티에이전트 토론")
        topics = load_all_topics()
        names = [f"{t['id']} | {t['title']}" for t in topics]
        idx = st.selectbox("토픽 선택", range(len(topics)), format_func=lambda i: names[i])
        if st.button("토론 시작", type="primary"):
            td = topics[idx]
            st.session_state.topic_dict = td
            personas = create_agents(topic=td, debate_format=DEBATE_FORMAT,
                                      user_stance=USER_STANCE, agent_intensities=AGENT_INTENSITIES)
            snaps = [AgentSnapshot(agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
                                   role_description=p.role_description, system_prompt=p.system_prompt,
                                   focus_area=p.focus_area) for p in personas]
            st.session_state.state = build_initial_state(
                topic=td["title"], user_stance=USER_STANCE, user_intensity=USER_INTENSITY,
                agents=snaps, topic_id=td["id"])
            st.session_state.messages = []
            st.session_state.phase = "opening_user"
            st.rerun()
        return

    # ══════════════════════════════════════
    # 1단계: 입론
    # ══════════════════════════════════════
    elif st.session_state.phase == "opening_user":
        topic = st.session_state.topic_dict["title"]
        st.header("1단계: 입론")
        st.info(f"**토픽:** {topic}\n\n**사용자 입장:** {'찬성' if USER_STANCE == 'PRO' else '반대'}\n\n"
                f"입론을 작성하면 AI 에이전트들도 입론을 생성합니다.")

        with st.form("opening_form"):
            st.subheader("✍️ 사용자 입론")
            intro = st.text_area("자기소개와 입장 표명", placeholder="자신을 소개하고 입장을 밝히세요.", height=80)
            arg1 = st.text_area("논거 1", placeholder="첫 번째 논거", height=120)
            arg2 = st.text_area("논거 2", placeholder="두 번째 논거", height=120)
            conclusion = st.text_area("결론", placeholder="핵심 주장 정리", height=80)
            submitted = st.form_submit_button("입론 제출", type="primary", use_container_width=True)

        if submitted:
            sections = []
            if intro.strip(): sections.append(f"### 자기소개와 입장 표명\n{intro.strip()}")
            if arg1.strip(): sections.append(f"### 논거 1\n{arg1.strip()}")
            if arg2.strip(): sections.append(f"### 논거 2\n{arg2.strip()}")
            if conclusion.strip(): sections.append(f"### 결론\n{conclusion.strip()}")
            if not sections:
                st.warning("최소 1개 섹션을 입력해주세요.")
                return
            user_opening = "\n\n".join(sections)

            with st.spinner("🤖 AI 에이전트들이 입론을 생성하고 있습니다..."):
                state = opening_arguments_node(st.session_state.state)
                st.session_state.state = dict(state)

            state = st.session_state.state
            user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
            state["debate_history"].append(DebateEntry(
                turn=user_turn, speaker_id="user", stance=USER_STANCE,
                phase="opening", content=user_opening, target_id=None,
                tool_calls_log=[], json_raw=""))
            state["debate_history"].sort(key=lambda e: e["turn"])
            state["phase"] = "chained_rebuttal"

            add_msg("assistant", "## 1단계: 입론 결과")
            for e in state["debate_history"]:
                if e["phase"] == "opening":
                    dn = display_name(e, state)
                    role = "user" if e["speaker_id"] == "user" else "assistant"
                    add_msg(role, f"**[{dn} 입론]**\n\n{e['content']}")

            st.session_state.phase = "rebuttal_user"
            st.rerun()

    # ══════════════════════════════════════
    # 2단계: 연쇄논박
    # ══════════════════════════════════════
    elif st.session_state.phase == "rebuttal_user":
        render_messages()

        if not st.session_state.get("rebuttal_done"):
            with st.spinner("🤖 AI 연쇄논박 생성 중..."):
                state = chained_rebuttal_node(st.session_state.state)
                st.session_state.state = dict(state)
                st.session_state.rebuttal_done = True

            add_msg("assistant", "---\n## 2단계: 연쇄논박")
            for e in st.session_state.state["debate_history"]:
                if e["phase"] == "chained_rebuttal":
                    dn = display_name(e, st.session_state.state)
                    add_msg("assistant", f"**[{dn} → {e.get('target_id','')}]**\n\n{e['content']}")

            attacker = None
            for e in reversed(st.session_state.state["debate_history"]):
                if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
                    attacker = e["speaker_id"]
                    break
            st.session_state.rebuttal_target = attacker or "agent_1"
            add_msg("assistant", f"⚔️ **{st.session_state.rebuttal_target}**가 사용자를 공격했습니다! 반박해주세요.")
            st.rerun()

        target = st.session_state.get("rebuttal_target", "agent_1")
        user_rebuttal = st.chat_input(f"💬 {target}에 대한 반박을 입력하세요")
        if user_rebuttal:
            state = st.session_state.state
            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                phase="chained_rebuttal", content=user_rebuttal.strip(),
                target_id=target, tool_calls_log=[], json_raw=""))
            state["current_turn"] += 1
            state["phase"] = "free_rebuttal"

            add_msg("user", f"**[사용자 → {target}]**\n\n{user_rebuttal.strip()}")
            add_msg("assistant", "---\n## 3단계: 자유논박")

            # 상대 자동 선택
            opposite = "CON" if USER_STANCE == "PRO" else "PRO"
            opp = next((a for a in state["agents"] if a["stance"] == opposite), None)
            if opp:
                state["selected_opponent_id"] = opp["agent_id"]

            st.session_state.phase = "free_rebuttal"
            st.rerun()

    # ══════════════════════════════════════
    # 3단계: 자유논박 (4.5턴 고정)
    # ══════════════════════════════════════
    elif st.session_state.phase == "free_rebuttal":
        render_messages()
        state = st.session_state.state
        selected_id = state.get("selected_opponent_id", "agent_1")
        user_turn_count = state.get("free_rebuttal_user_turns", 0)

        agent_fr = [e for e in state["debate_history"] if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]

        # 에이전트 응답이 필요한지 판단:
        # - 첫 공격 (agent_fr 없음)
        # - 사용자가 제출했는데 에이전트 응답이 아직 없음 (답변+공격 수 < 사용자 턴 수 + 1)
        user_fr_count = len([e for e in state["debate_history"] if e["speaker_id"] == "user" and e["phase"] == "free_rebuttal"])
        agent_should_respond = not agent_fr or (user_fr_count > 0 and len(agent_fr) < (user_fr_count // 2) + 1)
        if agent_should_respond:
            if should_end_free_rebuttal(state):
                # 마지막 답변 후 역할반전
                with st.spinner("🤖 최종 답변 생성 중..."):
                    state = free_rebuttal_node(state)
                    st.session_state.state = dict(state)
                new_entries = [e for e in st.session_state.state["debate_history"]
                               if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
                if new_entries:
                    e = new_entries[-1]
                    dn = display_name(e, st.session_state.state)
                    add_msg("assistant", f"**[{dn} - 답변]** {e['content']}")
                st.session_state.state["phase"] = "role_reversal"
                add_msg("assistant", "---\n## 4단계: 역할반전\n자유논박이 종료되었습니다.")
                st.session_state.phase = "role_reversal"
                st.rerun()
            else:
                label = "공격 생성" if not agent_fr else "답변+공격 생성"
                with st.spinner(f"🤖 {selected_id} {label} 중..."):
                    state = free_rebuttal_node(state)
                    st.session_state.state = dict(state)
                new_entries = [e for e in st.session_state.state["debate_history"]
                               if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
                for e in new_entries[len(agent_fr):]:
                    dn = display_name(e, st.session_state.state)
                    lbl = "답변" if len(new_entries) > len(agent_fr) + 1 and e == new_entries[len(agent_fr)] else "공격"
                    add_msg("assistant", f"**[{dn} - {lbl}]** {e['content']}")
                st.rerun()

        # 사용자 답변+공격
        turn_label = f"({user_turn_count + 1}/2)"
        st.markdown(f"##### 💬 답변 {turn_label}")
        defense = st.text_area("상대 공격에 반박하세요", key=f"defense_{user_turn_count}", height=100)
        st.markdown(f"##### ⚔️ 공격 {turn_label}")
        attack = st.text_area("상대 논거를 공격하세요", key=f"attack_{user_turn_count}", height=100)

        if st.button("답변+공격 제출", type="primary"):
            if not defense.strip() or not attack.strip():
                st.warning("답변과 공격을 모두 입력해주세요.")
            else:
                add_msg("user", f"**[답변]** {defense.strip()}")
                state["debate_history"].append(DebateEntry(
                    turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                    phase="free_rebuttal", content=defense.strip(),
                    target_id=selected_id, tool_calls_log=[], json_raw=""))
                state["current_turn"] += 1

                add_msg("user", f"**[공격]** {attack.strip()}")
                state["debate_history"].append(DebateEntry(
                    turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                    phase="free_rebuttal", content=attack.strip(),
                    target_id=selected_id, tool_calls_log=[], json_raw=""))
                state["current_turn"] += 1
                state["free_rebuttal_user_turns"] = user_turn_count + 1
                st.session_state.state = state
                st.rerun()

    # ══════════════════════════════════════
    # 4단계: 역할반전
    # ══════════════════════════════════════
    elif st.session_state.phase == "role_reversal":
        render_messages()
        state = st.session_state.state

        if not st.session_state.get("role_reversal_done"):
            with st.spinner("🔄 역할반전 발언 생성 중..."):
                state = role_reversal_node(state)
                st.session_state.state = dict(state)
                st.session_state.role_reversal_done = True

            for e in state["debate_history"]:
                if e["phase"] == "role_reversal" and e["speaker_id"] != "user":
                    dn = display_name(e, state)
                    rev_label = "찬성" if e["stance"] == "PRO" else "반대"
                    add_msg("assistant", f"**[{dn} → {rev_label} 옹호]**\n\n{e['content']}")

            reversed_label = "반대" if USER_STANCE == "PRO" else "찬성"
            add_msg("assistant", f"🔄 이제 사용자가 **{reversed_label}** 입장을 옹호하는 발언을 작성해주세요.")
            st.rerun()

        reversed_label = "반대" if USER_STANCE == "PRO" else "찬성"
        st.subheader(f"✍️ 역할반전: {reversed_label} 입장 옹호")
        with st.form("rr_form"):
            arg1 = st.text_area("논거 1", placeholder="상대 관점에서 첫 번째 논거", height=120)
            arg2 = st.text_area("논거 2", placeholder="상대 관점에서 두 번째 논거", height=120)
            conclusion = st.text_area("결론", placeholder="상대 관점 결론", height=80)
            submitted = st.form_submit_button("역할반전 제출", type="primary", use_container_width=True)

        if submitted:
            sections = []
            if arg1.strip(): sections.append(f"### 논거 1\n{arg1.strip()}")
            if arg2.strip(): sections.append(f"### 논거 2\n{arg2.strip()}")
            if conclusion.strip(): sections.append(f"### 결론\n{conclusion.strip()}")
            if not sections:
                st.warning("최소 1개 섹션을 입력해주세요.")
            else:
                user_rr = "\n\n".join(sections)
                reversed_stance = "CON" if USER_STANCE == "PRO" else "PRO"
                state["debate_history"].append(DebateEntry(
                    turn=state["current_turn"], speaker_id="user",
                    stance=reversed_stance, phase="role_reversal",
                    content=user_rr, target_id=None, tool_calls_log=[], json_raw=""))
                state["current_turn"] += 1
                state["phase"] = "synthesis"
                add_msg("user", f"**[사용자 → {reversed_label} 옹호]**\n\n{user_rr}")
                add_msg("assistant", "---\n## 5단계: 종합 및 재개념화")
                st.session_state.phase = "synthesis"
                st.rerun()

    # ══════════════════════════════════════
    # 5단계: 종합 회의 (2턴 고정)
    # ══════════════════════════════════════
    elif st.session_state.phase == "synthesis":
        render_messages()
        state = st.session_state.state
        nums = build_agent_stance_nums(state["agents"], state["speaking_order"])

        syn_entries = [e for e in state["debate_history"] if e["phase"] == "synthesis"]
        if not syn_entries:
            with st.spinner("🧠 토론자들이 최적해에 대한 의견을 제시하고 있습니다..."):
                state = synthesis_node(state)
                st.session_state.state = dict(state)
            for e in state["debate_history"]:
                if e["phase"] == "synthesis" and e["speaker_id"] != "user":
                    dn = display_name(e, state)
                    add_msg("assistant", f"**[{dn}]** {e['content']}")
            add_msg("assistant", "💬 토론자들의 의견을 들었습니다. 사용자의 생각을 말씀해주세요.")
            st.rerun()

        user_syn_count = state.get("synthesis_user_turns", 0)

        if should_end_synthesis(state):
            st.session_state.phase = "synthesis_final"
            st.rerun()

        turn_label = f"({user_syn_count + 1}/2)"
        user_opinion = st.chat_input(f"💬 최적해에 대한 의견 {turn_label}")
        if user_opinion:
            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                phase="synthesis", content=user_opinion.strip(),
                target_id=None, tool_calls_log=[], json_raw=""))
            state["current_turn"] += 1
            state["synthesis_user_turns"] = user_syn_count + 1
            add_msg("user", f"**[사용자]** {user_opinion.strip()}")

            with st.spinner("🧠 토론자들이 응답하고 있습니다..."):
                before = len([e for e in state["debate_history"] if e["phase"] == "synthesis" and e["speaker_id"] != "user"])
                state = synthesis_discuss_node(state)
                st.session_state.state = dict(state)
            after = [e for e in state["debate_history"] if e["phase"] == "synthesis" and e["speaker_id"] != "user"]
            for e in after[before:]:
                dn = display_name(e, state)
                add_msg("assistant", f"**[{dn}]** {e['content']}")
            st.rerun()

    # ══════════════════════════════════════
    # 5단계: 최적해 확정
    # ══════════════════════════════════════
    elif st.session_state.phase == "synthesis_final":
        render_messages()
        state = st.session_state.state

        st.subheader("✍️ 우리의 최적해")
        st.info("회의 내용을 바탕으로 최종 결론을 작성해주세요.")
        with st.form("final_form"):
            user_final = st.text_area("우리의 최적해", height=200,
                                       placeholder="예: AI는 빠르게 일자리를 바꾸지만...")
            submitted = st.form_submit_button("우리의 최적해 확정", type="primary", use_container_width=True)
        if submitted and user_final.strip():
            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                phase="synthesis", content=user_final.strip(),
                target_id=None, tool_calls_log=[], json_raw=""))
            state["synthesis_draft"] = user_final.strip()
            state["is_finished"] = True
            add_msg("assistant", f"---\n## 우리의 최적해\n\n{user_final.strip()}")
            st.session_state.phase = "finished"
            st.rerun()

    # ══════════════════════════════════════
    # 완료
    # ══════════════════════════════════════
    elif st.session_state.phase == "finished":
        render_messages()
        st.success("🎉 토론이 완료되었습니다! 사이드바에서 결과를 저장할 수 있습니다.")


if __name__ == "__main__":
    main()
