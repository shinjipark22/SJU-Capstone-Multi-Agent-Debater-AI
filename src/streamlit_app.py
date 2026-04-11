"""
streamlit_app.py — LangGraph 메인 그래프 기반 대화형 토론 UI

graph.stream()으로 에이전트 발언을 실시간 표시.

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
from langgraph.types import Command

from src.graph.main_graph import build_debate_graph
from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, build_initial_state
from src.phase1.stage2_rebuttal.nodes import build_agent_stance_nums

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


@st.cache_resource
def get_debate_graph():
    return build_debate_graph()


# ── 유틸리티 ─────────────────────────────────────────────────────────────

def get_display_name(entry, agents, speaking_order):
    if entry["speaker_id"] == "user":
        return "사용자"
    stance_nums = build_agent_stance_nums(agents, speaking_order)
    s_label = "찬성" if entry["stance"] == "PRO" else "반대"
    snum = stance_nums.get(entry["speaker_id"], 1)
    return f"{s_label}{snum}"


def get_phase_label(waiting_for: str) -> tuple:
    """(단계명, 안내 메시지)를 반환."""
    info = {
        "user_opening": (
            "1단계: 입론",
            "모든 토론자의 입론이 끝났습니다. 이제 사용자의 입론을 작성해주세요."
        ),
        "user_rebuttal": (
            "2단계: 연쇄논박",
            "상대가 사용자의 입론을 공격했습니다! 반박해주세요."
        ),
        "user_free_rebuttal": (
            "3단계: 자유논박",
            "상대의 공격에 답변하고, 상대 논거의 허점을 공격하세요."
        ),
        "_attack_pending": (
            "3단계: 자유논박",
            "답변이 제출되었습니다. 이제 상대 논거를 공격하세요."
        ),
        "user_role_reversal": (
            "4단계: 역할반전",
            "이제 역할을 바꿔서 상대 입장을 옹호하는 발언을 작성해주세요."
        ),
        "user_synthesis": (
            "5단계: 종합",
            "토론자들의 의견을 들었습니다. 사용자의 생각을 말씀해주세요."
        ),
        "user_finalize": (
            "5단계: 최적해 확정",
            "회의가 충분히 진행되었습니다. 최종 결론을 대표로 작성해주세요."
        ),
    }
    return info.get(waiting_for, ("진행 중...", ""))


def save_results():
    msgs = st.session_state.messages
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_id = st.session_state.topic_dict["id"]
    output_path = _OUTPUT_DIR / f"streamlit_{topic_id}_{timestamp}.txt"
    lines = [f"토픽: {st.session_state.topic_dict['title']}", f"시각: {timestamp}", ""]
    for msg in msgs:
        role = "사용자" if msg["role"] == "user" else "AI"
        lines.append(f"[{role}] {msg['content']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


# ── 그래프 실행 (stream 모드) ─────────────────────────────────────────────

def stream_graph(input_data, config):
    """그래프를 stream 모드로 실행하고, 새 발언이 나올 때마다 채팅에 실시간 표시."""
    graph = get_debate_graph()
    prev_count = len(st.session_state.get("seen_entries", []))

    for chunk in graph.stream(input_data, config=config, stream_mode="values"):
        if not isinstance(chunk, dict):
            continue
        history = chunk.get("debate_history", [])
        # 새로 추가된 엔트리만
        new_entries = history[prev_count:]
        agents = chunk.get("agents", st.session_state.get("agents", []))
        speaking_order = chunk.get("speaking_order", st.session_state.get("speaking_order", []))

        for entry in new_entries:
            display = get_display_name(entry, agents, speaking_order)
            if entry["speaker_id"] == "user":
                with st.chat_message("user"):
                    st.markdown(f"**{display}**\n\n{entry['content']}")
                st.session_state.messages.append({"role": "user", "content": f"**{display}**\n\n{entry['content']}"})
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**{display}**\n\n{entry['content']}")
                st.session_state.messages.append({"role": "assistant", "content": f"**{display}**\n\n{entry['content']}"})
            prev_count += 1

        st.session_state.seen_entries = history[:prev_count]

    # 그래프 상태 업데이트
    graph_state = graph.get_state(config)
    st.session_state.waiting_for = graph_state.next[0] if graph_state and graph_state.next else ""
    st.session_state.is_finished = not bool(graph_state and graph_state.next)

    # 단계 전환 안내 메시지 추가
    if st.session_state.waiting_for and not st.session_state.is_finished:
        _, guide = get_phase_label(st.session_state.waiting_for)
        if guide:
            with st.chat_message("assistant"):
                st.markdown(f"💬 {guide}")
            st.session_state.messages.append({"role": "assistant", "content": f"💬 {guide}"})


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="멀티에이전트 토론", page_icon="🎙️", layout="wide")

    # 초기화
    if "phase" not in st.session_state:
        st.session_state.phase = "topic_select"
        st.session_state.messages = []
        st.session_state.seen_entries = []
        st.session_state.waiting_for = ""
        st.session_state.is_finished = False
        st.session_state.topic_dict = None
        st.session_state.session_id = None
        st.session_state.agents = []
        st.session_state.speaking_order = []

    # 사이드바
    if st.session_state.phase != "topic_select":
        with st.sidebar:
            st.title("🎙️ 토론")
            if st.session_state.topic_dict:
                st.write(f"**토픽:** {st.session_state.topic_dict['title'][:30]}...")
                phase_label, _ = get_phase_label(st.session_state.waiting_for)
                st.write(f"**단계:** {phase_label}")
                st.write(f"**발언:** {len(st.session_state.seen_entries)}건")
            st.divider()
            if st.button("💾 결과 저장"):
                path = save_results()
                st.success(f"저장: {path}")
            if st.button("🔄 처음부터"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # ══════════════════════════════════════════════
    # 토픽 선택
    # ══════════════════════════════════════════════
    if st.session_state.phase == "topic_select":
        st.title("🎙️ 멀티에이전트 토론")
        topics = load_all_topics()
        topic_names = [f"{t['id']} | {t['title']}" for t in topics]
        selected_idx = st.selectbox("토픽 선택", range(len(topics)),
                                     format_func=lambda i: topic_names[i])
        if st.button("토론 시작", type="primary"):
            topic_dict = topics[selected_idx]
            st.session_state.topic_dict = topic_dict

            personas = create_agents(
                topic=topic_dict, debate_format=DEBATE_FORMAT,
                user_stance=USER_STANCE, agent_intensities=AGENT_INTENSITIES,
            )
            snapshots = [
                AgentSnapshot(
                    agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
                    role_description=p.role_description, system_prompt=p.system_prompt,
                    focus_area=p.focus_area,
                ) for p in personas
            ]
            initial_state = build_initial_state(
                topic=topic_dict["title"], user_stance=USER_STANCE,
                user_intensity=USER_INTENSITY, agents=snapshots,
                topic_id=topic_dict["id"],
            )
            st.session_state.agents = snapshots
            st.session_state.speaking_order = initial_state["speaking_order"]
            st.session_state.session_id = f"st_{datetime.now().strftime('%H%M%S')}"
            st.session_state.phase = "running"
            st.session_state.messages = []
            st.session_state.seen_entries = []

            config = {"configurable": {"thread_id": st.session_state.session_id}}

            # 시스템 메시지
            with st.chat_message("assistant"):
                st.markdown(f"🎙️ **토론 시작:** {topic_dict['title']}\n\nAI 에이전트들이 입론을 준비하고 있습니다...")
            st.session_state.messages.append({"role": "assistant", "content": f"🎙️ **토론 시작:** {topic_dict['title']}"})

            # 그래프 시작 (stream)
            stream_graph(dict(initial_state), config)

            st.rerun()
        return

    # ══════════════════════════════════════════════
    # 토론 진행
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "running":
        # 채팅 히스토리 렌더
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 완료
        if st.session_state.is_finished:
            st.success("🎉 토론이 완료되었습니다!")
            return

        waiting = st.session_state.waiting_for
        phase_label, guide = get_phase_label(waiting)

        # 안내 메시지 (히스토리에 없으면 표시)
        guide_msg = f"💬 {guide}"
        if not any(m["content"] == guide_msg for m in st.session_state.messages):
            with st.chat_message("assistant"):
                st.markdown(guide_msg)
            st.session_state.messages.append({"role": "assistant", "content": guide_msg})

        # ── 입론: 구조화 폼
        if waiting == "user_opening":
            with st.form("opening_form"):
                st.subheader("✍️ 사용자 입론")
                intro = st.text_area("자기소개와 입장 표명", placeholder="자신을 소개하고 찬성/반대 입장을 밝히세요.", height=80)
                arg1 = st.text_area("논거 1", placeholder="첫 번째 논거를 작성하세요.", height=120)
                arg2 = st.text_area("논거 2", placeholder="두 번째 논거를 작성하세요.", height=120)
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
                else:
                    user_opening = "\n\n".join(sections)
                    with st.chat_message("user"):
                        st.markdown(f"**사용자**\n\n{user_opening}")
                    st.session_state.messages.append({"role": "user", "content": f"**사용자**\n\n{user_opening}"})

                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    stream_graph(Command(resume=user_opening), config)
                    st.rerun()

        # ── 자유논박: 답변+공격
        elif waiting == "user_free_rebuttal":
            defense = st.chat_input("💬 답변 (상대 공격에 반박)")
            if defense:
                # 답변 표시
                with st.chat_message("user"):
                    st.markdown(f"**[답변]** {defense}")
                st.session_state.messages.append({"role": "user", "content": f"**[답변]** {defense}"})

                # 1차 resume (답변)
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                graph = get_debate_graph()
                graph.invoke(Command(resume=defense), config=config)

                # 공격 입력 대기 상태로 전환
                st.session_state.waiting_for = "_attack_pending"
                st.rerun()

        elif waiting == "_attack_pending":
            attack = st.chat_input("⚔️ 공격 (상대 논거의 허점)")
            if attack:
                with st.chat_message("user"):
                    st.markdown(f"**[공격]** {attack}")
                st.session_state.messages.append({"role": "user", "content": f"**[공격]** {attack}"})

                config = {"configurable": {"thread_id": st.session_state.session_id}}

                # 2차 resume (공격) → stream으로 AI 응답 실시간 표시
                stream_graph(Command(resume=attack), config)
                st.rerun()

        # ── 그 외: 단일 입력 (chat_input은 입론/자유논박 외에서만)
        elif waiting not in ("user_opening", "user_free_rebuttal", "_attack_pending"):
            user_input = st.chat_input(guide)
            if user_input:
                with st.chat_message("user"):
                    st.markdown(f"**사용자**\n\n{user_input}")
                st.session_state.messages.append({"role": "user", "content": f"**사용자**\n\n{user_input}"})

                config = {"configurable": {"thread_id": st.session_state.session_id}}

                # resume → stream으로 AI 응답 실시간 표시
                stream_graph(Command(resume=user_input), config)
                st.rerun()


if __name__ == "__main__":
    main()
