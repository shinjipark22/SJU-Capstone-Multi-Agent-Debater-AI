"""
streamlit_app.py — LangGraph 메인 그래프 기반 대화형 토론 UI

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
from src.stage2_rebuttal.nodes import build_agent_stance_nums

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


def save_results():
    state = st.session_state.graph_state
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_id = st.session_state.topic_dict["id"]
    output_path = _OUTPUT_DIR / f"streamlit_{topic_id}_{timestamp}.txt"

    lines = [
        f"토픽: {st.session_state.topic_dict['title']}",
        f"토픽 ID: {topic_id}",
        f"포맷: {DEBATE_FORMAT} | 사용자: {USER_STANCE}",
        f"생성 시각: {timestamp}",
        "",
    ]
    for phase_name, phase_label in [
        ("opening", "1단계: 입론"),
        ("chained_rebuttal", "2단계: 연쇄논박"),
        ("free_rebuttal", "3단계: 자유논박"),
        ("role_reversal", "4단계: 역할반전"),
        ("synthesis", "5단계: 종합 및 재개념화"),
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

    if state.get("synthesis_draft"):
        lines.append("=" * 70)
        lines.append(f" 우리의 최적해: {state['synthesis_draft']}")
        lines.append("=" * 70)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def get_display_name(entry, agents, speaking_order):
    if entry["speaker_id"] == "user":
        return "사용자"
    stance_nums = build_agent_stance_nums(agents, speaking_order)
    s_label = "찬성" if entry["stance"] == "PRO" else "반대"
    snum = stance_nums.get(entry["speaker_id"], 1)
    return f"{s_label}{snum}"


def render_history():
    """debate_history의 모든 발언을 채팅으로 렌더링."""
    state = st.session_state.graph_state
    if not state:
        return
    for entry in state.get("debate_history", []):
        display = get_display_name(entry, state["agents"], state["speaking_order"])
        role = "user" if entry["speaker_id"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(f"**[{display}]** {entry['content']}")


def run_graph(input_data):
    """그래프를 실행/재개하고 결과를 session_state에 저장."""
    graph = get_debate_graph()
    config = {"configurable": {"thread_id": st.session_state.session_id}}

    result = graph.invoke(input_data, config=config)

    # 상태 업데이트
    graph_state_obj = graph.get_state(config)
    st.session_state.graph_state = graph_state_obj.values if graph_state_obj else {}
    st.session_state.waiting_for = graph_state_obj.next[0] if graph_state_obj and graph_state_obj.next else ""


def get_phase_info(waiting_for: str) -> tuple:
    """waiting_for 노드명에서 단계 이름과 안내 메시지를 반환."""
    info = {
        "user_opening": ("1단계: 입론", "입론을 작성해주세요."),
        "user_rebuttal": ("2단계: 연쇄논박", "상대의 공격에 반박해주세요."),
        "user_free_rebuttal": ("3단계: 자유논박", "답변과 공격을 입력해주세요."),
        "user_role_reversal": ("4단계: 역할반전", "상대 입장을 옹호하는 발언을 작성해주세요."),
        "user_synthesis": ("5단계: 종합", "최적해에 대한 의견을 입력해주세요."),
        "user_finalize": ("5단계: 최적해 확정", "우리의 최적해를 작성해주세요."),
    }
    return info.get(waiting_for, ("", ""))


def main():
    st.set_page_config(page_title="멀티에이전트 토론", page_icon="🎙️", layout="wide")

    # ── 초기화
    if "phase" not in st.session_state:
        st.session_state.phase = "topic_select"
        st.session_state.graph_state = None
        st.session_state.topic_dict = None
        st.session_state.session_id = None
        st.session_state.waiting_for = ""

    # ── 사이드바
    if st.session_state.phase != "topic_select":
        with st.sidebar:
            st.title("🎙️ 토론")
            if st.session_state.topic_dict:
                st.write(f"**토픽:** {st.session_state.topic_dict['title'][:40]}...")
                phase_name, _ = get_phase_info(st.session_state.waiting_for)
                st.write(f"**단계:** {phase_name or st.session_state.waiting_for}")
                if st.session_state.graph_state:
                    total = len(st.session_state.graph_state.get("debate_history", []))
                    st.write(f"**발언:** {total}건")
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

            # 에이전트 생성 + 초기 State
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

            st.session_state.session_id = f"streamlit_{datetime.now().strftime('%H%M%S')}"
            st.session_state.phase = "running"

            # 그래프 시작 (AI 입론 → user_opening interrupt에서 멈춤)
            with st.spinner("🤖 AI 에이전트들이 입론을 생성하고 있습니다..."):
                run_graph(dict(initial_state))

            st.rerun()
        return

    # ══════════════════════════════════════════════
    # 토론 진행
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "running":
        render_history()

        waiting = st.session_state.waiting_for

        # 토론 완료
        if not waiting:
            st.success("🎉 토론이 완료되었습니다!")
            if st.session_state.graph_state.get("synthesis_draft"):
                st.markdown(f"## 우리의 최적해\n\n{st.session_state.graph_state['synthesis_draft']}")
            return

        phase_name, guide = get_phase_info(waiting)
        st.divider()
        st.subheader(f"✍️ {phase_name}")
        st.info(guide)

        # ── 자유논박: 답변+공격 2칸
        if waiting == "user_free_rebuttal":
            with st.form("free_form"):
                defense = st.text_area("💬 답변 (상대 공격에 대한 반박)", height=100)
                attack = st.text_area("⚔️ 공격 (상대 논거의 허점)", height=100)
                submitted = st.form_submit_button("제출", type="primary", use_container_width=True)

            if submitted and defense.strip() and attack.strip():
                # interrupt가 2번이므로 2번 resume
                graph = get_debate_graph()
                config = {"configurable": {"thread_id": st.session_state.session_id}}

                with st.spinner("🤖 AI가 응답하고 있습니다..."):
                    # 1차 resume: 답변
                    graph.invoke(Command(resume=defense.strip()), config=config)
                    # 2차 resume: 공격 → 다음 interrupt까지 진행
                    run_graph(Command(resume=attack.strip()))

                st.rerun()

        # ── 그 외: 단일 텍스트 입력
        else:
            with st.form("input_form"):
                user_input = st.text_area("입력", height=150,
                                           placeholder="내용을 입력하세요...")
                submitted = st.form_submit_button("제출", type="primary", use_container_width=True)

            if submitted and user_input.strip():
                with st.spinner("🤖 AI가 응답하고 있습니다..."):
                    run_graph(Command(resume=user_input.strip()))
                st.rerun()


if __name__ == "__main__":
    main()
