"""
streamlit_test.py — 1~5단계 대화형 토론 테스트 UI (채팅 형태)

실행:
    streamlit run src/stage5_synthesis/streamlit_test.py --server.port 8501
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
from src.stage2_rebuttal.nodes import chained_rebuttal_node, build_agent_stance_nums
from src.stage3_free_rebuttal.nodes import free_rebuttal_node
from src.stage4_role_reversal.nodes import role_reversal_node
from src.stage5_synthesis.nodes import synthesis_node

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


def add_msg(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main():
    st.set_page_config(page_title="토론 테스트 (1~5단계)", page_icon="🎙️", layout="wide")
    init_session()

    # ── 사이드바
    if st.session_state.phase != "topic_select":
        with st.sidebar:
            st.title("🎙️ 토론 테스트")
            if st.session_state.topic_dict:
                st.write(f"**토픽:** {st.session_state.topic_dict['title'][:40]}...")
                st.write(f"**단계:** {st.session_state.phase}")
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
                topic_id=topic_dict["id"],
            )
            st.session_state.state = state
            st.session_state.messages = []
            st.session_state.phase = "opening_user"
            st.rerun()
        return

    # ══════════════════════════════════════════════
    # 1단계: 입론
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "opening_user":
        topic = st.session_state.topic_dict["title"]
        st.header("1단계: 입론")
        st.info(f"**토픽:** {topic}\n\n**사용자 입장:** 찬성(PRO)\n\n"
                f"입론을 작성하고 제출하면 AI 에이전트들도 입론을 생성합니다.")

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
                return

            user_opening = "\n\n".join(sections)

            with st.spinner("🤖 AI 에이전트들이 입론을 생성하고 있습니다..."):
                state = opening_arguments_node(st.session_state.state)
                st.session_state.state = dict(state)

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

            add_msg("assistant", "## 1단계: 입론 결과")
            for entry in state["debate_history"]:
                if entry["phase"] == "opening":
                    s_label = "찬성" if entry["stance"] == "PRO" else "반대"
                    speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
                    role = "user" if entry["speaker_id"] == "user" else "assistant"
                    add_msg(role, f"**[{speaker} ({s_label}) 입론]**\n\n{entry['content']}")

            st.session_state.phase = "rebuttal_user"
            st.rerun()

    # ══════════════════════════════════════════════
    # 2단계: 연쇄논박
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "rebuttal_user":
        render_messages()

        if not st.session_state.get("rebuttal_done"):
            with st.spinner("🤖 AI 에이전트들이 연쇄논박을 생성하고 있습니다..."):
                state = chained_rebuttal_node(st.session_state.state)
                st.session_state.state = dict(state)
                st.session_state.rebuttal_done = True

            add_msg("assistant", "---\n## 2단계: 연쇄논박 결과")
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
            st.rerun()

        target = st.session_state.get("rebuttal_target", "agent_1")
        with st.form("rebuttal_form"):
            user_rebuttal = st.text_area(f"✍️ {target}에 대한 반박", height=150)
            submitted = st.form_submit_button("연쇄논박 제출 → 자유논박", type="primary", use_container_width=True)

        if submitted:
            if not user_rebuttal.strip():
                st.warning("연쇄논박을 입력해주세요.")
                return

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
    # 3단계: 자유논박 — 답변+공격
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "free_rebuttal":
        selected = st.session_state.selected_opponent
        render_messages()

        state = st.session_state.state
        agent_fr = [e for e in state["debate_history"] if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"]
        if not agent_fr:
            with st.spinner(f"{selected.agent_id} 첫 공격 생성 중..."):
                state = free_rebuttal_node(state)
                state = dict(state)
                st.session_state.state = state
            for e in state["debate_history"]:
                if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal":
                    add_msg("assistant", f"**[{selected.agent_id} - 공격]** {e['content']}")
            st.rerun()

        st.markdown("##### 💬 답변 (상대 공격에 대한 반박)")
        user_defense = st.text_area("상대의 공격에 반박하세요", key="user_defense", height=100)
        st.markdown("##### ⚔️ 공격 (상대 입론/발언의 허점 공격)")
        user_attack = st.text_area("상대의 논거를 공격하세요", key="user_attack", height=100)

        col1, col2 = st.columns(2)
        with col1:
            submit_free = st.button("발언 제출", type="primary")
        with col2:
            end_free = st.button("자유논박 종료 → 역할반전", type="secondary")

        if submit_free:
            if user_defense or user_attack:
                if user_defense:
                    add_msg("user", f"**[답변]** {user_defense}")
                    state["debate_history"].append(DebateEntry(
                        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                        phase="free_rebuttal", content=user_defense, target_id=selected.agent_id,
                        tool_calls_log=[], json_raw="",
                    ))
                    state["current_turn"] += 1

                if user_attack:
                    add_msg("user", f"**[공격]** {user_attack}")
                    state["debate_history"].append(DebateEntry(
                        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
                        phase="free_rebuttal", content=user_attack, target_id=selected.agent_id,
                        tool_calls_log=[], json_raw="",
                    ))
                    state["current_turn"] += 1

                with st.spinner(f"{selected.agent_id} 답변+공격 생성 중..."):
                    before_count = len([e for e in state["debate_history"] if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"])
                    state = free_rebuttal_node(state)
                    state = dict(state)
                    st.session_state.state = state

                all_agent = [e for e in state["debate_history"] if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"]
                new_entries = all_agent[before_count:]
                for e in new_entries:
                    label = "답변" if len(new_entries) > 1 and e == new_entries[0] else "공격"
                    add_msg("assistant", f"**[{selected.agent_id} - {label}]** {e['content']}")
                st.rerun()

        if end_free:
            state["phase"] = "role_reversal"
            st.session_state.state = state
            add_msg("assistant", "---\n## 4단계: 역할 반전\n자유논박이 종료되었습니다. 이제 역할반전을 진행합니다.")
            st.session_state.phase = "role_reversal"
            st.rerun()

    # ══════════════════════════════════════════════
    # 4단계: 역할반전
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "role_reversal":
        render_messages()

        state = st.session_state.state

        if not st.session_state.get("role_reversal_done"):
            with st.spinner("🔄 상대팀 대표 AI가 역할반전 발언을 생성하고 있습니다..."):
                state = role_reversal_node(state)
                state = dict(state)
                st.session_state.state = state
                st.session_state.role_reversal_done = True

            rr_entries = [e for e in state["debate_history"] if e["phase"] == "role_reversal" and e["speaker_id"] != "user"]
            stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])
            for entry in rr_entries:
                original_stance = None
                for a in state["agents"]:
                    if a["agent_id"] == entry["speaker_id"]:
                        original_stance = a["stance"]
                        break
                orig_label = "찬성" if original_stance == "PRO" else "반대"
                rev_label = "찬성" if entry["stance"] == "PRO" else "반대"
                snum = stance_nums.get(entry["speaker_id"], 1)
                display = f"{orig_label} 에이전트{snum}"
                add_msg("assistant", f"**[{display} → {rev_label} 옹호 (역할반전)]**\n\n{entry['content']}")

            reversed_user_label = "반대" if USER_STANCE == "PRO" else "찬성"
            add_msg("assistant", f"🔄 이제 사용자가 **{reversed_user_label}** 입장을 옹호하는 발언을 작성해주세요.")
            st.rerun()

        reversed_user_stance = "CON" if USER_STANCE == "PRO" else "PRO"
        reversed_label = "반대" if USER_STANCE == "PRO" else "찬성"

        st.subheader(f"✍️ 역할반전: {reversed_label} 입장 옹호")
        st.info(f"원래 사용자는 {'찬성' if USER_STANCE == 'PRO' else '반대'} 입장이었지만, "
                f"지금은 **{reversed_label}** 입장을 옹호해야 합니다.")

        with st.form("role_reversal_form"):
            arg1 = st.text_area("논거 1", placeholder="상대 관점에서 첫 번째 논거를 작성하세요.", height=120)
            arg2 = st.text_area("논거 2", placeholder="상대 관점에서 두 번째 논거를 작성하세요.", height=120)
            conclusion = st.text_area("결론", placeholder="상대 관점에서 결론을 정리하세요.", height=80)
            submitted = st.form_submit_button("역할반전 발언 제출", type="primary", use_container_width=True)

        if submitted:
            sections = []
            if arg1.strip():
                sections.append(f"### 논거 1\n{arg1.strip()}")
            if arg2.strip():
                sections.append(f"### 논거 2\n{arg2.strip()}")
            if conclusion.strip():
                sections.append(f"### 결론\n{conclusion.strip()}")

            if not sections:
                st.warning("최소 1개 섹션은 입력해주세요.")
                return

            user_rr = "\n\n".join(sections)

            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user",
                stance=reversed_user_stance,
                phase="role_reversal", content=user_rr,
                target_id=None, tool_calls_log=[], json_raw="",
            ))
            state["current_turn"] += 1
            state["phase"] = "synthesis"
            st.session_state.state = state

            add_msg("user", f"**[사용자 → {reversed_label} 옹호 (역할반전)]**\n\n{user_rr}")
            add_msg("assistant", "---\n## 5단계: 종합 및 재개념화\n역할반전이 완료되었습니다. 이제 최적해를 도출합니다.")
            st.session_state.phase = "synthesis"
            st.rerun()

    # ══════════════════════════════════════════════
    # 5단계: 종합 및 재개념화
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "synthesis":
        render_messages()

        state = st.session_state.state

        # AI 종합 발언 생성
        if not st.session_state.get("synthesis_done"):
            with st.spinner("🧠 모든 AI 에이전트가 최적해를 도출하고 있습니다..."):
                state = synthesis_node(state)
                state = dict(state)
                st.session_state.state = state
                st.session_state.synthesis_done = True

            syn_entries = [e for e in state["debate_history"] if e["phase"] == "synthesis" and e["speaker_id"] != "user"]
            stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])
            for entry in syn_entries:
                s_label = "찬성" if entry["stance"] == "PRO" else "반대"
                snum = stance_nums.get(entry["speaker_id"], 1)
                display = f"{s_label} 에이전트{snum}"
                add_msg("assistant", f"**[{display} — 최적해]**\n\n{entry['content']}")

            add_msg("assistant", "🧠 이제 사용자도 토론 전체를 종합한 **최적해**를 작성해주세요.\n\n"
                    "자기 입장 고수가 아닌, 양측 주장의 타당한 점을 인정하고 구체적 해결책을 제시하세요.")
            st.rerun()

        # 사용자 종합 입력 폼
        st.subheader("✍️ 종합 및 재개념화: 최적해 도출")
        st.info("지금까지의 토론을 종합하여, **논쟁의 핵심 충돌을 해결하는 구조적 최적해**를 도출하세요.\n\n"
                "일반적 정책 나열이 아닌, 이 주제 고유의 문제 구조를 분석하고 해결 메커니즘을 제시해주세요.")

        with st.form("synthesis_form"):
            pro_valid = st.text_area("찬성측 타당한 점", placeholder="찬성측 주장에서 인정할 만한 점 (1~2줄)", height=60)
            con_valid = st.text_area("반대측 타당한 점", placeholder="반대측 주장에서 인정할 만한 점 (1~2줄)", height=60)
            core_structure = st.text_area("문제의 핵심 구조", placeholder="찬성과 반대가 충돌하는 근본 원인은 무엇인가? (예: 속도 불균형, 비용 분배, 시간 지평 차이 등)", height=100)
            optimal = st.text_area("최적해", placeholder="문제 재정의 → 작동 메커니즘(누가/무엇을/어떤 조건에서) → 비용 부담 구조 → 왜 이것이 문제를 해결하는지", height=180)
            submitted = st.form_submit_button("최적해 제출 → 토론 종료", type="primary", use_container_width=True)

        if submitted:
            sections = []
            if pro_valid.strip():
                sections.append(f"### 찬성측 타당한 점\n{pro_valid.strip()}")
            if con_valid.strip():
                sections.append(f"### 반대측 타당한 점\n{con_valid.strip()}")
            if core_structure.strip():
                sections.append(f"### 문제의 핵심 구조\n{core_structure.strip()}")
            if optimal.strip():
                sections.append(f"### 최적해\n{optimal.strip()}")

            if not sections:
                st.warning("최소 1개 섹션은 입력해주세요.")
                return

            user_syn = "\n\n".join(sections)

            state["debate_history"].append(DebateEntry(
                turn=state["current_turn"], speaker_id="user",
                stance=USER_STANCE, phase="synthesis",
                content=user_syn, target_id=None,
                tool_calls_log=[], json_raw="",
            ))
            state["current_turn"] += 1
            state["is_finished"] = True
            st.session_state.state = state

            add_msg("user", f"**[사용자 — 최적해]**\n\n{user_syn}")
            add_msg("assistant", "---\n## 토론 완료\n\n모든 참여자의 최적해가 제출되었습니다. 사이드바에서 결과를 저장할 수 있습니다.")
            st.session_state.phase = "finished"
            st.rerun()

    # ══════════════════════════════════════════════
    # 완료
    # ══════════════════════════════════════════════
    elif st.session_state.phase == "finished":
        render_messages()
        st.success("🎉 5단계 토론이 모두 완료되었습니다! 사이드바에서 결과를 저장할 수 있습니다.")


if __name__ == "__main__":
    main()
