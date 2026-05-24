"""FastAPI 내 모든 기능 풀로 돌려 단일 JSON 으로 누적.

기능 흐름:
1. POST /debate/init → AI 입론 SSE (실시간 분석 포함)
2. 각 phase 시작 시 GET /debate/{id}/assistant/{phase}
3. 사용자 발언은 experiments._run_single.UserProxy 가 LLM 으로 자동 생성
4. POST /debate/{id}/submit → 다음 phase 까지 SSE
5. 토론 종료 후 GET /debate/{id}/final-report
6. UserProxy 로 사전·사후 PRO/CON 입론 별도 생성 → POST /evaluation
7. 모든 이벤트·응답을 하나의 JSON 으로 저장

사용:
    python tests/run_full_fastapi_flow.py --topic tech_003 --format 3:3 --stance PRO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


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


from experiments._run_single import UserProxy  # noqa: E402
from evaluation import analyze_user_before_after  # noqa: E402


_DEFAULT_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8001")
_INTENSITY_PRESETS = {
    # 실험 통제 — 모든 agent 강경도 균일 (3, 균형). 강경도 변수 제거.
    # 종합 단계의 다양성은 perspective_style + starter pool 로 보장 (stage5_synthesis/nodes.py).
    "1:1": [3],
    "2:2": [3, 3, 3],
    "3:3": [3, 3, 3, 3, 3],
}


# ── SSE 파서 ────────────────────────────────────────────────────────────────

def parse_sse_stream(response: httpx.Response) -> List[Dict[str, Any]]:
    """SSE 응답을 [{event, data}] 리스트로 파싱."""
    events: List[Dict[str, Any]] = []
    current_event = None
    current_data: List[str] = []
    for line in response.iter_lines():
        if not line:
            if current_event and current_data:
                payload = "\n".join(current_data)
                try:
                    parsed = json.loads(payload)
                except Exception:
                    parsed = payload
                events.append({"event": current_event, "data": parsed})
            current_event, current_data = None, []
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data.append(line[5:].strip())
    if current_event and current_data:
        payload = "\n".join(current_data)
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = payload
        events.append({"event": current_event, "data": parsed})
    return events


# ── 로깅 ────────────────────────────────────────────────────────────────────

class FlowLog:
    """전 phase 의 이벤트·응답을 누적."""

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta
        self.phases: List[Dict[str, Any]] = []
        self.assistant_guides: List[Dict[str, Any]] = []
        self.final_report: Optional[Dict[str, Any]] = None
        self.evaluation: Optional[Dict[str, Any]] = None
        self.pre_post_inputs: Dict[str, str] = {}
        # user 슬롯의 tool_calls (UserProxy 가 만든 검색 — graph transcript 에는 안 들어감)
        self.user_tool_calls: Dict[str, List[Dict[str, Any]]] = {}
        # 시간순 통합 timeline — 분석 시 어시스턴트 가이드와 turn 흐름을 함께 읽기 위함
        # 각 entry: {type: "assistant_guide" | "turn", ...}
        self.timeline: List[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}

    def log_phase(self, label: str, events: List[Dict[str, Any]]) -> None:
        self.phases.append({"label": label, "events": events})
        # timeline 에 phase 의 turn 들 시간순으로 추가 — primitive 값만 (cycle 회피)
        for ev in events:
            if ev["event"] != "turn":
                continue
            entry = ev["data"].get("entry", {}) or {}
            analysis = ev["data"].get("analysis") or {}
            # tool_calls 는 string summary 로만 (nested reference 회피)
            tc_summary = []
            for tc in (entry.get("tool_calls_log") or []):
                tc_summary.append({
                    "name": str(tc.get("name", "")),
                    "query": str((tc.get("args") or {}).get("query", "")),
                    "result_preview": str(tc.get("result", ""))[:300],
                })
            self.timeline.append({
                "type": "turn",
                "phase_label": str(label),
                "speaker_id": str(entry.get("speaker_id", "")),
                "stance": str(entry.get("stance", "")),
                "phase": str(entry.get("phase", "")),
                "target_id": str(entry.get("target_id") or ""),
                "content": str(entry.get("content", "")),
                "tool_calls": tc_summary,
                "argument_score": analysis.get("argument_score"),
                "evidence_score": analysis.get("evidence_score"),
                "language_score": analysis.get("language_score"),
                "weighted_score": analysis.get("weighted_score"),
                "feedback_argument": str((analysis.get("dimension_feedbacks") or {}).get("argument", "")),
                "feedback_evidence": str((analysis.get("dimension_feedbacks") or {}).get("evidence", "")),
                "feedback_language": str((analysis.get("dimension_feedbacks") or {}).get("language", "")),
                "feedback_overall": str(analysis.get("overall_feedback", "")),
                "pro_percent": analysis.get("pro_percent"),
                "con_percent": analysis.get("con_percent"),
            })

    def log_assistant(self, phase: str, text: str) -> None:
        self.assistant_guides.append({"phase": phase, "text": text})
        # 어시스턴트 가이드를 호출 시점에 timeline 에 삽입 (사용자가 가이드 후 응답하는 흐름 반영)
        self.timeline.append({
            "type": "assistant_guide",
            "phase": phase,
            "text": text,
        })

    def log_user_tc(self, phase_label: str, tc_log: List[Dict[str, Any]]) -> None:
        if tc_log:
            self.user_tool_calls[phase_label] = tc_log

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def _drop_cycles(obj, seen=None):
            """객체 안 circular reference 를 잘라 JSON 직렬화 가능하게 만든다."""
            if seen is None:
                seen = set()
            if isinstance(obj, dict):
                if id(obj) in seen:
                    return "<CYCLE_DICT>"
                seen = seen | {id(obj)}
                return {k: _drop_cycles(v, seen) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                if id(obj) in seen:
                    return "<CYCLE_LIST>"
                seen = seen | {id(obj)}
                return [_drop_cycles(x, seen) for x in obj]
            # primitive (str, int, float, bool, None) 또는 알 수 없는 타입
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            return str(obj)[:500]  # 마지막 fallback

        out = {
            "meta": self.meta,
            "timeline": self.timeline,
            "phases": self.phases,
            "assistant_guides": self.assistant_guides,
            "final_report": self.final_report,
            "evaluation": self.evaluation,
            "pre_post_inputs": self.pre_post_inputs,
            "user_tool_calls": self.user_tool_calls,
            "timings": self.timings,
        }
        safe_out = _drop_cycles(out)
        path.write_text(
            json.dumps(safe_out, ensure_ascii=False, indent=2), encoding="utf-8",
        )


# ── API 호출 헬퍼 ───────────────────────────────────────────────────────────

def init_session(
    client: httpx.Client, topic_id: str, user_stance: str, fmt: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    body = {
        "topic": topic_id,
        "user_stance": user_stance,
        "user_intensity": 3,
        "agent_intensities": _INTENSITY_PRESETS[fmt],
        "debate_format": fmt,
    }
    with client.stream("POST", "/debate/init", json=body, timeout=None) as r:
        r.raise_for_status()
        events = parse_sse_stream(r)
    session_id = ""
    for ev in events:
        if ev["event"] == "session":
            session_id = ev["data"].get("session_id", "")
            break
    return session_id, events


def submit_user(
    client: httpx.Client, session_id: str, content: str, target_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    body: Dict[str, Any] = {"content": content}
    if target_id:
        body["target_id"] = target_id
    with client.stream(
        "POST", f"/debate/{session_id}/submit", json=body, timeout=None,
    ) as r:
        r.raise_for_status()
        events = parse_sse_stream(r)
    return events


def get_assistant(client: httpx.Client, session_id: str, phase: str, opponent_id: Optional[str] = None) -> str:
    params = {}
    if opponent_id:
        params["opponent_id"] = opponent_id
    r = client.get(f"/debate/{session_id}/assistant/{phase}", params=params, timeout=180)
    r.raise_for_status()
    return r.json().get("text", "")


def get_state(client: httpx.Client, session_id: str) -> Dict[str, Any]:
    r = client.get(f"/debate/{session_id}/state", timeout=30)
    r.raise_for_status()
    return r.json()


def get_final_report(client: httpx.Client, session_id: str) -> Dict[str, Any]:
    r = client.get(f"/debate/{session_id}/final-report", timeout=300)
    r.raise_for_status()
    return r.json()


# ── waiting 분석 ────────────────────────────────────────────────────────────

def waiting_info(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """events 중 마지막 waiting 이벤트의 data 반환."""
    for ev in reversed(events):
        if ev["event"] == "waiting":
            return ev["data"]
    return {}


def find_last_attacker(events: List[Dict[str, Any]], stance_opposite_user: str) -> Optional[str]:
    """events 의 turn 중 사용자를 향한 마지막 공격자 (반대 진영 AI) 찾기."""
    for ev in reversed(events):
        if ev["event"] != "turn":
            continue
        entry = ev["data"].get("entry", {})
        if entry.get("target_id") == "user" and entry.get("stance") == stance_opposite_user:
            return entry.get("speaker_id")
    # fallback: 가장 최근 반대 진영 발언
    for ev in reversed(events):
        if ev["event"] != "turn":
            continue
        entry = ev["data"].get("entry", {})
        if entry.get("speaker_id") != "user" and entry.get("stance") == stance_opposite_user:
            return entry.get("speaker_id")
    return None


def _aggregate_evaluation_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """n_trials 결과를 평균·표준편차로 집계. 단일 trial 이면 그 결과만 반환."""
    import statistics
    valid = [t for t in trials if "result" in t]
    if not valid:
        return {"n_trials": len(trials), "trials": trials, "error": "no valid trials"}
    if len(valid) == 1:
        # 단일 trial — 기존 형식 그대로 반환 + meta 추가
        r = valid[0]["result"]
        r["n_trials"] = 1
        r["trials"] = valid
        return r

    # 다회 — 평균·표준편차
    metrics_meta = valid[0]["result"].get("metrics") or []
    metric_keys = [m["key"] for m in metrics_meta if isinstance(m, dict) and "key" in m]
    if not metric_keys:
        metric_keys = [
            "evidence_expansion", "knowledge_specificity", "evidence_validity",
            "reasoning_density", "perspective_diversity",
        ]
    summary: Dict[str, Any] = {"n_trials": len(valid), "trials": valid, "topic": valid[0]["result"].get("topic", "")}
    for stance in ["pro", "con"]:
        summary[stance] = {}
        for phase in ["pre", "post"]:
            avg5_list = [t["result"][stance][phase]["average_5"] for t in valid]
            avg100_list = [t["result"][stance][phase]["average_100"] for t in valid]
            metric_avg = {}
            for mk in metric_keys:
                scores = [
                    t["result"][stance][phase]["scores"].get(mk, {}).get("score", 0)
                    for t in valid
                ]
                metric_avg[mk] = {
                    "mean": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "values": scores,
                }
            summary[stance][phase] = {
                "average_5_mean": statistics.mean(avg5_list),
                "average_5_std": statistics.stdev(avg5_list) if len(avg5_list) > 1 else 0.0,
                "average_100_mean": statistics.mean(avg100_list),
                "average_100_std": statistics.stdev(avg100_list) if len(avg100_list) > 1 else 0.0,
                "average_5_values": avg5_list,
                "metric_scores": metric_avg,
            }
        delta_list = [t["result"][stance]["delta_100"] for t in valid]
        summary[stance]["delta_100_mean"] = statistics.mean(delta_list)
        summary[stance]["delta_100_std"] = statistics.stdev(delta_list) if len(delta_list) > 1 else 0.0
        summary[stance]["delta_100_values"] = delta_list
    return summary


# ── 메인 흐름 ──────────────────────────────────────────────────────────────

def run_full_flow(
    topic_id: str, fmt: str, user_stance: str, base_url: str, output: Path,
    n_trials: int = 1,
) -> int:
    # topic dict 로 UserProxy 만들려면 topics JSON 에서 직접 가져온다
    topics_path = _ROOT / "data" / "topics_20260323_processed.json"
    with topics_path.open(encoding="utf-8") as f:
        topics = json.load(f)
    topic_dict = None
    for cat in topics.get("categories", {}).values():
        for t in cat:
            if t["id"] == topic_id:
                topic_dict = t
                break
        if topic_dict:
            break
    if not topic_dict:
        print(f"[ERROR] topic_id 못 찾음: {topic_id}")
        return 1

    # user 슬롯 focus_area 는 같은 진영 AI 가 차지한 인덱스 다음 자리 (남는 자리)
    # 3v3 PRO: AI 가 PRO 2명 → PRO[0],[1] 차지 → user 가 PRO[2]
    # 2v2 PRO: AI 가 PRO 1명 → PRO[0] 차지 → user 가 PRO[1]
    _stance_list = {
        ("1:1", "PRO"): ["CON"],
        ("1:1", "CON"): ["PRO"],
        ("2:2", "PRO"): ["CON", "CON", "PRO"],
        ("2:2", "CON"): ["PRO", "PRO", "CON"],
        ("3:3", "PRO"): ["CON", "CON", "CON", "PRO", "PRO"],
        ("3:3", "CON"): ["PRO", "PRO", "PRO", "CON", "CON"],
    }.get((fmt, user_stance), [])
    same_stance_ai_count = sum(1 for st in _stance_list if st == user_stance)
    proxy = UserProxy(
        topic=topic_dict, stance=user_stance, intensity=3,
        topic_id=topic_id, slot_index=same_stance_ai_count,
    )

    flow = FlowLog(meta={
        "topic_id": topic_id,
        "topic_title": topic_dict["title"],
        "user_stance": user_stance,
        "debate_format": fmt,
        "agent_intensities": _INTENSITY_PRESETS[fmt],
        "user_focus_area": proxy.focus_area,
        "started_at": datetime.now().isoformat(),
        "base_url": base_url,
    })

    opposite = "CON" if user_stance == "PRO" else "PRO"

    client = httpx.Client(base_url=base_url, timeout=None)

    t_total = time.time()

    # ─── 1) /debate/init  + 입론 ─────────────────────────────────────────
    print("[1] /debate/init — AI 입론 + 분석")
    t0 = time.time()
    session_id, init_events = init_session(client, topic_id, user_stance, fmt)
    flow.log_phase("opening_ai_init", init_events)
    flow.timings["opening_ai_init"] = time.time() - t0
    print(f"    session_id={session_id}")

    # ─── 2) 어시스턴트 opening 가이드 ────────────────────────────────────
    print("[2] /assistant/opening")
    t0 = time.time()
    guide_open = get_assistant(client, session_id, "opening")
    flow.log_assistant("opening", guide_open)
    flow.timings["assistant_opening"] = time.time() - t0

    # ─── 3) 사용자 입론 생성 → submit ────────────────────────────────────
    print("[3] UserProxy.opening + submit")
    t0 = time.time()
    user_opening_text, user_open_tc = proxy.opening(display=f"{'찬성' if user_stance == 'PRO' else '반대'}{_INTENSITY_PRESETS[fmt].count(user_stance) + 1}")
    flow.pre_post_inputs["user_opening"] = user_opening_text
    flow.log_user_tc("opening", user_open_tc)
    open_submit_events = submit_user(client, session_id, user_opening_text)
    flow.log_phase("opening_user_submit", open_submit_events)
    flow.timings["opening_user_submit"] = time.time() - t0

    # 다음은 연쇄논박 — 위 submit_events 에 AI 들 연쇄논박 + 사용자 차례 대기
    wait = waiting_info(open_submit_events)
    print(f"    next wait: {wait.get('waiting_for')}")

    # ─── 4) 어시스턴트 chained_rebuttal 가이드 + 사용자 발언 ──────────────
    print("[4] /assistant/chained_rebuttal + submit")
    t0 = time.time()
    # 사용자에게 공격해온 agent 찾기
    attacker = find_last_attacker(open_submit_events, opposite)
    guide_chain = get_assistant(client, session_id, "chained_rebuttal", opponent_id=attacker)
    flow.log_assistant("chained_rebuttal", guide_chain)
    # 사용자 받아치기 생성 — 그 attacker 의 마지막 발언이 target
    target_speech = ""
    target_display = ""
    for ev in reversed(open_submit_events):
        if ev["event"] != "turn":
            continue
        entry = ev["data"].get("entry", {})
        if entry.get("speaker_id") == attacker:
            target_speech = entry.get("content", "")
            sp = entry.get("speaker_id", "")
            target_display = "반대1" if opposite == "CON" else "찬성1"  # placeholder
            break
    # history 는 비워둬도 chain build 됨
    user_chain_text, user_chain_tc = proxy.chained_rebuttal(
        target_speech, target_display, history=[],
    )
    flow.pre_post_inputs["user_chained_rebuttal"] = user_chain_text
    flow.log_user_tc("chained_rebuttal", user_chain_tc)
    chain_submit_events = submit_user(client, session_id, user_chain_text, target_id=attacker)
    flow.log_phase("chained_rebuttal_user_submit", chain_submit_events)
    flow.timings["chained_rebuttal_user_submit"] = time.time() - t0

    wait = waiting_info(chain_submit_events)
    print(f"    next wait: {wait.get('waiting_for')}")

    # ─── 5) 자유논박 — 상대 선택 + 2 round (defense + attack) ─────────────
    # 첫 submit 은 opponent 선택. main.py 의 free_rebuttal 흐름은 selected_opponent_id 필요.
    print("[5] 자유논박 — opponent 선택")
    # state 에서 자유논박 첫 사용자 인터럽트 확인
    state = get_state(client, session_id)
    print(f"    state phase: {state.get('phase')}")
    # 자유논박 사용자 차례에 opponent 선택. main.py 가 어떻게 받나 — submit 의 target_id 로 보냄.
    # 첫 자유논박 사용자 발언은 defense (AI 가 먼저 공격), 두 번째는 attack
    # 그러나 main.py 의 free_rebuttal 은 user_select_opponent_node 가 있음

    # 실제 흐름: chain_submit 후 → 자유논박 진입 → user_select_opponent waiting
    # opponent 선택: submit 으로 target_id 만 전송 (content 는 placeholder)
    # 그 후 AI 공격 → user_free_rebuttal waiting → 사용자 defense+attack submit

    free_rounds_logged = []
    opp_id = attacker  # 같은 상대로 자유논박
    if wait.get("waiting_for") == "user_select_opponent":
        sel_events = submit_user(client, session_id, "(opponent 선택)", target_id=opp_id)
        flow.log_phase("free_rebuttal_select_opponent", sel_events)
        wait = waiting_info(sel_events)
        print(f"    after select: {wait.get('waiting_for')}")
        free_rounds_logged.append(sel_events)

    # 자유논박 round 1, 2 — 각 round 마다 AI 공격 → 사용자 defense+attack
    opp_opening = ""
    my_opening = user_opening_text
    history_simple: List = []
    for round_idx in range(2):
        print(f"    [free_round {round_idx+1}]")
        # AI 가 공격하는 발언 찾기
        last_attack_speech = ""
        last_events = free_rounds_logged[-1] if free_rounds_logged else chain_submit_events
        for ev in reversed(last_events):
            if ev["event"] != "turn":
                continue
            entry = ev["data"].get("entry", {})
            if entry.get("phase") == "free_rebuttal" and entry.get("speaker_id") != "user":
                last_attack_speech = entry.get("content", "")
                if not opp_opening:
                    opp_opening = last_attack_speech
                break

        t0 = time.time()
        guide_free = get_assistant(client, session_id, "free_rebuttal", opponent_id=opp_id)
        flow.log_assistant(f"free_rebuttal_round{round_idx+1}", guide_free)

        # UserProxy.free_defense_attack → defense + attack 한 쌍
        defense_text, attack_text, free_tc = proxy.free_defense_attack(
            opp_attack=last_attack_speech,
            opp_opening=opp_opening,
            my_opening=my_opening,
            history=history_simple,
        )
        flow.pre_post_inputs[f"user_free_defense_{round_idx+1}"] = defense_text
        flow.pre_post_inputs[f"user_free_attack_{round_idx+1}"] = attack_text
        flow.log_user_tc(f"free_rebuttal_round{round_idx+1}", free_tc)

        def_events = submit_user(client, session_id, defense_text, target_id=opp_id)
        flow.log_phase(f"free_rebuttal_round{round_idx+1}_defense", def_events)
        wait = waiting_info(def_events)
        print(f"      after defense: {wait.get('waiting_for')}")
        free_rounds_logged.append(def_events)
        if wait.get("is_finished"):
            break

        atk_events = submit_user(client, session_id, attack_text, target_id=opp_id)
        flow.log_phase(f"free_rebuttal_round{round_idx+1}_attack", atk_events)
        wait = waiting_info(atk_events)
        print(f"      after attack: {wait.get('waiting_for')}")
        free_rounds_logged.append(atk_events)
        flow.timings[f"free_rebuttal_round{round_idx+1}"] = time.time() - t0
        if wait.get("is_finished") or wait.get("waiting_for") in ("user_role_reversal", "user_synthesis", "user_finalize"):
            break

    # ─── 6) 역할반전 (graph 분기에 따라 건너뛸 수 있음) ──────────────────
    if wait.get("waiting_for") == "user_role_reversal":
        print("[6] 역할반전")
        t0 = time.time()
        guide_rr = get_assistant(client, session_id, "role_reversal")
        flow.log_assistant("role_reversal", guide_rr)
        reversed_stance = "PRO" if user_stance == "CON" else "CON"
        opp_openings_text = ""
        for ev in init_events:
            if ev["event"] != "turn":
                continue
            entry = ev["data"].get("entry", {})
            if entry.get("phase") == "opening" and entry.get("stance") == user_stance:
                opp_openings_text += f"\n{entry.get('content','')}"
        rr_text, rr_tc = proxy.role_reversal(reversed_stance, opp_openings_text)
        flow.pre_post_inputs["user_role_reversal"] = rr_text
        flow.log_user_tc("role_reversal", rr_tc)
        rr_events = submit_user(client, session_id, rr_text)
        flow.log_phase("role_reversal_user_submit", rr_events)
        flow.timings["role_reversal"] = time.time() - t0
        wait = waiting_info(rr_events)
        print(f"    next wait: {wait.get('waiting_for')}")

    # ─── 7) 종합 — proposal + final + finalize (round 별 반복) ────────────
    # user_synthesis 인터럽트가 여러 라운드에 걸쳐 등장. synthesis_user_turns 로 round 구분
    synthesis_round = 0
    safety = 0
    while wait.get("waiting_for") == "user_synthesis" and not wait.get("is_finished") and safety < 5:
        synthesis_round += 1
        safety += 1
        print(f"[7.{synthesis_round}] 종합 - round {synthesis_round}")
        t0 = time.time()
        guide_syn = get_assistant(client, session_id, "synthesis")
        flow.log_assistant(f"synthesis_round{synthesis_round}", guide_syn)
        # round 1: proposal, round 2+: final (UserProxy 의 두 method 가 다른 prompt 씀)
        if synthesis_round == 1:
            syn_text, syn_tc = proxy.synthesis_proposal(history=history_simple)
        else:
            syn_text, syn_tc = proxy.synthesis_final(history=history_simple)
        flow.pre_post_inputs[f"user_synthesis_round{synthesis_round}"] = syn_text
        flow.log_user_tc(f"synthesis_round{synthesis_round}", syn_tc)
        syn_events = submit_user(client, session_id, syn_text)
        flow.log_phase(f"synthesis_round{synthesis_round}_user_submit", syn_events)
        flow.timings[f"synthesis_round{synthesis_round}"] = time.time() - t0
        wait = waiting_info(syn_events)
        print(f"    next wait: {wait.get('waiting_for')} (finished={wait.get('is_finished')})")

    # ─── 7b) finalize (최적해 확정 단계) ────────────────────────────────
    if wait.get("waiting_for") == "user_finalize":
        print("[7b] finalize - 최적해 확정")
        t0 = time.time()
        fin_text, _ = proxy.synthesis_final(history=history_simple)
        flow.pre_post_inputs["user_finalize"] = fin_text
        fin_events = submit_user(client, session_id, fin_text)
        flow.log_phase("finalize_user_submit", fin_events)
        flow.timings["finalize"] = time.time() - t0
        wait = waiting_info(fin_events)
        print(f"    finalized (finished={wait.get('is_finished')})")

    # ─── 8) final-report ─────────────────────────────────────────────
    print("[8] /final-report")
    t0 = time.time()
    try:
        flow.final_report = get_final_report(client, session_id)
    except Exception as e:
        flow.final_report = {"error": str(e)}
    flow.timings["final_report"] = time.time() - t0

    # ─── 9) 사전·사후 평가 (n_trials 회 반복 후 평균) ──────────────────
    # 각 trial: pre/post PRO/CON 답변 생성 (UserProxy) + 5 metric 평가
    # n_trials > 1 일 때 답변·평가 모두 새로 호출하여 분산·평균 측정
    print(f"[9] 사전·사후 평가 (n_trials={n_trials})")
    t0 = time.time()
    trials: List[Dict[str, Any]] = []
    for trial_idx in range(n_trials):
        print(f"  [trial {trial_idx+1}/{n_trials}] 답변 생성 + 평가")
        proxy_pro_pre = UserProxy(topic=topic_dict, stance="PRO", intensity=3, topic_id=topic_id)
        proxy_con_pre = UserProxy(topic=topic_dict, stance="CON", intensity=3, topic_id=topic_id)
        pre_pro_text, _ = proxy_pro_pre.opening(display="찬성")
        pre_con_text, _ = proxy_con_pre.opening(display="반대")
        proxy_pro_post = UserProxy(topic=topic_dict, stance="PRO", intensity=3, topic_id=topic_id)
        proxy_con_post = UserProxy(topic=topic_dict, stance="CON", intensity=3, topic_id=topic_id)
        post_pro_text, _ = proxy_pro_post.opening(display="찬성")
        post_con_text, _ = proxy_con_post.opening(display="반대")
        try:
            eval_result = analyze_user_before_after(
                pre_pro=pre_pro_text, pre_con=pre_con_text,
                post_pro=post_pro_text, post_con=post_con_text,
                topic=topic_dict["title"],
            )
            trials.append({
                "trial": trial_idx + 1,
                "result": eval_result,
                "inputs": {
                    "pre_pro": pre_pro_text, "pre_con": pre_con_text,
                    "post_pro": post_pro_text, "post_con": post_con_text,
                },
            })
        except Exception as e:
            trials.append({"trial": trial_idx + 1, "error": str(e)})
    # 평균 계산
    flow.evaluation = _aggregate_evaluation_trials(trials)
    # 마지막 trial 의 답변 기록 (대표값)
    if trials and "inputs" in trials[-1]:
        flow.pre_post_inputs.update(trials[-1]["inputs"])
    flow.timings["evaluation"] = time.time() - t0

    flow.meta["finished_at"] = datetime.now().isoformat()
    flow.meta["total_duration_seconds"] = round(time.time() - t_total, 1)
    flow.meta["session_id"] = session_id

    flow.dump(output)
    print(f"\n[OK] 저장: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="tech_003")
    parser.add_argument("--format", default="3:3", choices=list(_INTENSITY_PRESETS.keys()))
    parser.add_argument("--stance", default="PRO", choices=["PRO", "CON"])
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help="evaluation 다회 시도 (통계 분석용, n=3 권장). 각 trial 마다 답변·평가 새로 생성",
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output", default=str(_ROOT / "test_results" / "fastapi_full_flow" / f"flow_tech_003_{ts}.json"),
    )
    args = parser.parse_args()

    return run_full_flow(
        topic_id=args.topic, fmt=args.format, user_stance=args.stance,
        base_url=args.base_url, output=Path(args.output), n_trials=args.n_trials,
    )


if __name__ == "__main__":
    sys.exit(main())
