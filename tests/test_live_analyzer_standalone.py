"""live_analyzer 단독 테스트.

실제 토론 로그 JSON을 읽어 judge_turn()을 순차 호출하고
각 턴 점수 + 최종 live_debate 값을 출력한다.

사용:
    JUDGE_VLLM_BASE_URL=http://localhost:8000/v1 python tests/test_live_analyzer_standalone.py \
        --log data/logs/Qwen-2.5-32B-Instruct/tech_001_2v2_balanced.json \
        --max-turns 0  # 0=전부
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 프로젝트 root를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live_analyzer import DebatrixJudge, project_frontend_event


def load_log(path: Path) -> dict:
    candidates = [path]
    # 워크트리에 data가 없으면 메인 리포에서 같은 상대경로 시도
    MAIN = Path("/disk1/SJ/SJU-Capstone-Multi-Agent-Debater-AI")
    if "data/logs" in str(path):
        rel = str(path).split("data/logs", 1)[1].lstrip("/")
        candidates.append(MAIN / "data" / "logs" / rel)
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"not found: tried {candidates}")


def build_teams(turns: list) -> dict:
    pro, con = set(), set()
    for t in turns:
        sp = t.get("speaker")
        side = t.get("side")
        if side == "PRO":
            pro.add(sp)
        elif side == "CON":
            con.add(sp)
    return {"PRO": sorted(pro), "CON": sorted(con)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="debate log JSON 경로")
    ap.add_argument("--max-turns", type=int, default=3, help="0이면 전체")
    ap.add_argument("--out", default="/tmp/live_analyzer_test.json")
    args = ap.parse_args()

    log_path = Path(args.log)
    data = load_log(log_path)

    topic = data["topic"]
    debate_format = data.get("debate_format", "2:2")
    user_stance = data.get("user_stance", "PRO")
    turns = data["turns"]
    teams = build_teams(turns)

    print(f"[LOG] {log_path.name}")
    print(f"[TOPIC] {topic}")
    print(f"[FORMAT] {debate_format} / user={user_stance}")
    print(f"[TEAMS] {teams}")
    print(f"[TURNS] {len(turns)}개")
    print()

    judge = DebatrixJudge(
        topic=topic,
        debate_format=debate_format,
        user_id="user",
        user_stance=user_stance,
        teams=teams,
    )

    if args.max_turns > 0:
        turns = turns[: args.max_turns]

    t0 = time.time()
    frontend_events: list = []  # 턴마다 프론트로 나갈 경량 이벤트 (5숫자)
    for i, t in enumerate(turns):
        target = t.get("target_id")
        if target in (None, "None", ""):
            target = None
        speaker = t.get("speaker") or "unknown"
        stance = t.get("side") or "PRO"
        phase = t.get("phase") or "opening"
        text = t.get("text") or ""

        print(f"━━ T{i} [{speaker}/{stance}/{phase}] → target={target}")
        ts = time.time()
        analysis = judge.judge_turn(
            speaker_id=speaker,
            speaker_stance=stance,
            phase=phase,
            speech_content=text,
            target_id=target,
        )
        dt = time.time() - ts
        if analysis is None:
            print(f"   (skip: phase={phase} — 실시간 분석 대상 외)\n")
            continue
        live = judge.memory.live_debate_snapshot()
        # 프론트 이벤트 = 매 턴 5숫자 projection
        analysis_row = judge.memory.analysis_memory[-1]  # 방금 추가된 analysis
        event = project_frontend_event(analysis_row, live)
        frontend_events.append(event)
        print(
            f"   arg={analysis.argument.score:.1f} "
            f"evi={analysis.evidence.score:.1f} "
            f"lang={analysis.language.score:.1f} "
            f"→ weighted={analysis.weighted_score:.2f} "
            f"| PRO {live['pro_percent']:.1f} : {live['con_percent']:.1f} CON "
            f"({dt:.1f}s)"
        )
        print(f"   요약: {analysis.overall_summary[:120]}")
        print()

    total = time.time() - t0
    print(f"━━ 전체 {len(turns)}턴 완료 / {total:.1f}s (턴당 {total/len(turns):.1f}s)")

    # 저장: 실시간 흐름(프론트 이벤트) + 메모리 누적(요약·차원별 피드백·분석)
    out = {
        "topic": topic,
        "debate_format": debate_format,
        "user_stance": user_stance,
        "frontend_events": frontend_events,          # 턴별 5숫자 (실시간 흐름)
        "speech_summaries": judge.memory.speech_memory,  # 누적 발언 요약
        "analyses": judge.memory.analysis_memory,    # 누적 전체 분석 (차원별 점수·피드백)
        "final_live_debate": judge.memory.live_debate_snapshot(),
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
