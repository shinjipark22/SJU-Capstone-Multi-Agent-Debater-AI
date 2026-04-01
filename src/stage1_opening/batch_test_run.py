"""
batch_test_run.py — 입론(Opening Arguments) 전수 테스트 및 결과 JSON 저장

TOPIC_ID × DEBATE_FORMAT × AGENT_INTENSITIES 모든 조합을 순회하며
opening_arguments_node를 실행하고, 결과를 JSON 파일로 저장한다.
루브릭 평가용 데이터 생성 스크립트.

vLLM 서버 2개(포트 8000, 8001)를 사용하여 테스트 케이스를 절반씩 병렬 실행한다.

실행:
    python -m src.stage1_opening.batch_test_run
    (또는 PYTHONPATH=. python src/stage1_opening/batch_test_run.py)
"""

import json
import sys
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── 경로 설정 ────────────────────────────────────────────────────────────────

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "results" / "stage1_opening"


# ── 테스트 매트릭스 ──────────────────────────────────────────────────────────

ALL_TOPIC_IDS = [
    "tech_001", "tech_002", "tech_003",
    "econ_001", "econ_002", "econ_003",
    "poli_001", "poli_002", "poli_003",
    "env_001",  "env_002",  "env_003",
]

DEBATE_FORMATS = ["1:1", "2:2", "3:3"]

# 포맷별 대표 강경도 프로파일
# 1:1 → AI 1명, 2:2 → AI 3명, 3:3 → AI 5명
INTENSITY_PROFILES: dict[str, list[list[int]]] = {
    "1:1": [
        [1], [2], [3], [4], [5],
    ],
    "2:2": [
        [1, 1, 1],
        [3, 3, 3],
        [5, 5, 5],
        [1, 3, 5],
        [5, 3, 1],
    ],
    "3:3": [
        [1, 1, 1, 1, 1],
        [3, 3, 3, 3, 3],
        [5, 5, 5, 5, 5],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ],
}

VLLM_PORTS = [8000, 8001, 8002, 8003]

USER_STANCE = "PRO"
USER_INTENSITY = 3


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def _load_all_topics() -> dict[str, dict]:
    """토픽 ID → 토픽 dict 매핑을 반환한다."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {_DATA_PATH}")

    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    topic_map: dict[str, dict] = {}
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            topic_map[t["id"]] = t
    return topic_map


def _intensities_tag(intensities: list[int]) -> str:
    """강경도 리스트를 파일명용 문자열로 변환한다. e.g. [1,3,5] → '1-3-5'"""
    return "-".join(str(i) for i in intensities)


def _build_test_case_id(topic_id: str, fmt: str, intensities: list[int]) -> str:
    """고유 테스트 케이스 ID를 생성한다."""
    return f"{topic_id}__{fmt.replace(':', 'v')}__{_intensities_tag(intensities)}"


# ── 워커 프로세스에서 실행되는 함수 ──────────────────────────────────────────

def _run_worker(task: dict) -> dict:
    """
    별도 프로세스에서 실행된다.
    VLLM_BASE_URL 환경변수를 설정한 뒤 nodes 모듈을 임포트하여
    해당 포트의 vLLM 서버를 사용한다.
    """
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{task['port']}/v1"

    # 환경변수 설정 후 임포트해야 올바른 base_url이 적용된다
    from src.phase0.persona_factory import create_agents
    from src.state import AgentSnapshot, build_initial_state
    from src.stage1_opening.nodes import opening_arguments_node

    topic_dict = task["topic_dict"]
    debate_format = task["debate_format"]
    intensities = task["intensities"]
    case_id = task["case_id"]

    personas = create_agents(
        topic=topic_dict,
        debate_format=debate_format,
        user_stance=USER_STANCE,
        agent_intensities=intensities,
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
    )

    result_state = opening_arguments_node(state)

    agents_meta = [
        {
            "agent_id": p.agent_id,
            "stance": p.stance,
            "intensity": p.intensity,
            "role_description": p.role_description,
            "focus_area": p.focus_area,
        }
        for p in personas
    ]

    history_entries = []
    for entry in result_state["debate_history"]:
        history_entries.append({
            "turn": entry["turn"],
            "speaker_id": entry["speaker_id"],
            "stance": entry["stance"],
            "phase": entry["phase"],
            "content": entry["content"],
            "tool_calls_log": entry.get("tool_calls_log", []),
            "json_raw": entry.get("json_raw"),
        })

    return {
        "case_id": case_id,
        "topic_id": topic_dict["id"],
        "topic_title": topic_dict["title"],
        "debate_format": debate_format,
        "agent_intensities": intensities,
        "user_stance": USER_STANCE,
        "user_intensity": USER_INTENSITY,
        "agents": agents_meta,
        "speaking_order": result_state["speaking_order"],
        "opening_arguments": history_entries,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    topic_map = _load_all_topics()

    # 전체 테스트 케이스 목록 생성
    test_cases: list[tuple[str, str, list[int]]] = []
    for topic_id in ALL_TOPIC_IDS:
        for fmt in DEBATE_FORMATS:
            for intensities in INTENSITY_PROFILES[fmt]:
                test_cases.append((topic_id, fmt, intensities))

    total = len(test_cases)
    num_workers = len(VLLM_PORTS)

    print(f"{'═' * 70}")
    print(f" 입론(Opening Arguments) 배치 테스트")
    print(f" 총 {total}개 조합 | 토픽 {len(ALL_TOPIC_IDS)}개 × 포맷 {len(DEBATE_FORMATS)}개 × 강경도 프로파일")
    print(f" vLLM 서버 {num_workers}개 병렬 실행 (포트: {VLLM_PORTS})")
    print(f" 저장 경로: {run_dir}")
    print(f"{'═' * 70}\n")

    # 워커 태스크 생성 — 라운드로빈으로 포트 배분
    worker_tasks = []
    for idx, (topic_id, fmt, intensities) in enumerate(test_cases):
        topic_dict = topic_map.get(topic_id)
        if topic_dict is None:
            continue
        case_id = _build_test_case_id(topic_id, fmt, intensities)
        port = VLLM_PORTS[idx % num_workers]
        worker_tasks.append({
            "case_id": case_id,
            "topic_dict": topic_dict,
            "debate_format": fmt,
            "intensities": intensities,
            "port": port,
            "index": idx,
        })

    summary: list[dict] = []
    all_results: list[dict] = []
    success_count = 0
    fail_count = 0
    done_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(_run_worker, task): task
            for task in worker_tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            case_id = task["case_id"]
            done_count += 1

            try:
                result = future.result()
                result["timestamp"] = timestamp

                out_path = run_dir / f"{case_id}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                num_args = len(result["opening_arguments"])
                print(f"[{done_count}/{total}] {case_id} → OK (입론 {num_args}개) [port {task['port']}]")

                all_results.append(result)
                summary.append({
                    "case_id": case_id,
                    "topic_id": task["topic_dict"]["id"],
                    "debate_format": task["debate_format"],
                    "agent_intensities": task["intensities"],
                    "status": "success",
                    "num_arguments": num_args,
                    "file": str(out_path.name),
                })
                success_count += 1

            except Exception as e:
                print(f"[{done_count}/{total}] {case_id} → FAIL ({e}) [port {task['port']}]")
                summary.append({
                    "case_id": case_id,
                    "topic_id": task["topic_dict"]["id"],
                    "debate_format": task["debate_format"],
                    "agent_intensities": task["intensities"],
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                fail_count += 1

    # ── 요약 저장 ─────────────────────────────────────────────────────────────
    summary_data = {
        "run_timestamp": timestamp,
        "total_cases": total,
        "success": success_count,
        "failed": fail_count,
        "config": {
            "topic_ids": ALL_TOPIC_IDS,
            "debate_formats": DEBATE_FORMATS,
            "intensity_profiles": INTENSITY_PROFILES,
            "user_stance": USER_STANCE,
            "user_intensity": USER_INTENSITY,
            "vllm_ports": VLLM_PORTS,
        },
        "results": summary,
    }

    summary_path = run_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # ── 전체 결과 통합 파일 저장 ──────────────────────────────────────────────
    all_results_path = run_dir / "all_results.json"
    with all_results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'═' * 70}")
    print(f" 배치 테스트 완료")
    print(f" 성공: {success_count} / {total}  |  실패: {fail_count} / {total}")
    print(f" 결과 디렉토리: {run_dir}")
    print(f" 통합 결과    : {all_results_path}")
    print(f" 요약 파일    : {summary_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
