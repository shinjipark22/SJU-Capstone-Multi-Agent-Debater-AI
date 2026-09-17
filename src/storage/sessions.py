"""sessions.py — 토론 세션 수집 저장소.

`data-1787155428828.csv` 와 동일한 23개 컬럼(순서 포함)으로 세션을 누적한다.
저장소는 SQLite 파일 하나(`data/sessions.db`, DEBATE_DB_PATH 로 변경 가능)이고,
CSV 는 export 시점에 그 테이블에서 그대로 뽑아 쓴다.

    /debate/init   → create_session()   : 세션 메타 (닉네임·주제·진영·강도·1:1/2:2) 기록
    /evaluation    → save_evaluation()  : 토론 전·후 답변 + 채점 결과(평균/델타/요약) 기록
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# CSV 내보내기 컬럼 — data-1787155428828.csv 헤더와 순서까지 동일해야 한다.
CSV_COLUMNS = [
    "session_id", "debate_date", "nickname", "email", "topic",
    "user_stance", "user_intensity", "debate_format", "status",
    "pre_pro", "pre_con", "post_pro", "post_con",
    "pro_pre_average", "pro_post_average", "pro_delta",
    "con_pre_average", "con_post_average", "con_delta",
    "pro_pre_summary", "pro_post_summary", "con_pre_summary", "con_post_summary",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS debate_sessions (
    session_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_date      TEXT    NOT NULL,
    nickname         TEXT,
    email            TEXT,
    topic            TEXT    NOT NULL,
    user_stance      TEXT    NOT NULL,
    user_intensity   INTEGER,
    debate_format    TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'ACTIVE',
    pre_pro          TEXT,
    pre_con          TEXT,
    post_pro         TEXT,
    post_con         TEXT,
    pro_pre_average  REAL,
    pro_post_average REAL,
    pro_delta        REAL,
    con_pre_average  REAL,
    con_post_average REAL,
    con_delta        REAL,
    pro_pre_summary  TEXT,
    pro_post_summary TEXT,
    con_pre_summary  TEXT,
    con_post_summary TEXT,
    graph_session_id TEXT,
    mode             TEXT,
    synthesis_draft  TEXT,
    updated_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_graph_id ON debate_sessions(graph_session_id);

-- 턴별 실시간 분석. judge 인스턴스는 프로세스 메모리에만 있어 서버가 재시작되면
-- 최종 리포트를 만들 수 없으므로, 발생 즉시 여기에 남겨 복구 가능하게 한다.
CREATE TABLE IF NOT EXISTS turn_analyses (
    graph_session_id TEXT    NOT NULL,
    turn_index       INTEGER NOT NULL,
    analysis         TEXT    NOT NULL,
    speech           TEXT,
    live             TEXT,
    created_at       TEXT    NOT NULL,
    PRIMARY KEY (graph_session_id, turn_index)
);

-- 토론 전·후 연구 설문 (구글폼 대체). 한 세션당 pre/post 각 1행.
CREATE TABLE IF NOT EXISTS survey_responses (
    session_id   INTEGER NOT NULL,
    phase        TEXT    NOT NULL,
    mode         TEXT,
    answers      TEXT    NOT NULL,
    submitted_at TEXT    NOT NULL,
    PRIMARY KEY (session_id, phase)
);
"""

_write_lock = threading.Lock()


def _db_path() -> Path:
    override = os.environ.get("DEBATE_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "sessions.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS 로는 안 붙는 뒤늦게 추가된 컬럼을 채워 넣는다."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(debate_sessions)")}
    if "mode" not in existing:
        conn.execute("ALTER TABLE debate_sessions ADD COLUMN mode TEXT")
    if "synthesis_draft" not in existing:
        conn.execute("ALTER TABLE debate_sessions ADD COLUMN synthesis_draft TEXT")


def _now() -> str:
    return datetime.now().isoformat(sep=" ")


def create_session(
    topic: str,
    user_stance: str,
    user_intensity: int,
    debate_format: str,
    nickname: Optional[str] = None,
    email: Optional[str] = None,
    graph_session_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> int:
    """토론 시작 시점의 세션 행을 만들고 session_id 를 반환한다."""
    with _write_lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO debate_sessions
                (debate_date, nickname, email, topic, user_stance, user_intensity,
                 debate_format, status, graph_session_id, mode, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
            """,
            (_now(), nickname, email, topic, user_stance, user_intensity,
             debate_format, graph_session_id, mode, _now()),
        )
        return int(cur.lastrowid)


def save_evaluation(
    session_id: int,
    pre_pro: str,
    pre_con: str,
    post_pro: str,
    post_con: str,
    result: Dict[str, Any],
) -> bool:
    """토론 전·후 답변과 채점 결과를 기록하고 status 를 COMPLETED 로 올린다.

    `result` 는 evaluation.analyze_user_before_after() 의 반환 구조를 그대로 받는다.
    존재하지 않는 session_id 면 False.
    """
    pro, con = result["pro"], result["con"]
    with _write_lock, _connect() as conn:
        cur = conn.execute(
            """
            UPDATE debate_sessions SET
                pre_pro = ?, pre_con = ?, post_pro = ?, post_con = ?,
                pro_pre_average = ?, pro_post_average = ?, pro_delta = ?,
                con_pre_average = ?, con_post_average = ?, con_delta = ?,
                pro_pre_summary = ?, pro_post_summary = ?,
                con_pre_summary = ?, con_post_summary = ?,
                status = 'COMPLETED', updated_at = ?
            WHERE session_id = ?
            """,
            (
                pre_pro, pre_con, post_pro, post_con,
                pro["pre"]["average_100"], pro["post"]["average_100"], pro["delta_100"],
                con["pre"]["average_100"], con["post"]["average_100"], con["delta_100"],
                pro["pre"].get("overall_summary", ""), pro["post"].get("overall_summary", ""),
                con["pre"].get("overall_summary", ""), con["post"].get("overall_summary", ""),
                _now(), session_id,
            ),
        )
        return cur.rowcount > 0


def save_synthesis(graph_session_id: str, synthesis_draft: str) -> bool:
    """구성적 논쟁에서 합의한 최적해를 세션 행에 기록한다."""
    with _write_lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE debate_sessions SET synthesis_draft = ?, updated_at = ? WHERE graph_session_id = ?",
            (synthesis_draft, _now(), graph_session_id),
        )
        return cur.rowcount > 0


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM debate_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def list_sessions(limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """최신순 세션 목록."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM debate_sessions ORDER BY session_id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def save_turn_analysis(
    graph_session_id: str,
    turn_index: int,
    analysis: Dict[str, Any],
    speech: Optional[Dict[str, Any]] = None,
    live: Optional[Dict[str, Any]] = None,
) -> None:
    """턴 하나의 실시간 분석을 기록한다 (같은 턴 재분석 시 덮어씀)."""
    with _write_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO turn_analyses (graph_session_id, turn_index, analysis, speech, live, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(graph_session_id, turn_index) DO UPDATE SET
                analysis = excluded.analysis,
                speech = excluded.speech,
                live = excluded.live,
                created_at = excluded.created_at
            """,
            (
                graph_session_id, turn_index,
                json.dumps(analysis, ensure_ascii=False),
                json.dumps(speech, ensure_ascii=False) if speech is not None else None,
                json.dumps(live, ensure_ascii=False) if live is not None else None,
                _now(),
            ),
        )


def load_turn_analyses(graph_session_id: str) -> Dict[str, Any]:
    """기록된 턴 분석을 judge 메모리와 같은 형태로 되돌린다.

    Returns:
        {"analysis_memory": [...], "speech_memory": [...], "live_debate": {...}}
        기록이 없으면 세 값 모두 비어 있다.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT analysis, speech, live FROM turn_analyses WHERE graph_session_id = ? ORDER BY turn_index",
            (graph_session_id,),
        ).fetchall()

    analysis_memory = [json.loads(r["analysis"]) for r in rows]
    speech_memory = [json.loads(r["speech"]) for r in rows if r["speech"]]
    live_debate = json.loads(rows[-1]["live"]) if rows and rows[-1]["live"] else {}
    return {
        "analysis_memory": analysis_memory,
        "speech_memory": speech_memory,
        "live_debate": live_debate,
    }


def get_session_by_graph_id(graph_session_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM debate_sessions WHERE graph_session_id = ? ORDER BY session_id DESC LIMIT 1",
            (graph_session_id,),
        ).fetchone()
    return dict(row) if row else None


def save_survey(
    session_id: int,
    phase: str,
    answers: Dict[str, Any],
    mode: Optional[str] = None,
) -> bool:
    """토론 전(pre)·후(post) 설문 응답을 저장한다. 같은 단계 재제출은 덮어쓴다.

    참가자 응답을 잃지 않는 것이 우선이므로 세션 행이 없어도 저장한다.
    Returns:
        해당 session_id 의 세션 행이 실제로 존재했는지 여부.
    """
    with _write_lock, _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM debate_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.execute(
            """
            INSERT INTO survey_responses (session_id, phase, mode, answers, submitted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, phase) DO UPDATE SET
                mode = excluded.mode,
                answers = excluded.answers,
                submitted_at = excluded.submitted_at
            """,
            (session_id, phase, mode, json.dumps(answers, ensure_ascii=False), _now()),
        )
        return exists is not None


def get_survey(session_id: int, phase: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM survey_responses WHERE session_id = ? AND phase = ?",
            (session_id, phase),
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["answers"] = json.loads(record["answers"])
    return record


def _csv_cell(value: Any) -> str:
    """샘플 CSV 표기 규칙: NULL 은 그대로, 숫자는 무따옴표, 문자열은 큰따옴표."""
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace('"', '""') + '"'


def export_csv(path: Optional[Path] = None) -> str:
    """수집된 세션 전체를 CSV 문자열로 만든다 (path 주면 파일로도 저장)."""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(CSV_COLUMNS)} FROM debate_sessions ORDER BY session_id DESC"
        ).fetchall()

    lines = [",".join(f'"{c}"' for c in CSV_COLUMNS)]
    lines.extend(",".join(_csv_cell(row[c]) for c in CSV_COLUMNS) for row in rows)
    content = "\n".join(lines) + "\n"

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content


# 설문 CSV 앞쪽에 붙는 세션 식별 정보
SURVEY_META_COLUMNS = [
    "session_id", "debate_date", "nickname", "topic", "user_stance", "debate_format", "mode",
]


def export_survey_csv(path: Optional[Path] = None) -> str:
    """설문 응답을 세션당 한 행으로 펼쳐 CSV 로 만든다 (pre_*, post_* 컬럼)."""
    from src.survey import answer_keys  # 순환 import 방지를 위해 지연 로딩

    pre_keys, post_keys = answer_keys("pre"), answer_keys("post")
    columns = (
        SURVEY_META_COLUMNS
        + [f"pre_{k}" for k in pre_keys]
        + [f"post_{k}" for k in post_keys]
    )

    with _connect() as conn:
        sessions = conn.execute(
            f"SELECT {', '.join(SURVEY_META_COLUMNS)} FROM debate_sessions ORDER BY session_id DESC"
        ).fetchall()
        responses = conn.execute("SELECT session_id, phase, answers FROM survey_responses").fetchall()

    by_session: Dict[int, Dict[str, dict]] = {}
    for row in responses:
        by_session.setdefault(row["session_id"], {})[row["phase"]] = json.loads(row["answers"])

    def _row(session_id: int, meta: Dict[str, Any]) -> str:
        answers = by_session.get(session_id, {})
        pre, post = answers.get("pre", {}), answers.get("post", {})
        cells = [_csv_cell(meta.get(c)) for c in SURVEY_META_COLUMNS]
        cells += [_csv_cell(pre.get(k)) for k in pre_keys]
        cells += [_csv_cell(post.get(k)) for k in post_keys]
        return ",".join(cells)

    lines = [",".join(f'"{c}"' for c in columns)]
    for session in sessions:
        lines.append(_row(session["session_id"], dict(session)))

    # 세션 행이 없는 응답(세션 기록 유실 등)도 빠뜨리지 않는다.
    known = {s["session_id"] for s in sessions}
    for session_id in sorted(set(by_session) - known, reverse=True):
        lines.append(_row(session_id, {"session_id": session_id}))

    content = "\n".join(lines) + "\n"

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return content
