"""sessions.py — 토론 세션 수집 저장소.

`data-1787155428828.csv` 와 동일한 23개 컬럼(순서 포함)으로 세션을 누적한다.
저장소는 SQLite 파일 하나(`data/sessions.db`, DEBATE_DB_PATH 로 변경 가능)이고,
CSV 는 export 시점에 그 테이블에서 그대로 뽑아 쓴다.

    /debate/init   → create_session()   : 세션 메타 (닉네임·주제·진영·강도·1:1/2:2) 기록
    /evaluation    → save_evaluation()  : 토론 전·후 답변 + 채점 결과(평균/델타/요약) 기록
"""

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
    updated_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_graph_id ON debate_sessions(graph_session_id);
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
    return conn


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
) -> int:
    """토론 시작 시점의 세션 행을 만들고 session_id 를 반환한다."""
    with _write_lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO debate_sessions
                (debate_date, nickname, email, topic, user_stance, user_intensity,
                 debate_format, status, graph_session_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """,
            (_now(), nickname, email, topic, user_stance, user_intensity,
             debate_format, graph_session_id, _now()),
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
