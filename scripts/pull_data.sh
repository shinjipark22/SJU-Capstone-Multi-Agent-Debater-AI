#!/bin/bash
# 배포 서버의 수집 데이터를 한 번에 내려받는다.
#   bash scripts/pull_data.sh            → ./exports/<날짜시각>/ 에 sessions.csv, surveys.csv, sessions.db
set -e
HOST="${NCP_HOST:-ubuntu@211.233.220.12}"
BASE="${BASE_URL:-http://211.233.220.12}"
OUT="exports/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
curl -sf -o "$OUT/sessions.csv" "$BASE/sessions/export.csv"     # 팀 CSV 와 같은 23컬럼
curl -sf -o "$OUT/surveys.csv"  "$BASE/surveys/export.csv"      # 설문 58컬럼 (세션당 한 행)
scp -q "$HOST:/opt/debate/data/sessions.db" "$OUT/sessions.db"  # 원본 DB (턴별 분석·최적해 포함)
echo "저장: $OUT"
python3 - "$OUT" << 'PY'
import csv, sys, sqlite3, pathlib
out = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "sessions.csv", encoding="utf-8")))
done = sum(1 for r in rows if r["status"] == "COMPLETED")
print(f"세션 {len(rows)}건 (완료 {done}, 진행중/이탈 {len(rows) - done})")
c = sqlite3.connect(out / "sessions.db")
print("설문 응답:", c.execute("SELECT COUNT(*) FROM survey_responses").fetchone()[0], "행 | 턴 분석:", c.execute("SELECT COUNT(*) FROM turn_analyses").fetchone()[0], "행")
PY
