"""수집된 토론 세션을 data-*.csv 와 동일한 형식의 CSV 로 내보낸다.

    python scripts/export_sessions.py                    # stdout
    python scripts/export_sessions.py -o sessions.csv    # 파일로 저장
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import export_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, help="저장할 CSV 경로 (미지정 시 stdout)")
    args = parser.parse_args()

    content = export_csv(args.output)
    if args.output is None:
        sys.stdout.write(content)
    else:
        print(f"{args.output} 저장 완료")


if __name__ == "__main__":
    main()
