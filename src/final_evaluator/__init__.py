"""final_evaluator — 다중 에이전트 토론 최종 평가 에이전트.

Analysis Memory를 소비해 프론트엔드 리포트용 구조화 JSON을 생성한다.
"""

from .builder import build_final_report
from .models import FinalReport

__all__ = ["build_final_report", "FinalReport"]
