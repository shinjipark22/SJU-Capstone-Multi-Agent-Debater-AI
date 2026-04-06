"""
topic_recsys — 토론 주제 자동 추천 시스템

NewsAPI 뉴스 크롤링 → Ollama LLM 주제 생성 → 안전 필터링 → JSON 저장
"""

from .main import run

__all__ = ["run"]
