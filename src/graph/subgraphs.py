"""
subgraphs.py — LangGraph 서브그래프 빌더

각 토론 단계의 1턴 파이프라인을 StateGraph로 구성.
Search → Write → Review → (조건부 재시도 or 완료)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.graph.turn_state import TurnState
from src.graph.nodes import search_node, write_node, review_node, route_review


def build_search_write_review_graph():
    """범용 서브그래프: Search → Write → Review → (pass or retry)

    자유논박 공격, 연쇄논박, 역할반전 등 대부분의 발언 생성에 사용.
    """
    graph = StateGraph(TurnState)

    graph.add_node("search", search_node)
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", route_review, {
        "pass": END,
        "retry": "write",
    })

    return graph.compile()


def build_write_review_graph():
    """검색 없는 서브그래프: Write → Review → (pass or retry)

    방어 발언, 종합 회의 등 검색이 불필요한 경우.
    """
    graph = StateGraph(TurnState)

    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", route_review, {
        "pass": END,
        "retry": "write",
    })

    return graph.compile()


# ── 컴파일된 서브그래프 인스턴스 (싱글톤) ──────────────────────────────────
# 매번 컴파일하지 않고 모듈 로드 시 1회 컴파일

search_write_review = build_search_write_review_graph()
write_review = build_write_review_graph()
