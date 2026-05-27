"""평가 지표 정의.

Wachsmuth et al. (2017) "Computational Argumentation Quality Assessment in
Natural Language" (EACL) 의 argumentation quality taxonomy 에서 다섯 개
하위 차원을 그대로 선택해 사용한다 — 논리(Cogency) 3개 + 수사(Effectiveness)
2개. 정의는 모두 논문 원문에서 직접 인용한 표현을 한국어로 옮긴 것이며,
임의로 가공된 anchor·예시는 두지 않는다.
"""

METRICS = [
    {
        "key": "local_acceptability",
        "label": "수용 가능성",
        "alias": "acceptability",
        "description": (
            "논증의 전제는 다음의 경우 수용 가능하다: 그것이 합리적으로 진실이라고 "
            "믿을 만한 가치가 있을 때. (Wachsmuth et al., 2017)"
        ),
    },
    {
        "key": "local_relevance",
        "label": "관련성",
        "alias": "relevance",
        "description": (
            "논증의 전제는 다음의 경우 관련성이 있다: 논증의 결론을 수용하거나 "
            "거부하는 데 기여할 때. (Wachsmuth et al., 2017)"
        ),
    },
    {
        "key": "local_sufficiency",
        "label": "충분성",
        "alias": "sufficiency",
        "description": (
            "논증의 전제들은 다음의 경우 충분하다: 합쳐서 그 결론을 합리적으로 "
            "도출하기에 충분한 지지를 제공할 때. (Wachsmuth et al., 2017)"
        ),
    },
    {
        "key": "clarity",
        "label": "명확성",
        "alias": "clarity",
        "description": (
            "논증은 다음의 경우 명확한 문체를 가진다: 올바르고 광범위하게 모호하지 "
            "않은 언어를 사용하며, 불필요한 복잡성과 주제 이탈을 회피할 때. "
            "(Wachsmuth et al., 2017)"
        ),
    },
    {
        "key": "appropriateness",
        "label": "적절성",
        "alias": "appropriateness",
        "description": (
            "논증은 다음의 경우 적절한 문체를 가진다: 사용된 언어가 신뢰성과 감정의 "
            "창출을 지지하며, 동시에 그 이슈에 비례할 때. (Wachsmuth et al., 2017)"
        ),
    },
]

METRIC_LABELS = [metric["label"] for metric in METRICS]
