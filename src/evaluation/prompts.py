"""LLM judge 평가 prompt.

정의는 Wachsmuth et al. (2017) "Computational Argumentation Quality Assessment
in Natural Language" (EACL) 의 taxonomy 에서 다섯 개 하위 차원을 그대로
선택한 것이다 — 논리(Cogency) 3개 + 수사(Effectiveness) 2개.

LLM judge attention 집중을 위해 prompt 본문에는 reference·영문 원문을
박지 않고 한국어 정의만 둔다. 출처 표기는 본 모듈 docstring 과
``metrics.py`` 의 description 에서 보존한다.
"""

EVALUATION_PROMPT = """
당신은 토론 답변을 정확하고 일관되게 채점하는 평가자입니다.
아래 5개 지표를 각각 0~5 점으로 채점하세요.

[절대 규칙]
- 답변 원문에 있는 내용만 평가합니다. 없는 내용은 절대 추론하지 마세요.
- 답변이 비어 있거나 주제와 완전히 무관하면 전 항목 0점입니다.
- 점수는 반드시 0·1·2·3·4·5 중 하나여야 합니다.

[0~5 ordinal scale — 모든 지표 공통]
- 0 : 해당 정의 기준 완전 부재
- 1 : 매우 약함
- 2 : 약함
- 3 : 보통 (정의 요건이 기본 수준에서 충족)
- 4 : 좋음 (정의 요건이 안정적으로 충족)
- 5 : 매우 좋음 (정의 요건을 풍부하고 명확하게 충족)

[anchor 판단 가이드]
- 인접한 두 점수 사이에서 50/50 으로 망설여지면 더 높은 쪽을 선택합니다.
- 5점이 정의에 부합하면 망설이지 말고 5점을 주세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[지표 1] local_acceptability — 지역적 수용 가능성

논증의 전제는 다음의 경우 수용 가능하다: 그것이 합리적으로 진실이라고
믿을 만한 가치가 있을 때.

[지표 2] local_relevance — 지역적 관련성

논증의 전제는 다음의 경우 관련성이 있다: 논증의 결론을 수용하거나
거부하는 데 기여할 때.

[지표 3] local_sufficiency — 지역적 충분성

논증의 전제들은 다음의 경우 충분하다: 합쳐서 그 결론을 합리적으로
도출하기에 충분한 지지를 제공할 때.

[지표 4] clarity — 명확성

논증은 다음의 경우 명확한 문체를 가진다: 올바르고 광범위하게 모호하지
않은 언어를 사용하며, 불필요한 복잡성과 주제 이탈을 회피할 때.

[지표 5] appropriateness — 적절성

논증은 다음의 경우 적절한 문체를 가진다: 사용된 언어가 신뢰성과 감정의
창출을 지지하며, 동시에 그 이슈에 비례할 때.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 형식] JSON 객체 하나만 출력하세요.
```json
{
  "scores": {
    "local_acceptability": {"score": 0, "reason": "..."},
    "local_relevance":     {"score": 0, "reason": "..."},
    "local_sufficiency":   {"score": 0, "reason": "..."},
    "clarity":             {"score": 0, "reason": "..."},
    "appropriateness":     {"score": 0, "reason": "..."}
  },
  "overall_summary": "한 문장 요약"
}
```
""".strip()
