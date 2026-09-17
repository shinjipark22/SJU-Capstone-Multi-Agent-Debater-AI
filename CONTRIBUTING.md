# 협업 규칙

## 브랜치

- `main`: 발표·제출용. develop에서만 머지한다.
- `develop`: 통합 브랜치. 기능 브랜치는 여기서 나가고 여기로 돌아온다.
- 작업 브랜치: `feature/<이니셜>-<주제>`, `fix/<이니셜>-<주제>`. 예: `feature/psj-latency`, `fix/hcy-rebuttal-order`.

## 커밋 메시지

한국어 한 줄, `영역: 내용` 형식으로 쓴다.

```
README: 구조와 결과 추가
graph: 자유논박 상대 선택 interrupt 추가
```

## Pull Request

- develop을 대상으로 연다.
- 바꾼 이유와 확인 방법을 본문에 적는다.
- `data/cache`, `.env`, 실험 로그는 올리지 않는다. `.gitignore`에 있다.
