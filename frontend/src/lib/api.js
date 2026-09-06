// FastAPI 백엔드(src/main.py) 클라이언트.
//
// 기본값은 빈 문자열 = 페이지와 같은 출처로 호출한다. dev 서버는 vite.config.js 의
// 프록시가, 배포 시에는 정적 호스팅 앞단(리버스 프록시)이 백엔드로 넘겨주면 된다.
// 프론트와 API 를 다른 도메인에 둘 경우에만 frontend/.env 의 VITE_API_BASE_URL 을 설정.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function throwIfNotOk(res) {
  if (!res.ok) {
    let detail;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text().catch(() => res.statusText);
    }
    throw new ApiError(res.status, detail?.detail ?? detail);
  }
  return res;
}

export async function getTopics(signal) {
  const res = await fetch(`${API_BASE_URL}/topics`, { signal });
  await throwIfNotOk(res);
  return res.json();
}

export async function getFinalReport(sessionId, refresh = false, signal) {
  const res = await fetch(
    `${API_BASE_URL}/debate/${sessionId}/final-report?refresh=${refresh}`,
    { signal },
  );
  await throwIfNotOk(res);
  return res.json();
}

export async function getAssistantGuide(sessionId, phase, opponentId, signal) {
  const qs = opponentId ? `?opponent_id=${encodeURIComponent(opponentId)}` : '';
  const res = await fetch(`${API_BASE_URL}/debate/${sessionId}/assistant/${phase}${qs}`, { signal });
  await throwIfNotOk(res);
  return res.json();
}

/**
 * 토론 전·후 답변 채점. recordId 를 주면 백엔드 수집 테이블에 함께 저장된다
 * (data-*.csv 의 pre_/post_/average/delta/summary 컬럼).
 */
export async function submitEvaluation({ topicId, recordId, prePro, preCon, postPro, postCon }, signal) {
  const qs = new URLSearchParams({ topic_id: topicId });
  if (recordId != null) qs.set('session_id', String(recordId));
  const res = await fetch(`${API_BASE_URL}/evaluation?${qs}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pre_pro: prePro, pre_con: preCon, post_pro: postPro, post_con: postCon }),
    signal,
  });
  await throwIfNotOk(res);
  return res.json();
}

/** 연구 설문 문항 스키마 (주제·모드 문구가 치환된 상태로 온다). */
export async function getSurveySchema(phase, mode, topicId, signal) {
  const qs = new URLSearchParams({ mode });
  if (topicId) qs.set('topic_id', topicId);
  const res = await fetch(`${API_BASE_URL}/survey/schema/${phase}?${qs}`, { signal });
  await throwIfNotOk(res);
  return res.json();
}

/** 설문 응답 저장. 같은 단계를 다시 제출하면 덮어쓴다. */
export async function submitSurvey(sessionId, phase, answers, signal) {
  const res = await fetch(`${API_BASE_URL}/sessions/${sessionId}/survey`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phase, answers }),
    signal,
  });
  await throwIfNotOk(res);
  return res.json();
}

/**
 * text/event-stream 바디를 파싱해 { event, data } 를 순서대로 yield.
 * 백엔드가 EventSource(GET 전용) 대신 POST + StreamingResponse 를 쓰므로 직접 파싱한다.
 */
async function* parseSSEStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIndex;
      while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);

        let eventType = 'message';
        const dataLines = [];
        for (const line of rawEvent.split('\n')) {
          if (line.startsWith('event:')) eventType = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
        }
        if (dataLines.length > 0) yield { event: eventType, data: dataLines.join('\n') };
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function* streamDebateSSE(res) {
  await throwIfNotOk(res);
  if (!res.body) throw new Error('응답에 스트림 바디가 없습니다.');
  for await (const { event, data } of parseSSEStream(res.body)) {
    if (event === 'session' || event === 'turn' || event === 'waiting') {
      yield { event, data: JSON.parse(data) };
    }
  }
}

export function initDebate(request, signal) {
  const streamPromise = fetch(`${API_BASE_URL}/debate/init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  return (async function* () {
    yield* streamDebateSSE(await streamPromise);
  })();
}

export function submitUserInput(sessionId, request, signal) {
  const streamPromise = fetch(`${API_BASE_URL}/debate/${sessionId}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  return (async function* () {
    yield* streamDebateSSE(await streamPromise);
  })();
}
