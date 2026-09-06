import { useEffect, useRef, useState } from 'react';
import { ApiError, getAssistantGuide, initDebate, submitUserInput } from '../../lib/api';
import Markdown from './Markdown';
import TurnBubble from './TurnBubble';

const WAITING_LABELS = {
  user_opening: '입론을 작성하세요',
  user_rebuttal: '반박을 작성하세요',
  user_select_opponent: '논박할 상대를 선택하세요',
  user_free_rebuttal: '자유논박 발언을 작성하세요',
  user_role_reversal: '상대 입장을 옹호하는 발언을 작성하세요',
  user_synthesis: '의견을 작성하세요',
  user_finalize: '우리의 최적해를 작성하세요',
};

const ASSISTANT_PHASES = new Set([
  'opening',
  'chained_rebuttal',
  'free_rebuttal',
  'role_reversal',
  'synthesis',
]);

const DebateRoom = ({ initRequest, topic, onSessionStart, onFinished }) => {
  const [sessionId, setSessionId] = useState(null);
  const [turns, setTurns] = useState([]);
  const [waitingFor, setWaitingFor] = useState('');
  const [isFinished, setIsFinished] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [error, setError] = useState(null);
  const [livePercents, setLivePercents] = useState(null);
  const [guide, setGuide] = useState('');
  const [guideOpen, setGuideOpen] = useState(true);
  const [draft, setDraft] = useState('');
  const [selectedOpponent, setSelectedOpponent] = useState(null);

  const turnCounter = useRef(0);
  const transcriptEndRef = useRef(null);
  const startedRef = useRef(false);

  const opposingAgents = Array.from(
    new Set(
      turns
        .map((t) => t.entry)
        .filter((e) => e.speaker_id !== 'user' && e.stance !== initRequest.user_stance)
        .map((e) => e.speaker_id),
    ),
  );

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns]);

  const loadGuide = async (id, phase, opponentId) => {
    if (!ASSISTANT_PHASES.has(phase)) return;
    try {
      const res = await getAssistantGuide(id, phase, opponentId);
      setGuide(res.text);
      setGuideOpen(true);
    } catch {
      setGuide('');
    }
  };

  const consumeStream = async (gen) => {
    setIsStreaming(true);
    setError(null);
    let currentId = sessionId;
    try {
      for await (const evt of gen) {
        if (evt.event === 'session') {
          currentId = evt.data.session_id;
          setSessionId(evt.data.session_id);
          onSessionStart?.(evt.data);
        } else if (evt.event === 'turn') {
          turnCounter.current += 1;
          // 분석 결과는 화면에 띄우지 않고 실시간 평가 지수 갱신에만 쓴다 (상세는 최종 리포트에서).
          const { entry, analysis } = evt.data;
          setTurns((prev) => [...prev, { key: `${turnCounter.current}-${entry.speaker_id}`, entry }]);
          if (analysis?.pro_percent != null && analysis?.con_percent != null) {
            setLivePercents({ pro: analysis.pro_percent, con: analysis.con_percent });
          }
        } else if (evt.event === 'waiting') {
          setWaitingFor(evt.data.waiting_for);
          setIsFinished(evt.data.is_finished);
          setSelectedOpponent(null);
          setGuide('');
          if (!evt.data.is_finished && currentId) loadGuide(currentId, evt.data.phase);
        }
      }
    } catch (e) {
      setError(e instanceof ApiError ? String(e.detail ?? e.message) : String(e));
    } finally {
      setIsStreaming(false);
    }
  };

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    consumeStream(initDebate(initRequest));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!sessionId || isStreaming) return;

    if (waitingFor === 'user_select_opponent') {
      if (!selectedOpponent) return;
      consumeStream(submitUserInput(sessionId, { content: selectedOpponent, target_id: selectedOpponent }));
      return;
    }

    if (!draft.trim()) return;
    const targetId = waitingFor === 'user_free_rebuttal' ? opposingAgents[0] ?? null : null;
    consumeStream(submitUserInput(sessionId, { content: draft.trim(), target_id: targetId }));
    setDraft('');
  };

  const waitingLabel = WAITING_LABELS[waitingFor] ?? '';

  return (
    <div className="mx-auto flex h-full w-full max-w-4xl flex-col px-4 pb-6 pt-6">
      <header className="mb-4 shrink-0 rounded-3xl border border-stone-200 bg-white/90 px-6 py-5 shadow-sm">
        <h2 className="text-lg font-extrabold leading-snug text-stone-800">{topic.title}</h2>
        <div className="mt-2 flex flex-wrap gap-2 text-xs font-semibold">
          <span className="rounded-full bg-blue-50 px-3 py-1 text-blue-600">찬성 · {topic.pro}</span>
          <span className="rounded-full bg-rose-50 px-3 py-1 text-rose-600">반대 · {topic.con}</span>
          <span className="rounded-full bg-stone-100 px-3 py-1 text-stone-600">{initRequest.debate_format}</span>
        </div>

        {/* 실시간 평가 — 진영별 지수이동평균(EMA) 점수에서 나온 우세 지수. 턴별 상세 분석은 노출하지 않는다. */}
        {livePercents && (
          <div className="mt-4">
            <div className="flex items-baseline justify-center gap-2">
              <span className="text-xs font-semibold text-stone-400">실시간 평가</span>
              <span className="text-xl font-extrabold tabular-nums text-blue-500">
                {livePercents.pro.toFixed(0)}
              </span>
              <span className="text-lg font-bold text-stone-300">:</span>
              <span className="text-xl font-extrabold tabular-nums text-rose-500">
                {livePercents.con.toFixed(0)}
              </span>
            </div>
            <div className="mt-2 flex h-3 overflow-hidden rounded-full bg-stone-100">
              <div className="bg-blue-500 transition-all duration-500" style={{ width: `${livePercents.pro}%` }} />
              <div className="bg-rose-500 transition-all duration-500" style={{ width: `${livePercents.con}%` }} />
            </div>
          </div>
        )}
      </header>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto rounded-3xl px-1 py-2 hide-scrollbar">
        {turns.map((t) => (
          <TurnBubble key={t.key} entry={t.entry} />
        ))}
        {isStreaming && (
          <p className="py-4 text-center text-sm font-medium text-stone-400">AI가 발언을 생성하는 중...</p>
        )}
        <div ref={transcriptEndRef} />
      </div>

      {error && (
        <p className="mt-3 shrink-0 rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">오류: {error}</p>
      )}

      {guide && !isFinished && (
        <div className="mt-3 shrink-0 rounded-2xl border border-emerald-100 bg-emerald-50 text-sm text-emerald-900">
          <button
            type="button"
            onClick={() => setGuideOpen((open) => !open)}
            className="flex w-full items-center justify-between px-4 py-2.5 text-left font-bold"
          >
            <span>비비드의 단계 안내</span>
            <span className="text-xs font-semibold opacity-70">{guideOpen ? '접기' : '펼치기'}</span>
          </button>
          {guideOpen && (
            <div className="max-h-44 overflow-y-auto px-4 pb-3 leading-relaxed">
              <Markdown>{guide}</Markdown>
            </div>
          )}
        </div>
      )}

      {!isFinished && sessionId && (
        <form className="mt-3 shrink-0 rounded-3xl border border-stone-200 bg-white p-4 shadow-sm" onSubmit={handleSubmit}>
          {waitingLabel && <p className="mb-2 text-sm font-bold text-stone-700">{waitingLabel}</p>}

          {waitingFor === 'user_select_opponent' ? (
            <div className="flex flex-wrap items-center gap-2">
              {opposingAgents.length === 0 && <p className="text-sm text-stone-400">선택 가능한 상대가 없습니다.</p>}
              {opposingAgents.map((id) => (
                <button
                  type="button"
                  key={id}
                  onClick={() => setSelectedOpponent(id)}
                  className={`rounded-full px-5 py-2.5 text-sm font-semibold transition-all ${
                    selectedOpponent === id
                      ? 'bg-stone-900 text-white'
                      : 'border border-stone-200 bg-white text-stone-600 hover:border-stone-300'
                  }`}
                >
                  {id.replace('agent_', 'AI ')}
                </button>
              ))}
              <button
                type="submit"
                disabled={!selectedOpponent || isStreaming}
                className="ml-auto rounded-full bg-stone-900 px-6 py-2.5 text-sm font-bold text-white disabled:bg-stone-200 disabled:text-stone-400"
              >
                선택 완료
              </button>
            </div>
          ) : (
            <>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="발언을 입력하세요..."
                rows={4}
                disabled={isStreaming}
                className="w-full resize-none rounded-2xl border border-stone-200 px-4 py-3 text-[15px] leading-relaxed outline-none focus:border-stone-400 disabled:bg-stone-50"
              />
              <div className="mt-2 flex justify-end">
                <button
                  type="submit"
                  disabled={isStreaming || !draft.trim()}
                  className="rounded-full bg-stone-900 px-8 py-2.5 text-sm font-bold text-white transition-all hover:bg-black disabled:bg-stone-200 disabled:text-stone-400"
                >
                  제출
                </button>
              </div>
            </>
          )}
        </form>
      )}

      {isFinished && sessionId && (
        <div className="mt-3 flex shrink-0 items-center justify-between rounded-3xl border border-stone-200 bg-white px-6 py-5 shadow-sm">
          <p className="text-sm font-semibold text-stone-700">토론이 종료되었습니다.</p>
          <button
            onClick={() => onFinished(sessionId)}
            className="rounded-full bg-stone-900 px-7 py-2.5 text-sm font-bold text-white hover:bg-black"
          >
            다음 단계로
          </button>
        </div>
      )}
    </div>
  );
};

export default DebateRoom;
