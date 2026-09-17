import { useEffect, useState } from 'react';
import { ApiError, getFinalReport } from '../../lib/api';

const SWING_TYPE_LABELS = {
  biggest_swing: '최대 반전',
  best_rebuttal: '최고의 반박',
  worst_turn: '아쉬운 발언',
  logical_error: '논리적 허점',
};

const DIMENSION_LABELS = { argument: '논증', evidence: '근거', language: '표현' };

const SideCard = ({ label, tone, side }) => {
  const delta = side.delta_100;
  return (
    <div className="rounded-3xl border border-stone-200 bg-white p-6 shadow-sm">
      <h4 className={`text-sm font-bold ${tone === 'pro' ? 'text-blue-600' : 'text-rose-600'}`}>{label}</h4>
      <div className="mt-3 flex items-end gap-3">
        <span className="text-3xl font-extrabold text-stone-800">{side.post.average_100.toFixed(1)}</span>
        <span className="pb-1 text-sm text-stone-400">/ 토론 전 {side.pre.average_100.toFixed(1)}</span>
        <span
          className={`ml-auto pb-1 text-sm font-bold ${
            delta > 0 ? 'text-emerald-600' : delta < 0 ? 'text-rose-500' : 'text-stone-400'
          }`}
        >
          {delta > 0 ? '+' : ''}
          {delta.toFixed(1)}
        </span>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-stone-500">{side.post.overall_summary}</p>
    </div>
  );
};

const ResultView = ({ sessionId, evaluation, onRestart }) => {
  const [report, setReport] = useState(null);
  const [reportError, setReportError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    getFinalReport(sessionId, false, controller.signal)
      .then(setReport)
      .catch((e) => {
        if (e.name !== 'AbortError') {
          setReportError(e instanceof ApiError ? String(e.detail ?? e.message) : String(e));
        }
      });
    return () => controller.abort();
  }, [sessionId]);

  return (
    <div className="mx-auto h-full w-full max-w-4xl overflow-y-auto px-4 py-10 hide-scrollbar">
      <h2 className="text-center text-4xl font-extrabold tracking-tight text-stone-800">토론 리포트</h2>
      <p className="mt-3 text-center text-base text-stone-500">{evaluation.topic}</p>

      <section className="mt-8">
        <h3 className="mb-4 text-xl font-bold text-stone-800">토론 전·후 내 답변 변화</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <SideCard label={`찬성 · ${evaluation.pro_label}`} tone="pro" side={evaluation.pro} />
          <SideCard label={`반대 · ${evaluation.con_label}`} tone="con" side={evaluation.con} />
        </div>
      </section>

      {report && (
        <>
          <section className="mt-10 rounded-3xl border border-stone-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-bold text-stone-800">
              {report.winner.side === 'DRAW' ? '무승부' : `${report.winner.side === 'PRO' ? '찬성' : '반대'} 우세`}
            </h3>
            <div className="mt-4 flex h-3 overflow-hidden rounded-full bg-stone-100">
              <div className="bg-blue-500" style={{ width: `${report.winner.pro_percent}%` }} />
              <div className="bg-rose-500" style={{ width: `${report.winner.con_percent}%` }} />
            </div>
            <p className="mt-2 text-xs font-medium text-stone-500">
              찬성 {report.winner.pro_percent.toFixed(0)}% · 반대 {report.winner.con_percent.toFixed(0)}%
            </p>
            <p className="mt-4 text-sm leading-relaxed text-stone-600">{report.winner.summary}</p>
          </section>

          {report.swing_turns.length > 0 && (
            <section className="mt-6">
              <h3 className="mb-4 text-xl font-bold text-stone-800">주요 발언</h3>
              <ul className="flex flex-col gap-3">
                {report.swing_turns.map((t, idx) => (
                  <li key={idx} className="rounded-2xl border border-stone-200 bg-white px-5 py-4 shadow-sm">
                    <span className="rounded-full bg-stone-100 px-3 py-1 text-xs font-bold text-stone-600">
                      {SWING_TYPE_LABELS[t.type] ?? t.type}
                    </span>
                    <span className="ml-2 text-sm font-semibold text-stone-700">
                      {t.speaker_id === 'user' ? '나' : t.speaker_id.replace('agent_', 'AI ')}
                    </span>
                    <p className="mt-2 text-sm leading-relaxed text-stone-600">{t.narrative}</p>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section className="mt-6">
            <h3 className="mb-4 text-xl font-bold text-stone-800">코치 피드백</h3>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.keys(DIMENSION_LABELS).map((dim) => (
                <div key={dim} className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
                  <h4 className="mb-3 text-sm font-bold text-stone-700">{DIMENSION_LABELS[dim]}</h4>
                  <p className="text-sm leading-relaxed text-stone-600">👍 {report.coach_feedback[dim].praise}</p>
                  <p className="mt-2 text-sm leading-relaxed text-stone-600">⚠️ {report.coach_feedback[dim].critique}</p>
                  <p className="mt-2 text-sm leading-relaxed text-stone-600">💡 {report.coach_feedback[dim].suggestion}</p>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {reportError && (
        <p className="mt-6 rounded-2xl bg-stone-100 px-4 py-3 text-sm text-stone-500">
          토론 분석 리포트를 불러오지 못했습니다: {reportError}
        </p>
      )}

      <div className="mt-10 flex justify-center pb-10">
        <button
          onClick={onRestart}
          className="rounded-full bg-stone-900 px-12 py-4 text-lg font-bold text-white transition-all hover:bg-black"
        >
          새 토론 시작
        </button>
      </div>
    </div>
  );
};

export default ResultView;
