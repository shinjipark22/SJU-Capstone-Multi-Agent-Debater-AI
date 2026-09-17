import Markdown from './Markdown';

const PHASE_LABELS = {
  opening: '입론',
  chained_rebuttal: '연쇄 논박',
  free_rebuttal: '자유 논박',
  role_reversal: '역할 반전',
  synthesis: '종합',
};

const ScoreChip = ({ label, value }) =>
  value == null ? null : (
    <span className="rounded-full bg-stone-100 px-2.5 py-1 text-xs font-semibold text-stone-600">
      {label} {value.toFixed(1)}
    </span>
  );

const TurnBubble = ({ entry, analysis }) => {
  const isUser = entry.speaker_id === 'user';
  const isPro = entry.stance === 'PRO';
  const speakerLabel = isUser ? '나' : entry.speaker_id.replace('agent_', 'AI ');

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] md:max-w-[70%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-2`}>
        <div className="flex items-center gap-2 text-xs font-semibold text-stone-500">
          <span className={`rounded-full px-2 py-0.5 text-white ${isPro ? 'bg-blue-500' : 'bg-rose-500'}`}>
            {isPro ? '찬성' : '반대'}
          </span>
          <span>{speakerLabel}</span>
          <span className="text-stone-400">{PHASE_LABELS[entry.phase] ?? entry.phase}</span>
          {entry.target_id && <span className="text-stone-400">→ {entry.target_id}</span>}
        </div>

        <div
          className={`rounded-3xl border px-5 py-4 text-[15px] leading-relaxed shadow-sm ${
            isUser
              ? 'border-stone-800 bg-stone-900 text-white'
              : 'border-stone-200 bg-white text-stone-800'
          }`}
        >
          <Markdown>{entry.content}</Markdown>
        </div>

        {analysis && (
          <div className="flex flex-col gap-1.5 rounded-2xl bg-stone-50 px-4 py-3">
            <div className="flex flex-wrap gap-1.5">
              <ScoreChip label="논증" value={analysis.argument_score} />
              <ScoreChip label="근거" value={analysis.evidence_score} />
              <ScoreChip label="표현" value={analysis.language_score} />
            </div>
            {analysis.overall_feedback && (
              <p className="text-xs leading-relaxed text-stone-500">{analysis.overall_feedback}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TurnBubble;
