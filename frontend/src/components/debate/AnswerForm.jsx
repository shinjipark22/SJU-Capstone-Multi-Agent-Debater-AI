import { useState } from 'react';

/**
 * 토론 전(pre)·후(post) 사용자 서술 답변. 양 진영 모두 채우면 /evaluation 이 5개 지표로 채점한다.
 * 사전은 아직 아는 게 없을 수 있으므로 비워도 넘어갈 수 있고, 사후는 필수다.
 */
const AnswerForm = ({ topic, phase, submitting, error, onSubmit }) => {
  const [pro, setPro] = useState('');
  const [con, setCon] = useState('');

  const isPre = phase === 'pre';
  const canSubmit = (isPre || (pro.trim() && con.trim())) && !submitting;

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col overflow-y-auto px-4 py-10 hide-scrollbar">
      <span className="mx-auto mb-4 rounded-full bg-stone-100 px-4 py-1 text-sm font-semibold text-stone-500">
        {isPre ? '토론 전 답변' : '토론 후 답변'}
      </span>
      <h2 className="text-center text-3xl font-extrabold tracking-tight text-stone-800 md:text-4xl">
        {isPre
          ? '이 주제에 대해 알고 있는 내용을 모두 적어주세요'
          : '토론을 마친 지금, 알고 있는 내용을 모두 적어주세요'}
      </h2>
      <p className="mt-3 text-center text-base leading-relaxed text-stone-500">
        동의하지 않는 입장이라도, 알고 있거나 떠오르는 내용을 그대로 적어주시면 됩니다.
        {isPre && ' 잘 모르겠다면 비워두고 넘어가셔도 괜찮습니다.'}
      </p>
      <p className="mt-2 text-center text-sm text-stone-400">{topic.title}</p>

      <div className="mt-8 flex flex-col gap-6">
        <div>
          <label className="mb-2 block text-sm font-bold text-blue-600">
            찬성 입장 · {topic.pro}
            {!isPre && <span className="ml-1 text-rose-500">*</span>}
          </label>
          <textarea
            value={pro}
            onChange={(e) => setPro(e.target.value)}
            rows={5}
            placeholder="찬성 입장에 대해 알고 있는 내용을 모두 적어주세요."
            className="w-full resize-none rounded-2xl border border-stone-200 bg-white px-4 py-3 text-[15px] leading-relaxed outline-none focus:border-blue-400"
          />
        </div>
        <div>
          <label className="mb-2 block text-sm font-bold text-rose-600">
            반대 입장 · {topic.con}
            {!isPre && <span className="ml-1 text-rose-500">*</span>}
          </label>
          <textarea
            value={con}
            onChange={(e) => setCon(e.target.value)}
            rows={5}
            placeholder="반대 입장에 대해 알고 있는 내용을 모두 적어주세요."
            className="w-full resize-none rounded-2xl border border-stone-200 bg-white px-4 py-3 text-[15px] leading-relaxed outline-none focus:border-rose-400"
          />
        </div>
      </div>

      {error && <p className="mt-4 rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">오류: {error}</p>}

      {!isPre && !canSubmit && !submitting && (
        <p className="mt-6 text-center text-sm text-stone-400">양쪽 입장 모두 작성해야 다음으로 넘어갈 수 있습니다.</p>
      )}

      <button
        onClick={() => onSubmit({ pro: pro.trim(), con: con.trim() })}
        disabled={!canSubmit}
        className="mx-auto mt-6 mb-10 rounded-full bg-stone-900 px-12 py-4 text-lg font-bold text-white transition-all hover:bg-black disabled:bg-stone-200 disabled:text-stone-400"
      >
        {submitting ? '저장 중...' : isPre ? '토론 시작하기' : '사후 설문으로'}
      </button>
    </div>
  );
};

export default AnswerForm;
