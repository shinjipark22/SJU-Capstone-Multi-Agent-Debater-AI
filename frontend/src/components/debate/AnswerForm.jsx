import { useState } from 'react';

/** 토론 전(pre)·후(post) 사용자 답변 입력. 두 진영 모두 작성해야 채점이 가능하다. */
const AnswerForm = ({ topic, phase, submitting, error, onSubmit }) => {
  const [pro, setPro] = useState('');
  const [con, setCon] = useState('');

  const isPre = phase === 'pre';
  const canSubmit = pro.trim() && con.trim() && !submitting;

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col overflow-y-auto px-4 py-10 hide-scrollbar">
      <span className="mx-auto mb-4 rounded-full bg-stone-100 px-4 py-1 text-sm font-semibold text-stone-500">
        {isPre ? '토론 전 답변' : '토론 후 답변'}
      </span>
      <h2 className="text-center text-3xl font-extrabold tracking-tight text-stone-800 md:text-4xl">
        {isPre ? '토론에 들어가기 전, 지금 생각을 적어주세요' : '토론을 마친 지금, 생각이 어떻게 바뀌었나요?'}
      </h2>
      <p className="mt-3 text-center text-base text-stone-500">{topic.title}</p>

      <div className="mt-8 flex flex-col gap-6">
        <div>
          <label className="mb-2 block text-sm font-bold text-blue-600">찬성 입장 · {topic.pro}</label>
          <textarea
            value={pro}
            onChange={(e) => setPro(e.target.value)}
            rows={5}
            placeholder="찬성 입장에서 근거를 들어 서술해주세요."
            className="w-full resize-none rounded-2xl border border-stone-200 bg-white px-4 py-3 text-[15px] leading-relaxed outline-none focus:border-blue-400"
          />
        </div>
        <div>
          <label className="mb-2 block text-sm font-bold text-rose-600">반대 입장 · {topic.con}</label>
          <textarea
            value={con}
            onChange={(e) => setCon(e.target.value)}
            rows={5}
            placeholder="반대 입장에서 근거를 들어 서술해주세요."
            className="w-full resize-none rounded-2xl border border-stone-200 bg-white px-4 py-3 text-[15px] leading-relaxed outline-none focus:border-rose-400"
          />
        </div>
      </div>

      {error && <p className="mt-4 rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">오류: {error}</p>}

      <button
        onClick={() => onSubmit({ pro: pro.trim(), con: con.trim() })}
        disabled={!canSubmit}
        className="mx-auto mt-8 rounded-full bg-stone-900 px-12 py-4 text-lg font-bold text-white transition-all hover:bg-black disabled:bg-stone-200 disabled:text-stone-400"
      >
        {submitting ? '채점 중...' : isPre ? '토론 시작하기' : '결과 확인하기'}
      </button>
    </div>
  );
};

export default AnswerForm;
