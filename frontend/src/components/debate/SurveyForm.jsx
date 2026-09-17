import { useEffect, useMemo, useState } from 'react';
import { ApiError, getSurveySchema } from '../../lib/api';

/**
 * 연구 설문(구글폼 4종 대체) 렌더러.
 * 문항 정의는 백엔드 /survey/schema/{phase} 가 단일 원본이라 여기서는 타입별 표시만 담당한다.
 */

const ScaleInput = ({ item, value, onChange }) => (
  <div className="flex flex-wrap items-center gap-2">
    {item.scale_low && <span className="text-xs text-stone-400">{item.scale_low}</span>}
    {item.options.map((opt) => {
      const selected = String(value) === String(opt);
      return (
        <button
          type="button"
          key={opt}
          onClick={() => onChange(Number(opt))}
          className={`h-10 w-10 rounded-full text-sm font-bold transition-all ${
            selected
              ? 'bg-stone-900 text-white'
              : 'border border-stone-200 bg-white text-stone-500 hover:border-stone-400'
          }`}
        >
          {opt}
        </button>
      );
    })}
  </div>
);

const RadioInput = ({ item, value, onChange }) => (
  <div className="flex flex-wrap gap-2">
    {item.options.map((opt) => (
      <button
        type="button"
        key={opt}
        onClick={() => onChange(opt)}
        className={`rounded-full px-4 py-2 text-sm font-medium transition-all ${
          value === opt
            ? 'bg-stone-900 text-white'
            : 'border border-stone-200 bg-white text-stone-600 hover:border-stone-400'
        }`}
      >
        {opt}
      </button>
    ))}
  </div>
);

const QuestionField = ({ item, value, onChange }) => {
  if (item.type === 'linear_scale') return <ScaleInput item={item} value={value} onChange={onChange} />;
  if (item.type === 'radio' || item.type === 'dropdown') {
    return <RadioInput item={item} value={value} onChange={onChange} />;
  }
  if (item.type === 'paragraph') {
    return (
      <textarea
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        className="w-full resize-none rounded-2xl border border-stone-200 bg-white px-4 py-3 text-[15px] outline-none focus:border-stone-400"
      />
    );
  }
  return (
    <input
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      className="w-full max-w-sm rounded-full border border-stone-200 bg-white px-5 py-2.5 text-[15px] outline-none focus:border-stone-400"
    />
  );
};

const SurveyForm = ({ phase, mode, topicId, topic, submitting, error, onSubmit }) => {
  const [sections, setSections] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [answers, setAnswers] = useState({});
  const [showMissing, setShowMissing] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getSurveySchema(phase, mode, topicId, controller.signal)
      .then((res) => setSections(res.sections))
      .catch((e) => {
        if (e.name !== 'AbortError') {
          setLoadError(e instanceof ApiError ? String(e.detail ?? e.message) : String(e));
        }
      });
    return () => controller.abort();
  }, [phase, mode, topicId]);

  // 조건부 섹션(대학생/직장인 추가 질문)은 상위 응답에 따라 노출된다.
  const visibleSections = useMemo(
    () =>
      (sections ?? []).filter(
        (s) => !s.depends_on || answers[s.depends_on.key] === s.depends_on.equals,
      ),
    [sections, answers],
  );

  const missingKeys = useMemo(() => {
    const required = visibleSections.flatMap((s) => s.items.filter((i) => i.required));
    return required.filter((i) => answers[i.key] === undefined || answers[i.key] === '').map((i) => i.key);
  }, [visibleSections, answers]);

  const isPre = phase === 'pre';

  if (loadError) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-16">
        <p className="rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">
          설문 문항을 불러오지 못했습니다: {loadError}
        </p>
      </div>
    );
  }

  if (!sections) {
    return <div className="flex h-full items-center justify-center text-stone-400">설문을 불러오는 중...</div>;
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col overflow-y-auto px-4 py-10 hide-scrollbar">
      <span className="mx-auto mb-4 rounded-full bg-stone-100 px-4 py-1 text-sm font-semibold text-stone-500">
        {isPre ? '사전 설문' : '사후 설문'}
      </span>
      <h2 className="text-center text-3xl font-extrabold tracking-tight text-stone-800 md:text-4xl">
        {isPre ? '토론 전 설문에 응답해주세요' : '토론 후 설문에 응답해주세요'}
      </h2>
      <p className="mt-3 text-center text-sm leading-relaxed text-stone-500">{topic.title}</p>

      <div className="mt-8 flex flex-col gap-8">
        {visibleSections.map((section, si) => (
          <section key={section.section ?? si} className="rounded-3xl border border-stone-200 bg-white p-6 shadow-sm">
            {section.section && <h3 className="text-lg font-bold text-stone-800">{section.section}</h3>}
            {section.description && (
              <p className="mt-2 text-sm leading-relaxed text-stone-500">{section.description}</p>
            )}
            <div className="mt-5 flex flex-col gap-6">
              {section.items.map((item) => (
                <div key={item.key}>
                  <p className="mb-3 text-[15px] font-medium leading-relaxed text-stone-700">
                    {item.title}
                    {item.required && <span className="ml-1 text-rose-500">*</span>}
                  </p>
                  <QuestionField
                    item={item}
                    value={answers[item.key]}
                    onChange={(v) => setAnswers((prev) => ({ ...prev, [item.key]: v }))}
                  />
                  {showMissing && missingKeys.includes(item.key) && (
                    <p className="mt-2 text-xs font-semibold text-rose-500">응답이 필요한 문항입니다.</p>
                  )}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {error && <p className="mt-6 rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">오류: {error}</p>}

      <button
        onClick={() => {
          if (missingKeys.length > 0) {
            setShowMissing(true);
            return;
          }
          onSubmit(answers);
        }}
        disabled={submitting}
        className="mx-auto my-10 rounded-full bg-stone-900 px-12 py-4 text-lg font-bold text-white transition-all hover:bg-black disabled:bg-stone-200 disabled:text-stone-400"
      >
        {submitting ? '제출 중...' : isPre ? '다음 단계로' : '결과 확인하기'}
      </button>
    </div>
  );
};

export default SurveyForm;
