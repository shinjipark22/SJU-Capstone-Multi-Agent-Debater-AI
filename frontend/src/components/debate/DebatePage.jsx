import { useState } from 'react';
import { ApiError, submitEvaluation } from '../../lib/api';
import AnswerForm from './AnswerForm';
import DebateRoom from './DebateRoom';
import ResultView from './ResultView';

/**
 * 토론 전체 흐름: 사전 답변 → 토론(SSE) → 사후 답변 → 채점·리포트.
 * 사전/사후 답변과 채점 결과는 /evaluation 호출 시 백엔드 수집 테이블에 함께 저장된다.
 */
const DebatePage = ({ topic, initRequest, visible, onRestart }) => {
  const [step, setStep] = useState('pre');
  const [preAnswers, setPreAnswers] = useState(null);
  const [recordId, setRecordId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!visible) return null;

  const handlePostSubmit = async ({ pro, con }) => {
    setSubmitting(true);
    setError(null);
    try {
      const result = await submitEvaluation({
        topicId: topic.id,
        recordId,
        prePro: preAnswers.pro,
        preCon: preAnswers.con,
        postPro: pro,
        postCon: con,
      });
      setEvaluation(result);
      setStep('result');
    } catch (e) {
      setError(e instanceof ApiError ? String(e.detail ?? e.message) : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="absolute inset-0 bg-[#F5F5F4]">
      {step === 'pre' && (
        <AnswerForm
          topic={topic}
          phase="pre"
          submitting={false}
          error={null}
          onSubmit={(answers) => {
            setPreAnswers(answers);
            setStep('debate');
          }}
        />
      )}

      {step === 'debate' && (
        <DebateRoom
          initRequest={initRequest}
          topic={topic}
          onSessionStart={(data) => setRecordId(data.record_id ?? null)}
          onFinished={(id) => {
            setSessionId(id);
            setStep('post');
          }}
        />
      )}

      {step === 'post' && (
        <AnswerForm
          topic={topic}
          phase="post"
          submitting={submitting}
          error={error}
          onSubmit={handlePostSubmit}
        />
      )}

      {step === 'result' && evaluation && (
        <ResultView sessionId={sessionId} evaluation={evaluation} onRestart={onRestart} />
      )}
    </div>
  );
};

export default DebatePage;
