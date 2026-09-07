import { useState } from 'react';
import { ApiError, submitEvaluation, submitSurvey } from '../../lib/api';
import AnswerForm from './AnswerForm';
import DebateRoom from './DebateRoom';
import ResultView from './ResultView';
import SurveyForm from './SurveyForm';

/**
 * 토론 전체 흐름:
 *   사전 설문 → 사전 서술 답변 → 토론(SSE) → 사후 서술 답변 → 사후 설문 → 채점·리포트
 *
 * 사전 설문은 세션 행이 생기기 전(=/debate/init 이전)에 받으므로 답변을 들고 있다가,
 * session 이벤트로 record_id 를 받는 즉시 저장한다.
 */
const DebatePage = ({ topic, initRequest, visible, onRestart }) => {
  const [step, setStep] = useState('pre_survey');
  const [preSurvey, setPreSurvey] = useState(null);
  const [preAnswers, setPreAnswers] = useState(null);
  const [postAnswers, setPostAnswers] = useState(null);
  const [postSurvey, setPostSurvey] = useState(null);
  const [preSurveySaved, setPreSurveySaved] = useState(false);
  const [surveyWarning, setSurveyWarning] = useState(null);
  const [recordId, setRecordId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [synthesis, setSynthesis] = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!visible) return null;

  const handleSessionStart = (data) => {
    const id = data.record_id ?? null;
    setRecordId(id);
    if (id != null && preSurvey) {
      // 실패해도 토론은 계속 — 사후 제출 때 한 번 더 시도한다.
      submitSurvey(id, 'pre', preSurvey)
        .then(() => setPreSurveySaved(true))
        .catch((e) => console.warn('사전 설문 저장 실패:', e));
    }
  };

  // 설문 저장이 실패해도 채점·리포트는 진행한다. 응답은 state 에 남아 결과 화면에서 재시도할 수 있다.
  const saveSurveys = async (postSurveyAnswers) => {
    if (recordId == null) return '세션 기록을 찾지 못해 설문이 저장되지 않았습니다.';
    try {
      if (!preSurveySaved && preSurvey) {
        await submitSurvey(recordId, 'pre', preSurvey);
        setPreSurveySaved(true);
      }
      await submitSurvey(recordId, 'post', postSurveyAnswers);
      return null;
    } catch (e) {
      return e instanceof ApiError ? String(e.detail ?? e.message) : String(e);
    }
  };

  const handlePostSurvey = async (answers) => {
    setSubmitting(true);
    setError(null);
    setPostSurvey(answers);
    const warning = await saveSurveys(answers);
    setSurveyWarning(warning);
    try {
      const result = await submitEvaluation({
        topicId: topic.id,
        recordId,
        prePro: preAnswers.pro,
        preCon: preAnswers.con,
        postPro: postAnswers.pro,
        postCon: postAnswers.con,
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
      {step === 'pre_survey' && (
        <SurveyForm
          phase="pre"
          mode={initRequest.mode}
          topicId={topic.id}
          topic={topic}
          submitting={false}
          error={null}
          onSubmit={(answers) => {
            setPreSurvey(answers);
            setStep('pre_answer');
          }}
        />
      )}

      {step === 'pre_answer' && (
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
          onSessionStart={handleSessionStart}
          onFinished={(id, synthesisDraft) => {
            setSessionId(id);
            setSynthesis(synthesisDraft || '');
            setStep('post_answer');
          }}
        />
      )}

      {step === 'post_answer' && (
        <AnswerForm
          topic={topic}
          phase="post"
          submitting={false}
          error={null}
          onSubmit={(answers) => {
            setPostAnswers(answers);
            setStep('post_survey');
          }}
        />
      )}

      {step === 'post_survey' && (
        <SurveyForm
          phase="post"
          mode={initRequest.mode}
          topicId={topic.id}
          topic={topic}
          submitting={submitting}
          error={error}
          onSubmit={handlePostSurvey}
        />
      )}

      {step === 'result' && evaluation && (
        <ResultView
          sessionId={sessionId}
          evaluation={evaluation}
          mode={initRequest.mode}
          synthesis={synthesis}
          surveyWarning={surveyWarning}
          onRetrySurvey={async () => {
            const warning = await saveSurveys(postSurvey);
            setSurveyWarning(warning);
            return warning;
          }}
          onRestart={onRestart}
        />
      )}
    </div>
  );
};

export default DebatePage;
