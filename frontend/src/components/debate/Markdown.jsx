import ReactMarkdown from 'react-markdown';

// LLM 발화·안내문은 마크다운(### 제목, **강조**, 목록, 링크)으로 온다.
// typography 플러그인 없이 요소별로 최소한의 서식만 입힌다.
const COMPONENTS = {
  h1: ({ children }) => <h3 className="mt-3 mb-1 text-base font-bold first:mt-0">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-3 mb-1 text-base font-bold first:mt-0">{children}</h3>,
  h3: ({ children }) => <h3 className="mt-3 mb-1 text-[15px] font-bold first:mt-0">{children}</h3>,
  p: ({ children }) => <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-1.5 list-disc pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 list-decimal pl-5">{children}</ol>,
  li: ({ children }) => <li className="my-0.5">{children}</li>,
  strong: ({ children }) => <strong className="font-bold">{children}</strong>,
  hr: () => <hr className="my-3 border-current opacity-20" />,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
      {children}
    </a>
  ),
  code: ({ children }) => <code className="rounded bg-black/10 px-1 py-0.5 text-[13px]">{children}</code>,
};

const Markdown = ({ children }) => <ReactMarkdown components={COMPONENTS}>{children}</ReactMarkdown>;

export default Markdown;
