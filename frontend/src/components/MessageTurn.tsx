import { AlertTriangle, Check, Copy, FileText, Info, LoaderCircle, ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { ApiError, citationHref } from "../api";
import { formatDate, formatTime } from "../format";
import type { AskResponse } from "../types";
import { AnswerText } from "./AnswerText";
import { BotMark } from "./BotMark";

export interface Turn {
  key: string;
  question: string;
  askedAt: string;
  category: string | null;
  status: "pending" | "done" | "error";
  response?: AskResponse;
  error?: ApiError;
}

interface Props {
  turn: Turn;
  userInitial: string;
  selected: boolean;
  onSelect: () => void;
  onCite: (n: number) => void;
  onFeedback: (rating: "up" | "down") => void;
}

function copyText(response: AskResponse): string {
  const sources = response.citations
    .map((c) => `[${c.n}] ${c.title}${c.page ? `, page ${c.page}` : ""}${c.doc_date ? ` (${c.doc_date})` : ""}`)
    .join("\n");
  return sources ? `${response.answer}\n\nSources:\n${sources}` : response.answer;
}

export function MessageTurn({ turn, userInitial, selected, onSelect, onCite, onFeedback }: Props) {
  const [copied, setCopied] = useState(false);
  const response = turn.response;

  const copy = async () => {
    if (!response) return;
    try {
      await navigator.clipboard.writeText(copyText(response));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked; nothing useful to do */
    }
  };

  return (
    <article className="turn">
      <div className="msg user">
        <div className="avatar" aria-hidden>
          {userInitial}
        </div>
        <div>
          <div className="msg-head">
            <span className="msg-name">You</span>
            {turn.category && <span className="pill neutral">{turn.category}</span>}
            <time className="msg-time">{formatTime(turn.askedAt)}</time>
          </div>
          <div className="msg-body">{turn.question}</div>
        </div>
      </div>

      <div
        className={`msg assistant${selected ? " selected" : ""}`}
        onClick={response ? onSelect : undefined}
      >
        <div className="avatar bot" aria-hidden>
          <BotMark size={18} />
        </div>
        <div>
          <div className="msg-head">
            <span className="msg-name">DIU Assistant</span>
            {response && <time className="msg-time">{formatTime(response.details.answered_at)}</time>}
          </div>

          {turn.status === "pending" && (
            <div className="typing" role="status">
              <LoaderCircle size={18} className="spin" />
              Searching documents and checking sources…
            </div>
          )}

          {turn.status === "error" && turn.error && <ErrorNotice error={turn.error} />}

          {response && response.abstained && (
            <div className="notice">
              <Info size={18} />
              <div>
                <strong>No answer found in the documents</strong>
                <div>{response.answer}</div>
              </div>
            </div>
          )}

          {response && !response.abstained && (
            <>
              <AnswerText
                text={response.answer}
                support={response.support}
                citationNumbers={new Set(response.citations.map((c) => c.n))}
                onCite={(n) => {
                  onSelect();
                  onCite(n);
                }}
              />

              {response.citations.length > 0 && (
                <div className="source-box">
                  {response.citations.map((c) => (
                    <div className="source-line" key={c.n}>
                      <FileText size={15} />
                      <div>
                        <div>
                          <strong>Source {c.n}: </strong>
                          <a href={citationHref(c.url, c.file_type, c.page)} target="_blank" rel="noopener noreferrer">
                            {c.title}
                            {c.page ? ` – Page ${c.page}` : ""}
                          </a>
                        </div>
                        {c.doc_date && <div className="source-meta">Dated {formatDate(c.doc_date)}</div>}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {response.warnings.length > 0 && (
                <ul className="warnings">
                  {response.warnings.map((w) => (
                    <li key={w}>
                      <AlertTriangle size={14} />
                      {w}
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}

          {response && (
            <div className="msg-actions">
              <button
                type="button"
                className="icon-btn"
                aria-label="Helpful"
                title="Helpful"
                aria-pressed={response.feedback === "up"}
                onClick={(e) => {
                  e.stopPropagation();
                  onFeedback("up");
                }}
              >
                <ThumbsUp size={16} />
              </button>
              <button
                type="button"
                className="icon-btn"
                aria-label="Not helpful"
                title="Not helpful"
                aria-pressed={response.feedback === "down"}
                onClick={(e) => {
                  e.stopPropagation();
                  onFeedback("down");
                }}
              >
                <ThumbsDown size={16} />
              </button>
              <button
                type="button"
                className="icon-btn"
                aria-label="Copy answer with sources"
                title="Copy answer with sources"
                onClick={(e) => {
                  e.stopPropagation();
                  void copy();
                }}
              >
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>
          )}
        </div>
      </div>
    </article>
  );
}

function ErrorNotice({ error }: { error: ApiError }) {
  if (error.code === "llm_unavailable") {
    return (
      <div className="notice error">
        <AlertTriangle size={18} />
        <div>
          <strong>The answer model is not available</strong>
          <pre>{error.message}</pre>
          <div className="source-meta">
            {error.message.includes("GROQ_API_KEY")
              ? "Add the key to .env, then ask again."
              : "After changing config.yaml, restart the server (Ctrl+C, then python cli.py serve)."}
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="notice error">
      <AlertTriangle size={18} />
      <div>
        <strong>Could not answer</strong>
        <div>{error.message}</div>
      </div>
    </div>
  );
}
