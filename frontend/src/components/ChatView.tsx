import { Send, X } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Health } from "../types";
import { BotMark } from "./BotMark";
import { MessageTurn, type Turn } from "./MessageTurn";
import { SourcesPanel } from "./SourcesPanel";

interface Props {
  turns: Turn[];
  health: Health | null;
  category: string | null;
  selectedKey: string | null;
  highlight: number | null;
  onAsk: (question: string) => void;
  onClearCategory: () => void;
  onSelect: (key: string) => void;
  onCite: (key: string, n: number) => void;
  onFeedback: (key: string, rating: "up" | "down") => void;
}

const EXAMPLES = [
  "When is the last date for late registration?",
  "What is the late registration fine?",
  "Where is CSE315 section 67_B held on Saturday?",
  "How do I apply for an improvement exam?",
];

export function ChatView(props: Props) {
  const { turns, health, category, selectedKey, highlight } = props;
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const pending = turns.some((t) => t.status === "pending");
  const ready = health?.index.state === "ready";

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns.length, turns[turns.length - 1]?.status]);

  const submit = (text = draft) => {
    const question = text.trim();
    if (!question || pending) return;
    props.onAsk(question);
    setDraft("");
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const selected =
    turns.find((t) => t.key === selectedKey && t.response) ??
    [...turns].reverse().find((t) => t.response) ??
    null;

  let statusText = "Checking…";
  let statusClass = "online degraded";
  if (health) {
    if (health.status === "ok") {
      statusText = "Online";
      statusClass = "online";
    } else if (!ready) {
      statusText = "Index not ready";
    } else {
      statusText = "Answer model not configured";
    }
  }

  return (
    <div className="view view-chat">
      <section className="card chat" aria-label="Chat">
        <div className="card-head">
          <div>
            <h1 className="card-title">Chat with DIU Academic Assistant</h1>
            <p className="card-sub">Ask about DIU policies, courses, registration, examinations and notices.</p>
          </div>
          <span className={statusClass}>
            <span className={`dot ${health?.status === "ok" ? "ok" : "warn"}`} />
            {statusText}
          </span>
        </div>

        <div className="messages" ref={scrollRef} aria-live="polite">
          {turns.length === 0 ? (
            <div className="empty">
              <div>
                <div className="brand-mark" style={{ margin: "0 auto", width: 56, height: 56 }}>
                  <BotMark size={30} />
                </div>
                <h3>How can I help?</h3>
                <p>
                  Every answer is built only from indexed university documents and cites the page it came from. If no
                  document covers your question, I will say so instead of guessing.
                </p>
                {health && !ready && <p className="notice">{health.index.message}</p>}
                <div className="examples">
                  {EXAMPLES.map((q) => (
                    <button key={q} type="button" className="example" onClick={() => submit(q)} disabled={!ready}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            turns.map((turn) => (
              <MessageTurn
                key={turn.key}
                turn={turn}
                userInitial="Y"
                selected={turn.key === selected?.key}
                onSelect={() => props.onSelect(turn.key)}
                onCite={(n) => props.onCite(turn.key, n)}
                onFeedback={(rating) => props.onFeedback(turn.key, rating)}
              />
            ))
          )}
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          {category && (
            <div className="composer-filter">
              Searching only
              <span className="chip">
                {category}
                <button type="button" onClick={props.onClearCategory} aria-label={`Stop filtering by ${category}`}>
                  <X size={13} />
                </button>
              </span>
            </div>
          )}
          <div className="composer-row">
            <label htmlFor="question" className="sr-only">
              Your question
            </label>
            <textarea
              id="question"
              rows={1}
              value={draft}
              maxLength={1000}
              placeholder="Type your question here…"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKey}
            />
            <button type="submit" className="btn primary" disabled={!draft.trim() || pending}>
              Send <Send size={16} />
            </button>
          </div>
          <div className="composer-hint">Enter to send · Shift+Enter for a new line</div>
        </form>
      </section>

      <SourcesPanel turn={selected} highlight={selected?.key === selectedKey ? highlight : null} />
    </div>
  );
}
