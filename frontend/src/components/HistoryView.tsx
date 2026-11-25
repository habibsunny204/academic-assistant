import { AlertTriangle, LoaderCircle, MessageCircle, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { formatDateTime } from "../format";
import type { ConversationSummary } from "../types";

interface Props {
  activeId: string | null;
  onOpen: (id: string) => void;
  onDeleted: (id: string) => void;
}

export function HistoryView({ activeId, onOpen, onDeleted }: Props) {
  const [items, setItems] = useState<ConversationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .conversations()
      .then(setItems)
      .catch((e) => setError(e.message));
  }, []);

  const remove = async (item: ConversationSummary) => {
    if (!window.confirm(`Delete "${item.title}"? This cannot be undone.`)) return;
    try {
      await api.deleteConversation(item.id);
      setItems((prev) => prev?.filter((i) => i.id !== item.id) ?? null);
      onDeleted(item.id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="view">
      <div style={{ maxWidth: 820 }}>
        <h1 className="page-title">Chat History</h1>
        <p className="page-sub">Earlier conversations on this server, most recent first.</p>
        {error && (
          <div className="notice error" style={{ marginBottom: 12 }}>
            <AlertTriangle size={18} />
            <div>{error}</div>
          </div>
        )}
        {!items && !error && (
          <div className="typing">
            <LoaderCircle size={18} className="spin" /> Loading…
          </div>
        )}
        {items?.length === 0 && <p className="muted">No conversations yet.</p>}
        <div className="doc-list">
          {items?.map((item) => (
            <div key={item.id} className="card history-item">
              <MessageCircle size={18} color="var(--text-3)" />
              <button type="button" className="history-open" onClick={() => onOpen(item.id)}>
                <div className="doc-title">{item.title}</div>
                <div className="source-meta">
                  {item.n_messages} {item.n_messages === 1 ? "question" : "questions"} ·{" "}
                  {formatDateTime(item.updated_at)}
                  {item.id === activeId && " · open now"}
                </div>
              </button>
              <button
                type="button"
                className="icon-btn"
                aria-label={`Delete ${item.title}`}
                title="Delete"
                onClick={() => void remove(item)}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
