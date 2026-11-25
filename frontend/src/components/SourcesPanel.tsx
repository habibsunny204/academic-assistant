import { useEffect, type CSSProperties } from "react";
import { citationHref } from "../api";
import { categoryStripe, formatDate, formatDateTime } from "../format";
import type { ConfidenceBand, SentenceSupport } from "../types";
import type { Turn } from "./MessageTurn";

interface Props {
  turn: Turn | null;
  highlight: number | null;
}

const BAND: Record<ConfidenceBand, { label: string; tone: string }> = {
  high: { label: "High", tone: "good" },
  medium: { label: "Medium", tone: "warn" },
  low: { label: "Low", tone: "bad" },
  none: { label: "No answer", tone: "neutral" },
};

function factCheck(support: SentenceSupport[]): string {
  const checked = support.filter((s) => s.method === "terms");
  const verified = checked.filter((s) => s.supported).length;
  const unverifiable = support.length - checked.length;
  const parts = [];
  if (checked.length) parts.push(`${verified}/${checked.length} verified`);
  if (checked.length - verified) parts.push(`${checked.length - verified} flagged`);
  if (unverifiable) parts.push(`${unverifiable} not checkable`);
  return parts.join(" · ");
}

export function SourcesPanel({ turn, highlight }: Props) {
  const response = turn?.response ?? null;

  useEffect(() => {
    if (highlight == null) return;
    document.getElementById(`source-${highlight}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [highlight, turn?.key]);

  const details = response?.details;
  const band = details ? BAND[details.confidence_band] : null;

  return (
    <aside className="side" aria-label="Sources and answer details">
      <section className="card">
        <div className="card-head">
          <h2 className="card-title">Sources &amp; Citations</h2>
        </div>
        {!response && <p className="side-empty">Ask a question. The documents behind each answer appear here.</p>}
        {response && response.citations.length === 0 && (
          <p className="side-empty">
            {response.abstained
              ? "No sources: the assistant declined to answer because no indexed document covers this question."
              : "This answer cites no source. Treat it as unverified."}
          </p>
        )}
        {response && response.citations.length > 0 && (
          <div className="sources">
            {response.citations.map((c) => (
              <a
                key={c.n}
                id={`source-${c.n}`}
                className={`source-card${highlight === c.n ? " highlight" : ""}`}
                style={{ "--stripe": categoryStripe(c.category) } as CSSProperties}
                href={citationHref(c.url, c.file_type, c.page)}
                target="_blank"
                rel="noopener noreferrer"
              >
                <div className="source-title">
                  <span className="source-num">[{c.n}]</span>
                  {c.title}
                </div>
                <span className="type-badge">{c.file_type}</span>
                <div className="source-meta">
                  {c.page ? `Page ${c.page}` : c.section || "Document"}
                  {c.doc_date ? ` · ${formatDate(c.doc_date)}` : ""}
                  {c.category ? ` · ${c.category}` : ""}
                  <br />
                  Relevance: {c.relevance.toFixed(2)}
                </div>
              </a>
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2 className="card-title">Answer Details</h2>
        </div>
        {!details && <p className="side-empty">Details for the selected answer appear here.</p>}
        {details && band && (
          <dl className="details">
            <div>
              <dt>Model</dt>
              <dd>{details.llm_called ? details.model : "Not called"}</dd>
            </div>
            <div>
              <dt>Pipeline</dt>
              <dd>Hybrid RAG + reranking</dd>
            </div>
            <div>
              <dt>Response time</dt>
              <dd>{details.response_time_s.toFixed(2)} sec</dd>
            </div>
            <div>
              <dt>Retrieved chunks</dt>
              <dd>{details.retrieved_chunks}</dd>
            </div>
            <div>
              <dt>Top relevance</dt>
              <dd>{details.top_relevance.toFixed(2)}</dd>
            </div>
            {!response!.abstained && response!.support.length > 0 && (
              <div>
                <dt>Fact check</dt>
                <dd>{factCheck(response!.support)}</dd>
              </div>
            )}
            <div>
              <dt>Confidence score</dt>
              <dd>
                {response!.abstained ? "–" : details.confidence.toFixed(2)}{" "}
                <span className={`pill ${band.tone}`}>{band.label}</span>
              </dd>
            </div>
            <div>
              <dt>Date</dt>
              <dd>{formatDateTime(details.answered_at)}</dd>
            </div>
          </dl>
        )}
      </section>
    </aside>
  );
}
