import { AlertTriangle, CalendarDays, ExternalLink, FileText, FolderSync, Layers, LoaderCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { api, ApiError } from "../api";
import { formatDate, formatDateTime } from "../format";
import type { DocumentRow, ReindexResult } from "../types";

interface Props {
  docs: DocumentRow[] | null;
  error: string | null;
  query: string;
  onQuery: (q: string) => void;
  noticesOnly: boolean;
  onChanged: () => void;
}

type Filter = "all" | "indexed" | "quarantined";

export function DocumentsView({ docs, error, query, onQuery, noticesOnly, onChanged }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [reindexing, setReindexing] = useState(false);
  const [reindex, setReindex] = useState<{ result?: ReindexResult; error?: ApiError } | null>(null);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (docs ?? [])
      .filter((d) => !noticesOnly || d.doc_type === "notice")
      .filter((d) => filter === "all" || d.status === filter)
      .filter(
        (d) =>
          !q ||
          [d.title, d.filename, d.category, d.semester ?? "", d.doc_type].some((v) => v.toLowerCase().includes(q)),
      )
      .sort((a, b) => (b.doc_date ?? "").localeCompare(a.doc_date ?? ""));
  }, [docs, query, filter, noticesOnly]);

  const quarantined = (docs ?? []).filter((d) => d.status === "quarantined").length;

  const runReindex = async () => {
    setReindexing(true);
    setReindex(null);
    try {
      setReindex({ result: await api.reindex() });
      onChanged();
    } catch (e) {
      setReindex({ error: e as ApiError });
    } finally {
      setReindexing(false);
    }
  };

  return (
    <div className="view">
      <h1 className="page-title">{noticesOnly ? "Notices" : "Documents"}</h1>
      <p className="page-sub">
        {noticesOnly
          ? "Notices in the knowledge base, newest first."
          : "Everything in the knowledge base. Quarantined documents could not be read and are never used to answer."}
      </p>

      <div className="toolbar">
        <div className="header-search" style={{ width: "min(340px, 100%)" }}>
          <label htmlFor="doc-search" className="sr-only">
            Filter documents
          </label>
          <input
            id="doc-search"
            type="search"
            placeholder="Filter by title, category, semester…"
            value={query}
            onChange={(e) => onQuery(e.target.value)}
          />
        </div>
        <div className="tabs" role="group" aria-label="Status filter">
          {(["all", "indexed", "quarantined"] as Filter[]).map((f) => (
            <button key={f} type="button" aria-pressed={filter === f} onClick={() => setFilter(f)}>
              {f === "all" ? "All" : f === "indexed" ? "Indexed" : `Quarantined${quarantined ? ` (${quarantined})` : ""}`}
            </button>
          ))}
        </div>
        <div className="header-spacer" />
        {!noticesOnly && (
          <button type="button" className="btn" onClick={runReindex} disabled={reindexing}>
            {reindexing ? <LoaderCircle size={16} className="spin" /> : <FolderSync size={16} />}
            Re-scan Sources folder
          </button>
        )}
      </div>

      {reindex?.result && (
        <div className="notice" style={{ marginBottom: 14 }}>
          <FolderSync size={18} />
          <div>
            Re-scan finished:{" "}
            {Object.entries(reindex.result.counts)
              .map(([k, v]) => `${v} ${k}`)
              .join(", ") || "no files found"}
            {reindex.result.pruned.length > 0 && `, ${reindex.result.pruned.length} removed`}.
          </div>
        </div>
      )}
      {reindex?.error && (
        <div className="notice error" style={{ marginBottom: 14 }}>
          <AlertTriangle size={18} />
          <div>
            {reindex.error.message}
            {reindex.error.code === "admin_token_required" && " Add the admin token in Settings."}
          </div>
        </div>
      )}

      {error && (
        <div className="notice error">
          <AlertTriangle size={18} />
          <div>{error}</div>
        </div>
      )}
      {!docs && !error && (
        <div className="typing">
          <LoaderCircle size={18} className="spin" /> Loading documents…
        </div>
      )}
      {docs && rows.length === 0 && <p className="muted">No documents match.</p>}

      <div className="doc-list">
        {rows.map((d) => (
          <article key={d.doc_id} className="card doc">
            <div>
              <h2 className="doc-title">{d.title}</h2>
              <div className="doc-file">{d.filename}</div>
              <div className="doc-facts" style={{ marginTop: 8 }}>
                <span>{d.category}</span>
                {d.semester && <span>{d.semester}</span>}
                {d.doc_date && (
                  <span>
                    <CalendarDays size={14} /> {formatDate(d.doc_date)}
                  </span>
                )}
                <span>
                  <FileText size={14} /> {d.pages} {d.pages === 1 ? "page" : "pages"}
                  {d.ocr_pages.length > 0 && ` (${d.ocr_pages.length} OCR)`}
                </span>
                <span>
                  <Layers size={14} /> {d.chunks} {d.chunks === 1 ? "chunk" : "chunks"}
                </span>
              </div>
            </div>
            <div className="doc-side">
              <span className={`pill ${d.status === "indexed" ? "good" : "warn"}`}>{d.status}</span>
              <span className="type-badge">{d.file_type}</span>
              <a className="btn" href={d.url} target="_blank" rel="noopener noreferrer">
                Open <ExternalLink size={14} />
              </a>
            </div>
            {d.warnings.length > 0 && (
              <ul className="warnings">
                {d.warnings.map((w) => (
                  <li key={w}>
                    <AlertTriangle size={14} /> {w}
                  </li>
                ))}
              </ul>
            )}
            {d.ingested_at && (
              <div className="source-meta" style={{ gridColumn: "1 / -1" }}>
                Indexed {formatDateTime(d.ingested_at)}
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
