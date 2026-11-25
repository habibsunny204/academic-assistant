import { AlertTriangle, ChartBar, Table2 } from "lucide-react";
import { useRef, useState } from "react";
import { formatCount, formatDate } from "../format";
import type { Category, DocumentRow, Health } from "../types";

interface Props {
  health: Health | null;
  categories: Category[];
  docs: DocumentRow[] | null;
  onOpenDocuments: () => void;
  onPickCategory: (name: string) => void;
}

export function DashboardView({ health, categories, docs, onOpenDocuments, onPickCategory }: Props) {
  const [asTable, setAsTable] = useState(false);
  const [tip, setTip] = useState<{ name: string; value: number; share: number; top: number } | null>(null);
  const barsRef = useRef<HTMLDivElement>(null);

  const index = health?.index;
  const sorted = [...categories].sort((a, b) => b.documents - a.documents || a.name.localeCompare(b.name));
  const max = Math.max(1, ...sorted.map((c) => c.documents));
  const total = sorted.reduce((sum, c) => sum + c.documents, 0);
  const attention = (docs ?? []).filter((d) => d.status !== "indexed" || d.warnings.length > 0);
  const feedbackTotal = (health?.feedback.up ?? 0) + (health?.feedback.down ?? 0);

  const showTip = (el: HTMLElement, c: Category) => {
    const host = barsRef.current;
    if (!host) return;
    const top = el.getBoundingClientRect().top - host.getBoundingClientRect().top - 44;
    setTip({ name: c.name, value: c.documents, share: total ? c.documents / total : 0, top });
  };

  return (
    <div className="view">
      <h1 className="page-title">Dashboard</h1>
      <p className="page-sub">
        What the assistant can currently answer from
        {index?.last_updated ? `, last updated ${formatDate(index.last_updated)}` : ""}.
      </p>

      <div className="tiles">
        <div className="card tile">
          <div className="tile-label">Documents indexed</div>
          <div className="tile-value">{index ? formatCount(index.documents) : "–"}</div>
        </div>
        <div className="card tile">
          <div className="tile-label">Pages</div>
          <div className="tile-value">{index ? formatCount(index.pages) : "–"}</div>
        </div>
        <div className="card tile">
          <div className="tile-label">Searchable chunks</div>
          <div className="tile-value">{index ? formatCount(index.chunks) : "–"}</div>
        </div>
        <div className="card tile">
          <div className="tile-label">Quarantined</div>
          <div className="tile-value">{index ? formatCount(index.quarantined) : "–"}</div>
          <div className="tile-note">Unreadable, never used</div>
        </div>
        <div className="card tile">
          <div className="tile-label">Answers rated helpful</div>
          <div className="tile-value">
            {health ? (feedbackTotal ? `${Math.round((health.feedback.up / feedbackTotal) * 100)}%` : "–") : "–"}
          </div>
          <div className="tile-note">
            {health ? `${health.feedback.up} helpful · ${health.feedback.down} not helpful` : ""}
          </div>
        </div>
      </div>

      <section className="card chart-card" style={{ marginBottom: 20 }}>
        <div className="chart-head">
          <div>
            <h2 className="card-title">Documents by category</h2>
            <p className="card-sub">Indexed documents only. Select a category to ask questions within it.</p>
          </div>
          <div className="header-spacer" />
          <div className="tabs" role="group" aria-label="Chart or table">
            <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)} title="Chart">
              <ChartBar size={15} />
              <span className="sr-only">Chart view</span>
            </button>
            <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)} title="Table">
              <Table2 size={15} />
              <span className="sr-only">Table view</span>
            </button>
          </div>
        </div>

        {sorted.length === 0 && <p className="muted">No indexed documents yet.</p>}

        {sorted.length > 0 && !asTable && (
          <div className="bars" ref={barsRef} onMouseLeave={() => setTip(null)}>
            {sorted.map((c) => (
              <div
                key={c.name}
                className="bar-row"
                role="button"
                tabIndex={0}
                aria-label={`${c.name}: ${c.documents} documents. Ask within this category.`}
                onMouseEnter={(e) => showTip(e.currentTarget, c)}
                onFocus={(e) => showTip(e.currentTarget, c)}
                onBlur={() => setTip(null)}
                onClick={() => onPickCategory(c.name)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onPickCategory(c.name);
                  }
                }}
              >
                <span className="bar-label" title={c.name}>
                  {c.name}
                </span>
                <span className="bar-track">
                  <span className="bar" style={{ width: `calc((100% - 44px) * ${c.documents / max})` }} />
                  <span className="bar-value">{c.documents}</span>
                </span>
              </div>
            ))}
            {tip && (
              <div className="tooltip" style={{ top: Math.max(0, tip.top), left: 162 }} role="presentation">
                <div className="t-head">{tip.name}</div>
                <div className="t-row">
                  {tip.value} {tip.value === 1 ? "document" : "documents"} · {Math.round(tip.share * 100)}% of indexed
                </div>
              </div>
            )}
          </div>
        )}

        {sorted.length > 0 && asTable && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Category</th>
                <th className="num">Documents</th>
                <th className="num">Share</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td className="num">{c.documents}</td>
                  <td className="num">{total ? Math.round((c.documents / total) * 100) : 0}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card chart-card">
        <div className="chart-head">
          <div>
            <h2 className="card-title">Needs attention</h2>
            <p className="card-sub">Quarantined documents and documents indexed with warnings.</p>
          </div>
          <div className="header-spacer" />
          <button type="button" className="btn" onClick={onOpenDocuments}>
            All documents
          </button>
        </div>
        {docs && attention.length === 0 && <p className="muted">Nothing to review.</p>}
        <ul className="warnings" style={{ color: "var(--text)" }}>
          {attention.map((d) => (
            <li key={d.doc_id} style={{ padding: "6px 0" }}>
              <AlertTriangle size={14} color="var(--warn)" />
              <span>
                <strong>{d.title}</strong>
                <span className="muted"> — {d.status !== "indexed" ? d.status : ""}</span>{" "}
                <span className="muted">{d.warnings.join("; ")}</span>
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
