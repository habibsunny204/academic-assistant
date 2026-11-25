import { useState } from "react";
import { getAdminToken, setAdminToken } from "../api";
import { formatDateTime } from "../format";
import type { Health } from "../types";

export type ThemeChoice = "system" | "light" | "dark";

interface Props {
  health: Health | null;
  theme: ThemeChoice;
  onTheme: (t: ThemeChoice) => void;
}

export function SettingsView({ health, theme, onTheme }: Props) {
  const [token, setToken] = useState(getAdminToken());
  const [saved, setSaved] = useState(false);

  return (
    <div className="view">
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">System status and preferences for this browser.</p>

      <div className="settings">
        <section className="card">
          <h2>Knowledge base</h2>
          {!health && <p className="muted">Loading…</p>}
          {health && (
            <dl className="details" style={{ padding: 0 }}>
              <div>
                <dt>Index</dt>
                <dd>
                  <span className={`pill ${health.index.state === "ready" ? "good" : "warn"}`}>{health.index.state}</span>
                </dd>
              </div>
              {health.index.message && (
                <div>
                  <dt>Action needed</dt>
                  <dd>{health.index.message}</dd>
                </div>
              )}
              <div>
                <dt>Embedding model</dt>
                <dd>{health.index.embedding_model}</dd>
              </div>
              <div>
                <dt>Last updated</dt>
                <dd>{formatDateTime(health.index.last_updated) || "–"}</dd>
              </div>
            </dl>
          )}
        </section>

        <section className="card">
          <h2>Answer model</h2>
          {health && (
            <>
              <dl className="details" style={{ padding: 0 }}>
                <div>
                  <dt>Provider</dt>
                  <dd>{health.llm.provider}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{health.llm.model}</dd>
                </div>
                <div>
                  <dt>Status</dt>
                  <dd>
                    <span className={`pill ${health.llm.configured ? "good" : "warn"}`}>
                      {health.llm.configured ? "configured" : "not configured"}
                    </span>
                  </dd>
                </div>
              </dl>
              {!health.llm.configured && (
                <div className="notice" style={{ marginTop: 10 }}>
                  <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{health.llm.message}</pre>
                </div>
              )}
            </>
          )}
        </section>

        <section className="card">
          <h2>Admin token</h2>
          <p className="muted" style={{ margin: 0 }}>
            Needed for uploading and re-scanning only when the server sets <code>api.admin_token</code>. Stored in this
            browser.
          </p>
          <form
            className="field"
            onSubmit={(e) => {
              e.preventDefault();
              setAdminToken(token.trim());
              setSaved(true);
              setTimeout(() => setSaved(false), 1500);
            }}
          >
            <label htmlFor="admin-token" className="sr-only">
              Admin token
            </label>
            <input
              id="admin-token"
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Leave empty if the server does not require one"
            />
            <button type="submit" className="btn primary">
              {saved ? "Saved" : "Save"}
            </button>
          </form>
        </section>

        <section className="card">
          <h2>Appearance</h2>
          <div className="tabs" role="group" aria-label="Theme">
            {(["system", "light", "dark"] as ThemeChoice[]).map((t) => (
              <button key={t} type="button" aria-pressed={theme === t} onClick={() => onTheme(t)}>
                {t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
