import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { uid } from "./format";
import { ChatView } from "./components/ChatView";
import { DashboardView } from "./components/DashboardView";
import { DocumentsView } from "./components/DocumentsView";
import { HistoryView } from "./components/HistoryView";
import type { Turn } from "./components/MessageTurn";
import { SettingsView, type ThemeChoice } from "./components/SettingsView";
import { Header, Sidebar, StatusBar } from "./components/Shell";
import { UploadView } from "./components/UploadView";
import type { Category, DocumentRow, Health, View } from "./types";

const THEME_KEY = "diu.theme";

function readTheme(): ThemeChoice {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readTheme);
  const [systemDark, setSystemDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") delete root.dataset.theme;
    else root.dataset.theme = choice;
    try {
      if (choice === "system") localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, choice);
    } catch {
      /* preference not remembered */
    }
  }, [choice]);

  const dark = choice === "dark" || (choice === "system" && systemDark);
  return { choice, setChoice, dark };
}

export default function App() {
  const theme = useTheme();
  const [view, setView] = useState<View>("chat");
  const [menuOpen, setMenuOpen] = useState(false);

  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const [docs, setDocs] = useState<DocumentRow[] | null>(null);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [docQuery, setDocQuery] = useState("");

  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [category, setCategory] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);

  const refreshHealth = useCallback(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setHealthError(null);
      })
      .catch((e: ApiError) => setHealthError(e.message));
  }, []);

  const refreshIndex = useCallback(() => {
    refreshHealth();
    api.categories().then(setCategories).catch(() => undefined);
    api
      .documents()
      .then((d) => {
        setDocs(d);
        setDocsError(null);
      })
      .catch((e: ApiError) => setDocsError(e.message));
  }, [refreshHealth]);

  useEffect(() => {
    refreshIndex();
    const timer = window.setInterval(refreshHealth, 30_000);
    return () => window.clearInterval(timer);
  }, [refreshIndex, refreshHealth]);

  const navigate = (next: View) => {
    setView(next);
    setMenuOpen(false);
  };

  const newChat = () => {
    setTurns([]);
    setConversationId(null);
    setCategory(null);
    setSelectedKey(null);
    setHighlight(null);
    navigate("chat");
  };

  const ask = async (question: string) => {
    const key = uid();
    setTurns((prev) => [
      ...prev,
      { key, question, askedAt: new Date().toISOString(), category, status: "pending" },
    ]);
    try {
      const response = await api.ask(question, category, conversationId);
      setConversationId(response.conversation_id);
      setTurns((prev) => prev.map((t) => (t.key === key ? { ...t, status: "done", response } : t)));
      setSelectedKey(key);
      setHighlight(null);
    } catch (e) {
      const error = e instanceof ApiError ? e : new ApiError(0, "unknown", String(e));
      setTurns((prev) => prev.map((t) => (t.key === key ? { ...t, status: "error", error } : t)));
      if (error.code === "llm_unavailable" || error.code.startsWith("index_")) refreshHealth();
    }
  };

  const feedback = async (key: string, rating: "up" | "down") => {
    const turn = turns.find((t) => t.key === key);
    if (!turn?.response) return;
    const previous = turn.response.feedback;
    const setRating = (value: "up" | "down" | null) =>
      setTurns((prev) =>
        prev.map((t) => (t.key === key && t.response ? { ...t, response: { ...t.response, feedback: value } } : t)),
      );
    setRating(rating);
    try {
      await api.feedback(turn.response.message_id, rating);
    } catch {
      setRating(previous);
    }
  };

  const openConversation = async (id: string) => {
    try {
      const convo = await api.conversation(id);
      const loaded: Turn[] = convo.messages.map((m) => ({
        key: m.message_id,
        question: m.question,
        askedAt: m.details.answered_at,
        category: m.category,
        status: "done",
        response: m,
      }));
      setTurns(loaded);
      setConversationId(convo.id);
      setCategory(loaded.at(-1)?.category ?? null);
      setSelectedKey(loaded.at(-1)?.key ?? null);
      setHighlight(null);
      navigate("chat");
    } catch {
      refreshHealth();
    }
  };

  let content;
  switch (view) {
    case "dashboard":
      content = (
        <DashboardView
          health={health}
          categories={categories}
          docs={docs}
          onOpenDocuments={() => navigate("documents")}
          onPickCategory={(name) => {
            setCategory(name);
            navigate("chat");
          }}
        />
      );
      break;
    case "history":
      content = (
        <HistoryView
          activeId={conversationId}
          onOpen={(id) => void openConversation(id)}
          onDeleted={(id) => {
            if (id === conversationId) newChat();
          }}
        />
      );
      break;
    case "upload":
      content = <UploadView onIndexed={refreshIndex} onOpenSettings={() => navigate("settings")} />;
      break;
    case "documents":
    case "notices":
      content = (
        <DocumentsView
          docs={docs}
          error={docsError}
          query={docQuery}
          onQuery={setDocQuery}
          noticesOnly={view === "notices"}
          onChanged={refreshIndex}
        />
      );
      break;
    case "settings":
      content = <SettingsView health={health} theme={theme.choice} onTheme={theme.setChoice} />;
      break;
    default:
      content = (
        <ChatView
          turns={turns}
          health={health}
          category={category}
          selectedKey={selectedKey}
          highlight={highlight}
          onAsk={(q) => void ask(q)}
          onClearCategory={() => setCategory(null)}
          onSelect={(key) => {
            if (key !== selectedKey) setHighlight(null);
            setSelectedKey(key);
          }}
          onCite={(key, n) => {
            setSelectedKey(key);
            setHighlight(n);
          }}
          onFeedback={(key, rating) => void feedback(key, rating)}
        />
      );
  }

  return (
    <div className="app">
      <Header
        darkActive={theme.dark}
        onToggleTheme={() => theme.setChoice(theme.dark ? "light" : "dark")}
        onMenu={() => setMenuOpen(true)}
        onHome={() => navigate("chat")}
        onSearch={(q) => {
          setDocQuery(q);
          navigate("documents");
        }}
      />
      <div className="shell">
        <Sidebar
          open={menuOpen}
          view={view}
          categories={categories}
          activeCategory={category}
          onNavigate={navigate}
          onNewChat={newChat}
          onCategory={(name) => {
            setCategory(name);
            navigate("chat");
          }}
        />
        {menuOpen && (
          <button type="button" className="nav-scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />
        )}
        <main style={{ minHeight: 0, display: "grid" }}>{content}</main>
      </div>
      <StatusBar health={health} error={healthError} />
    </div>
  );
}
