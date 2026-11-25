import {
  Bell,
  BookOpen,
  FileText,
  History,
  LayoutDashboard,
  Library,
  Menu,
  MessageCircle,
  MessageSquarePlus,
  Moon,
  Search,
  Settings,
  Sun,
  Upload,
} from "lucide-react";
import { useState } from "react";
import { formatCount, formatDate } from "../format";
import type { Category, Health, View } from "../types";
import { BotMark } from "./BotMark";

interface HeaderProps {
  darkActive: boolean;
  onToggleTheme: () => void;
  onMenu: () => void;
  onHome: () => void;
  onSearch: (q: string) => void;
}

export function Header({ darkActive, onToggleTheme, onMenu, onHome, onSearch }: HeaderProps) {
  const [q, setQ] = useState("");
  return (
    <header className="header">
      <button type="button" className="icon-btn menu-btn" aria-label="Open navigation" onClick={onMenu}>
        <Menu size={20} />
      </button>
      <button type="button" className="brand" onClick={onHome}>
        <span className="brand-mark" aria-hidden>
          <BotMark />
        </span>
        <span style={{ minWidth: 0 }}>
          <span className="brand-name" style={{ display: "block" }}>
            Daffodil International University (DIU)
          </span>
          <span className="brand-sub">Academic Assistant</span>
        </span>
      </button>
      <div className="header-spacer" />
      <form
        className="header-search"
        role="search"
        onSubmit={(e) => {
          e.preventDefault();
          onSearch(q);
        }}
      >
        <label htmlFor="global-search" className="sr-only">
          Search documents
        </label>
        <input
          id="global-search"
          type="search"
          placeholder="Search documents…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <Search size={16} />
      </form>
      <button
        type="button"
        className="icon-btn"
        aria-label={darkActive ? "Switch to light theme" : "Switch to dark theme"}
        title={darkActive ? "Light theme" : "Dark theme"}
        onClick={onToggleTheme}
      >
        {darkActive ? <Sun size={19} /> : <Moon size={19} />}
      </button>
    </header>
  );
}

interface SidebarProps {
  open: boolean;
  view: View;
  categories: Category[];
  activeCategory: string | null;
  onNavigate: (view: View) => void;
  onNewChat: () => void;
  onCategory: (name: string | null) => void;
}

const MAIN: Array<{ view: View; label: string; icon: typeof FileText }> = [
  { view: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { view: "chat", label: "Ask DIU Assistant", icon: MessageCircle },
  { view: "upload", label: "Upload Document", icon: Upload },
  { view: "documents", label: "Documents", icon: FileText },
  { view: "notices", label: "Notices", icon: Bell },
];

export function Sidebar({ open, view, categories, activeCategory, onNavigate, onNewChat, onCategory }: SidebarProps) {
  return (
    <nav className={`sidebar${open ? " open" : ""}`} aria-label="Main">
      <button type="button" className="new-chat" onClick={onNewChat}>
        <MessageSquarePlus size={18} /> New Chat
      </button>
      <button
        type="button"
        className="nav-item"
        aria-current={view === "history" ? "page" : undefined}
        onClick={() => onNavigate("history")}
      >
        <History size={18} /> Chat History
      </button>

      <div className="nav-label">Main</div>
      {MAIN.map(({ view: v, label, icon: Icon }) => (
        <button
          key={v}
          type="button"
          className="nav-item"
          aria-current={view === v ? "page" : undefined}
          onClick={() => onNavigate(v)}
        >
          <Icon size={18} /> {label}
        </button>
      ))}

      <div className="nav-label">Ask within a category</div>
      <button
        type="button"
        className="nav-item"
        aria-pressed={activeCategory === null && view === "chat"}
        onClick={() => onCategory(null)}
      >
        <Library size={18} /> All documents
      </button>
      {categories.map((c) => (
        <button
          key={c.name}
          type="button"
          className="nav-item"
          aria-pressed={activeCategory === c.name}
          onClick={() => onCategory(activeCategory === c.name ? null : c.name)}
        >
          <BookOpen size={18} /> {c.name}
          <span className="count">{c.documents}</span>
        </button>
      ))}

      <div className="sidebar-foot">
        <button
          type="button"
          className="nav-item"
          aria-current={view === "settings" ? "page" : undefined}
          onClick={() => onNavigate("settings")}
        >
          <Settings size={18} /> Settings
        </button>
      </div>
    </nav>
  );
}

export function StatusBar({ health, error }: { health: Health | null; error: string | null }) {
  let dot = "warn";
  let text = "Checking…";
  if (error) {
    dot = "bad";
    text = "Server unreachable";
  } else if (health) {
    if (health.status === "ok") {
      dot = "ok";
      text = "All systems operational";
    } else if (health.index.state !== "ready") {
      text = "Index needs attention";
    } else {
      text = "Answer model not configured";
    }
  }
  const index = health?.index;
  return (
    <footer className="statusbar">
      <span>
        <span className={`dot ${dot}`} />
        System status: <strong>{text}</strong>
      </span>
      <span className="grow">
        {index && (
          <>
            Knowledge base: <strong>{formatCount(index.chunks)} chunks</strong> ·{" "}
            <strong>{formatCount(index.documents)} documents</strong>
            {index.last_updated && <> · Last updated: {formatDate(index.last_updated)}</>}
          </>
        )}
      </span>
      <span className="powered">
        Powered by <strong>Hybrid RAG{health?.llm.configured ? ` + ${health.llm.model}` : ""}</strong>
      </span>
    </footer>
  );
}
