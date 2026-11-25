import type {
  AskResponse,
  Category,
  Conversation,
  ConversationSummary,
  DocumentRow,
  Health,
  ReindexResult,
  UploadResult,
} from "./types";

/** Error carrying the backend's stable `code`, so the UI can branch on it. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

const ADMIN_TOKEN_KEY = "diu.adminToken";

export function getAdminToken(): string {
  try {
    return localStorage.getItem(ADMIN_TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setAdminToken(token: string): void {
  try {
    if (token) localStorage.setItem(ADMIN_TOKEN_KEY, token);
    else localStorage.removeItem(ADMIN_TOKEN_KEY);
  } catch {
    /* storage unavailable: the token simply is not remembered */
  }
}

async function request<T>(path: string, init: RequestInit = {}, admin = false): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const token = admin ? getAdminToken() : "";
  if (token) headers.set("X-Admin-Token", token);

  let response: Response;
  try {
    response = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the assistant server. Is 'python cli.py serve' running?");
  }

  if (response.status === 204) return undefined as T;

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    /* non-JSON body */
  }

  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail) && "code" in detail) {
      const d = detail as { code: string; message: string };
      throw new ApiError(response.status, d.code, d.message);
    }
    if (Array.isArray(detail)) {
      // FastAPI request-validation errors
      const message = detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("; ");
      throw new ApiError(response.status, "invalid_request", message || "Invalid request.");
    }
    throw new ApiError(response.status, "http_error", `Request failed (${response.status}).`);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  categories: () => request<Category[]>("/api/categories"),
  ask: (question: string, category: string | null, conversationId: string | null) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question, category, conversation_id: conversationId }),
    }),
  conversations: () => request<ConversationSummary[]>("/api/conversations"),
  conversation: (id: string) => request<Conversation>(`/api/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) =>
    request<void>(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  feedback: (messageId: string, rating: "up" | "down") =>
    request<void>("/api/feedback", {
      method: "POST",
      body: JSON.stringify({ message_id: messageId, rating }),
    }),
  documents: () => request<DocumentRow[]>("/api/documents"),
  upload: (file: File, replace: boolean) => {
    const form = new FormData();
    form.append("file", file);
    form.append("replace", String(replace));
    return request<UploadResult>("/api/documents", { method: "POST", body: form }, true);
  },
  reindex: (force = false) =>
    request<ReindexResult>("/api/reindex", { method: "POST", body: JSON.stringify({ force }) }, true),
};

/** Open a citation at its page. Browsers' PDF viewers honour #page=N. */
export function citationHref(url: string | null, fileType: string, page: number | null): string | undefined {
  if (!url) return undefined;
  return fileType === "PDF" && page ? `${url}#page=${page}` : url;
}
