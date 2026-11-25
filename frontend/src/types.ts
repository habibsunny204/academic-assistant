// Mirrors the JSON produced by diu/api/service.py. Keep the two in step.

export interface Citation {
  n: number;
  doc_id: string;
  chunk_id: string;
  title: string;
  filename: string;
  file_type: string;
  page: number | null;
  section: string;
  doc_date: string | null;
  category: string;
  relevance: number;
  url: string | null;
}

export interface SentenceSupport {
  sentence: string;
  cited: number[];
  similarity: number;
  supported: boolean;
  /** "terms": dates, fees, times and codes were looked up in the cited source.
   *  "similarity": the sentence has none, so it could not really be verified. */
  method?: "terms" | "similarity";
  missing_terms?: string[];
}

export type ConfidenceBand = "high" | "medium" | "low" | "none";

export interface AnswerDetails {
  provider: string;
  model: string | null;
  llm_called: boolean;
  response_time_s: number;
  retrieved_chunks: number;
  confidence: number;
  confidence_band: ConfidenceBand;
  top_relevance: number;
  answered_at: string;
}

export interface AskResponse {
  message_id: string;
  conversation_id: string;
  question: string;
  category: string | null;
  answer: string;
  abstained: boolean;
  reason: string;
  citations: Citation[];
  support: SentenceSupport[];
  warnings: string[];
  details: AnswerDetails;
  feedback: "up" | "down" | null;
}

export interface Health {
  status: "ok" | "degraded";
  index: {
    state: "ready" | "empty" | "incompatible";
    message: string;
    documents: number;
    quarantined: number;
    chunks: number;
    pages: number;
    last_updated: string | null;
    embedding_model: string;
  };
  llm: { configured: boolean; provider: string; model: string; message: string };
  feedback: { up: number; down: number };
}

export interface Category {
  name: string;
  documents: number;
}

export interface DocumentRow {
  doc_id: string;
  title: string;
  filename: string;
  file_type: string;
  doc_type: string;
  category: string;
  semester: string | null;
  version_label: string | null;
  doc_date: string | null;
  status: "indexed" | "quarantined" | string;
  pages: number;
  chunks: number;
  ocr_pages: number[];
  warnings: string[];
  ingested_at: string | null;
  url: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  n_messages: number;
}

export interface Conversation extends ConversationSummary {
  messages: AskResponse[];
}

export interface UploadResult {
  filename: string;
  status: "indexed" | "unchanged" | "quarantined" | "failed" | string;
  title: string;
  category: string;
  chunks: number;
  reason: string;
  warnings: string[];
}

export interface ReindexResult {
  counts: Record<string, number>;
  pruned: string[];
  vectors: number;
  results: Array<Omit<UploadResult, "category">>;
}

export type View = "chat" | "dashboard" | "history" | "upload" | "documents" | "notices" | "settings";
