import { AlertTriangle, CircleCheck, FileUp, LoaderCircle, TriangleAlert } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { api, ApiError } from "../api";
import { uid } from "../format";
import type { UploadResult } from "../types";

const ACCEPT = ".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp";

interface Item {
  id: string;
  name: string;
  state: "queued" | "uploading" | "done" | "error";
  result?: UploadResult;
  error?: ApiError;
}

interface Props {
  onIndexed: () => void;
  onOpenSettings: () => void;
}

export function UploadView({ onIndexed, onOpenSettings }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [replace, setReplace] = useState(false);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const busy = items.some((i) => i.state === "uploading" || i.state === "queued");

  const update = (id: string, patch: Partial<Item>) =>
    setItems((prev) => prev.map((i) => (i.id === id ? { ...i, ...patch } : i)));

  const start = async (files: FileList | File[]) => {
    const list = Array.from(files);
    if (!list.length) return;
    const queued = list.map((f) => ({ id: uid(), name: f.name, state: "queued" as const }));
    setItems((prev) => [...queued, ...prev]);

    // One at a time: the server indexes each upload before accepting the next.
    let changed = false;
    for (const [index, file] of list.entries()) {
      const id = queued[index].id;
      update(id, { state: "uploading" });
      try {
        const result = await api.upload(file, replace);
        update(id, { state: "done", result });
        changed = true;
      } catch (e) {
        update(id, { state: "error", error: e as ApiError });
      }
    }
    if (changed) onIndexed();
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (!busy) void start(e.dataTransfer.files);
  };

  return (
    <div className="view">
      <div style={{ maxWidth: 760 }}>
        <h1 className="page-title">Upload Document</h1>
        <p className="page-sub">
          Add a notice, policy, routine or form. It is read (with OCR for scans), indexed and available to answers as
          soon as the upload finishes. Documents that cannot be read are quarantined, not used.
        </p>

        <div
          className={`dropzone${over ? " over" : ""}`}
          role="button"
          tabIndex={0}
          aria-disabled={busy}
          onClick={() => !busy && input.current?.click()}
          onKeyDown={(e) => {
            if ((e.key === "Enter" || e.key === " ") && !busy) {
              e.preventDefault();
              input.current?.click();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
        >
          <FileUp size={34} />
          <strong>{busy ? "Indexing…" : "Drop files here or click to choose"}</strong>
          <span>PDF, DOCX, images (JPG, PNG, TIFF…), TXT or Markdown</span>
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) void start(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        <label className="check">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          Replace a file that was uploaded before with the same name
        </label>

        <div className="results">
          {items.map((item) => (
            <div key={item.id} className="card result">
              {(item.state === "queued" || item.state === "uploading") && (
                <LoaderCircle size={18} className="spin" />
              )}
              {item.state === "done" && item.result?.status === "indexed" && (
                <CircleCheck size={18} color="var(--good)" />
              )}
              {item.state === "done" && item.result?.status !== "indexed" && (
                <TriangleAlert size={18} color="var(--warn)" />
              )}
              {item.state === "error" && <AlertTriangle size={18} color="var(--bad)" />}
              <div style={{ minWidth: 0 }}>
                <strong style={{ overflowWrap: "anywhere" }}>{item.result?.title || item.name}</strong>
                <div className="source-meta">
                  {item.state === "queued" && "Waiting…"}
                  {item.state === "uploading" && "Uploading and indexing…"}
                  {item.result?.status === "indexed" &&
                    `Indexed: ${item.result.chunks} chunks · ${item.result.category}`}
                  {item.result && item.result.status !== "indexed" && `${item.result.status}: ${item.result.reason}`}
                  {item.error && item.error.message}
                  {item.error?.code === "admin_token_required" && (
                    <>
                      {" "}
                      <button type="button" className="btn" style={{ padding: "2px 8px" }} onClick={onOpenSettings}>
                        Open Settings
                      </button>
                    </>
                  )}
                </div>
                {item.result && item.result.warnings.length > 0 && (
                  <ul className="warnings">
                    {item.result.warnings.map((w) => (
                      <li key={w}>
                        <AlertTriangle size={14} /> {w}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
