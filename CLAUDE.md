# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A retrieval-augmented question-answering assistant for Daffodil International University (DIU). Source documents in `Sources/` (PDFs, scanned PDFs, DOCX, images, text) are indexed. Questions get answers built only from those documents, with a citation on every factual sentence. If no document answers the question, the assistant declines to answer. Trustworthiness is the main goal, so an honest refusal beats a fluent wrong answer. `README.md` explains the reasons behind each design choice. `System architecture.jpeg` and `UI.jpeg` are the student's original prototype mockups.

Status: ingestion, retrieval, reranking, grounded answering, the web UI (FastAPI + React, modelled on `UI.jpeg`) and the evaluation harness are done. The full end-to-end evaluation has not been run yet (it needs a Groq key). Still to build: query rewriting for follow-up questions and user authentication.

## Commands

Environment: Windows, Python 3.14. Everything runs from the project root through `cli.py`.

```powershell
pip install -r requirements.txt
python cli.py doctor                 # check Tesseract, OCR languages, embedding model, index signature
python cli.py ingest                 # incremental: skips unchanged files, replaces revised ones, prunes deleted ones
python cli.py ingest --force         # re-extract everything (needed after changing extraction/cleaning/chunking code)
python cli.py ingest --reset         # wipe and rebuild (needed after changing embedding.model or embedding.dim)
python cli.py ingest --path Sources\file.pdf
python cli.py search "query" [--mode hybrid|dense|sparse] [--category X] [--k N]   # retrieval only, no LLM
python cli.py ask "question" [--debug]      # full RAG; --debug shows the per-sentence grounding check
python cli.py calibrate [--file q.yaml]     # suggests retrieval.min_relevance (YAML keys: in_domain, out_of_domain)
python cli.py docs | stats | models
python cli.py serve [--host H] [--port P]  # API + built UI at http://127.0.0.1:8000, API docs at /docs
python cli.py eval-check                   # every eval question answerable from the current index?
python cli.py evaluate [--retrieval-only] [--split test|dev|all] [--no-baseline] [--limit N] [--delay S]
python cli.py eval-feedback               # rated web-UI answers -> eval/feedback_candidates.yaml

cd frontend; npm install; npm run build    # production UI into frontend/dist (served by `serve`)
cd frontend; npm run dev                   # UI on :5173 with hot reload, proxies /api to :8000 (run `serve` too)

python -m pytest tests/ -q
python -m pytest tests/test_pipeline.py::test_routine_state_carries_day_across_pages -q
```

- `ask` needs `GROQ_API_KEY` in a `.env` file at the project root (gitignored). `search` and all tests work without it. The tests in `tests/test_answer.py` swap in a fake LLM by setting `answerer._client`.
- `tests/test_evaluation.py` uses stubs and a hand-built registry (under a second). `tests/test_answer.py` loads the embedding model (about 70s). `tests/test_api.py` builds a temp corpus, indexes it with the real models and stubs only the LLM (about 60s). `tests/test_pipeline.py` finishes in under a second.
- Frontend: Vite 8, React 19, TypeScript 7 (`npm run build` runs `tsc --noEmit` first), lucide-react icons, plain CSS tokens in `src/styles.css` with light/dark themes. There is no frontend test suite; `tsc` is the check.
- Models download from Hugging Face on first use: `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L6-v2`. Ignore the HF symlink and PyMuPDF deprecation warnings in the output.
- Tesseract is installed at `C:/Program Files/Tesseract-OCR/`. Its traineddata (`eng`, `ben`, `osd`) lives in the project-local `models/tessdata/` (gitignored), which `TESSDATA_PREFIX` points to.
- All tunables live in `config.yaml`, loaded through `diu/config.py`. Relative paths there are resolved against the project root.

## Architecture

There are two pipelines. Ingestion is orchestrated by `diu/ingest.py` and answering by `diu/answer.py`. They share `Embedder`, `VectorStore` and `Registry`.

**Ingestion, per file:** sha256 → skip if the file is unchanged → `extract/dispatch.py` (chooses the extractor by file extension) → `clean.py` → quality gate → `chunk.py` → `embed.py` → `store.py` + `registry.py`. After all files are processed, the BM25 index is rebuilt from the registry.

**Query:** `retrieve.py` runs hybrid search (Chroma dense + BM25, fused with RRF, 30 candidates). `rerank.py` narrows those to the top 5 with a cross-encoder plus a small recency tie-break. `answer.py` then decides whether to answer, generates with `llm.py` (Groq/Llama), and verifies the result.

**Evaluation:** `diu/evaluation.py` + `eval/questions.yaml`. Three layers are scored separately: retrieval (dense / BM25 / hybrid / hybrid+rerank ranks of the evidence passage), the relevance floor, and end to end (RAG vs the same LLM with no documents). `evaluate` writes `eval/results/<time>/report.md`, `summary.json` and `items.jsonl`.

**Web layer:** `diu/api/app.py` (FastAPI routes) → `diu/api/service.py` (`AssistantService`) → the same `Answerer`/`Retriever`/`Registry` objects as the CLI. `diu/api/history.py` stores conversations, answers (the full JSON response) and thumbs up/down feedback in `data/app/app.sqlite`, separate from the registry so a re-index never loses them. `frontend/src/types.ts` mirrors the JSON from `service._serialise`/`service.documents`; change both together.

- **One process owns the index while serving.** `AssistantService` loads the models once and passes its own registry/store/embedder into `ingest()`/`ingest_file()` for uploads and re-scans, instead of opening a second Chroma client. A `ReadWriteLock` lets questions run concurrently but excludes them during index writes, because BM25 and Chroma are swapped in several steps.
- **API errors** always have the shape `{"detail": {"code", "message"}}`. The UI branches on `code` (`llm_unavailable` shows setup steps, `admin_token_required` points to Settings). Raise `ServiceError(status, code, message)` from the service.
- **Uploads** go to `Sources/uploads/<sanitised name>` and are indexed synchronously before the response. `POST /api/documents` and `POST /api/reindex` require `X-Admin-Token` only when `api.admin_token` is set. `GET /api/documents/{doc_id}/file` serves only files under `Sources/` (checked against the resolved path, not the registry's stored path).
- **The UI never renders model output as HTML.** `components/AnswerText.tsx` parses paragraphs, lists, `**bold**` and `[n]` into React nodes, and underlines sentences that the grounding check flagged as unsupported.

Key ideas that span several files:

- **Three stores, one source of truth.** SQLite (`data/registry/`) holds documents and chunk text. Chroma (`data/vectordb/`) holds vectors. `bm25.pkl` is always derived from the registry's chunk texts. Any write must update all of them. The pattern is `store.delete_document(doc_id)`, then upsert, then `registry.upsert_document(...)` **before** `registry.replace_chunks(...)`. That order matters because `chunks.doc_id` is a foreign key.
- **`doc_id` comes from the file path, not the content hash** (`registry.doc_id_for`). A revised file keeps its `doc_id`, so its old chunks get replaced rather than duplicated.
- **Embedding signature guard.** `Embedder.signature()` (`model@dim`) is stored in the registry's `index_meta`. `ingest` refuses to run when it doesn't match the config, unless `--reset` is passed. BGE models need `query_prefix` on queries only, never on documents, and `Embedder` handles that difference.
- **Citation metadata is captured at ingest time or never.** `DocMeta` (title, dates, semester, version, category) comes from `extract/meta.py`, which parses the filename and first-page text. It is copied into every chunk's metadata in `chunk._chunk_meta`, and ends up in `Chunk.citation()` and `answer.Citation`. The UI's source cards read these fields. `store._sanitize` coerces metadata values because Chroma only accepts str/int/float/bool (no None).
- **Context headers.** `chunk.py` puts `Title | Semester | Version > Section` in front of every chunk's text. The header is part of both the embedded text and the text the LLM sees. Don't strip it: facts like routine rows can't be retrieved without it.
- **Tables become fact sentences** (`extract/tables.py`). Class routines (Room/Course/Teacher triples under day and time-slot rows) turn into one sentence per class. `RoutineState` is a single object per PDF shared across pages, because a day's grid continues onto the next page without repeating its headers. A cell only becomes a fact if the course column matches `COURSE_LIKE_RE`. Rows that don't match are counted as rejected rather than guessed. Inside a routine document, tables that aren't routine grids become `routine_aux` Markdown tables. They never become `generic_row` sentences, because pairing cells against the repeating headers produced garbage.
- **OCR is per page.** In `extract/pdf.py`, any page whose text layer has fewer than `min_text_chars_per_page` characters is rasterised and OCR'd. The OCR backend (`extract/ocr.py`) reports a confidence score, and low confidence raises a warning.
- **Quarantine, never index garbage.** A document that yields fewer than `quality.min_doc_chars` gets `status='quarantined'`, a JSON record in `data/quarantine/`, and its vectors removed.
- **Layered abstention in `answer.py`:** (1) if the top rerank score is below `retrieval.min_relevance`, the LLM is never called; (2) the LLM can return the `INSUFFICIENT_CONTEXT` sentinel; (3) after generation, citations to passages that weren't retrieved are stripped, and each cited sentence is compared by cosine similarity with its cited chunk (`answer.min_support`). `min_relevance` (0.05) is on the reranker's score scale: MiniLM gives real questions 0.23-1.0 and junk under 0.01. `bge-reranker-base` was dropped because it took 20-30 s per question on CPU and centred junk at 0.50. Switching `retrieval.reranker_model` requires re-running `cli.py calibrate`, and tests must set scores relative to the configured floor rather than hard-coding them.
- **Grounding is checked on digit-bearing terms, not similarity** (`answer._check_support`, `textnorm.checkable_terms`). Every date, fee, time, course code, section and room in a cited sentence must appear in a cited passage or its source header (`Answerer._evidence_text`). Embedding similarity could not separate a sentence's own evidence from the wrong routine passage (0.53-0.81 vs 0.48-0.70), so it is only a fallback for sentences with no such terms; those are reported as `method="similarity"` ("not checkable" in the UI) and earn half credit in confidence. Don't reintroduce a similarity threshold as the main check.
- **`textnorm.normalize` is shared by verification and evaluation**, so answers are scored by the same rules they are checked with. `contains_term` treats `_` as part of a token (`67` must not match `67_b`).
- **BM25 tokenizer keeps underscores** (`store._TOKEN_RE = [a-z0-9_]+`), so section and teacher codes such as `67_B` and `FNT_4` stay single high-IDF tokens.

## Gotchas

- **Evaluation hygiene.** Tune on the `dev` split, report on `test`. Don't change retrieval or threshold settings because of test results. Gold answers must come from the source documents, not from the index (the routine answers were cross-checked with pdfplumber; the notice from its scanned image). A new answerable item needs `evidence` text that exists in the index, which `eval-check` enforces. Keep `reviewed: false` until a person confirms an item.
- **Measured on the test split (retrieval only):** BM25 alone (Hit@1 82%, MRR 0.867) beats hybrid+rerank (73%, 0.788); dense search misses course-code lookups. The floor stops only 7/13 unanswerable test questions: near-domain questions score ~0.99. `calibrate` on the 13-question dev split suggests 0.968, which would wrongly refuse 4/22 test answerables; the floor stays at 0.05.

- Titles come from the first heading-like line of page 1 unless it is letterhead (`extract/meta.LETTERHEAD_RE`), in which case the filename is used. Metadata changes only reach existing documents after `ingest --force`.
- There is no authentication and no query rewriting: every question is retrieved on its own, so a follow-up like "what about Sunday?" loses the earlier context.
- `.docx` has no page numbers, so every DOCX block is reported as page 1.
- Routine chunks hold up to 12 facts each (`chunking.table_rows_per_chunk`), about 400 tokens per rerank pair. Reranking cost grows with candidates × chunk length, so raise `retrieval.candidates` or chunk size only after re-timing on CPU.
- `bge-small-en-v1.5` only handles English. For Bangla documents, use `ocr.langs: eng+ben`, then switch `embedding.model` to `BAAI/bge-m3` with `dim: 1024` and run `ingest --reset`.
