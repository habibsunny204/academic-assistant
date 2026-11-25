# DIU Academic Assistant — Cited, Document-Grounded QA

A cited, document-grounded question-answering assistant for Daffodil International
University students and staff.

**Parts 1–4 are complete and runnable**: ingestion, grounded answering, a web interface, and an evaluation harness with a verified question set. Any source format goes in — born-digital
PDF, scanned image, DOCX, photo, plain text — and a question comes back with an answer
grounded in those documents, each factual sentence carrying a citation to the document,
page and date it came from. When no indexed document answers the question, the
assistant says so instead of guessing.

---

## Quick start

```bash
pip install -r requirements.txt

# One-time: install the Tesseract binary (Windows)
winget install --id UB-Mannheim.TesseractOCR

python cli.py doctor                     # verify OCR, model, index
python cli.py ingest                     # index everything under Sources/
python cli.py ask "When is the last date for late registration?"

# Web interface
cd frontend && npm install && npm run build && cd ..
python cli.py serve                      # open http://127.0.0.1:8000
```

Answering needs a free Groq API key. Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Get one at [console.groq.com/keys](https://console.groq.com/keys) — no card required.
Retrieval (`search`) works without it; only `ask` needs a key.

Configuration lives in one file: `config.yaml`.

### All commands

| Command | What it does |
|---|---|
| `doctor` | Checks OCR, embedding model, index state |
| `ingest` | Indexes everything in `Sources/` |
| `ask "..."` | **Grounded answer with citations** |
| `search "..."` | Raw retrieval, no LLM — useful for debugging |
| `docs` | Lists indexed documents and warnings |
| `stats` | Corpus statistics |
| `calibrate` | Suggests an abstention threshold for your corpus |
| `models` | Lists models your provider currently serves |
| `serve` | Runs the web interface and HTTP API (API docs at `/docs`) |
| `eval-check` | Confirms every evaluation question is answerable from the current index |
| `evaluate` | Measures retrieval, refusals, accuracy and citations; writes a report |
| `eval-feedback` | Turns answers rated 👍/👎 in the web UI into draft evaluation questions |

---

## Web interface

`python cli.py serve` serves the React app built into `frontend/dist` together with the
API, as one process. The layout follows the prototype in `UI.jpeg`:

- **Ask DIU Assistant** — chat with clickable `[n]` citation chips. Clicking one
  highlights the matching card in the **Sources & Citations** panel; clicking a card opens
  the original document (PDFs open at the cited page). **Answer Details** shows the model,
  response time, retrieved chunks, top relevance and a confidence band.
- **Refusals look different from answers.** When no document covers a question, the reply
  says so in an amber notice and the sources panel explains why it is empty.
- **Flagged sentences are underlined**, with the date, fee, time or code that was not found
  in the cited source shown in the tooltip. **Answer Details** reports the fact check, e.g.
  *"2/2 verified · 1 not checkable"*.
- **Categories** in the sidebar come from the index and restrict retrieval to that
  category.
- **Upload Document** indexes a file immediately — it is answerable as soon as the upload
  finishes. Unreadable files are reported as quarantined.
- **Documents / Notices** list everything indexed, with status, OCR pages and warnings, and
  a *Re-scan Sources folder* button for files copied in by hand.
- **Dashboard** shows corpus size, quarantined documents, the share of answers rated
  helpful, and documents per category (chart or table).
- **Chat History** and 👍/👎 feedback are stored in `data/app/app.sqlite`. Every rated
  answer is stored with its exact citations and scores; `python cli.py eval-feedback`
  turns them into draft evaluation questions.
- Light and dark themes, and a phone layout with a slide-out menu.

For frontend development, run `python cli.py serve` and, in `frontend/`, `npm run dev`
(port 5173, API calls proxied to port 8000).

**Security:** the server listens on `127.0.0.1` only and has no user login. Before exposing
it on a network, set `api.admin_token` in `config.yaml`; uploads and re-scans then require
that token (enter it under *Settings* in the UI).

---

## Evaluation

```bash
python cli.py eval-check                    # is every question answerable from the index?
python cli.py evaluate --retrieval-only     # retrieval + relevance floor (no API key needed)
python cli.py evaluate                      # full run: RAG vs the same LLM with no documents
```

Each run writes `eval/results/<time>/report.md` (tables ready for a thesis, with 95%
confidence intervals), `summary.json`, and `items.jsonl` with every question's retrieved
ranks, answer, citations and flagged terms.

### The question set: `eval/questions.yaml`

48 questions over the three DIU documents: **30 answerable** and **18 unanswerable**.

- **Answerable** questions name the gold document, the exact **evidence** text the right
  passage must contain, and the **answer facts** a correct answer must state (with
  acceptable forms, e.g. `17 September 2026` / `September 17, 2026`). Answers are scored by
  facts, not by text overlap, because an LLM can phrase a correct answer any way.
- **Unanswerable** questions are where trustworthiness is tested. Beyond off-topic ones,
  most are **near-domain** (*"When is the DIU convocation in 2026?"*) or **adversarial**:
  a real course on a day it does not meet, a course that does not exist, next semester's
  fine, a deadline the form does not state.
- **Gold answers were checked against the original documents, not against the index.**
  Routine facts were cross-checked with pdfplumber, a different PDF parser from the
  pipeline's PyMuPDF (18/18 confirmed, including time slots by column position); notice
  facts were read from the scanned image rather than from OCR output. All items are still
  marked `reviewed: false` until someone at DIU confirms them, and the report says so.
- **dev / test split** (13 / 35). Tune on dev, report on test. Numbers measured on the
  questions used for tuning overstate how well a system works.

### What is measured

| Layer | Needs the LLM | Metrics |
|---|---|---|
| Retrieval: dense, BM25, hybrid, hybrid+rerank | no | Hit@1/3/5, evidence recall@5, MRR, right document in top 5, time |
| Relevance floor | no | answerable questions let through, unanswerable questions stopped |
| End to end: RAG vs **LLM only** | yes | accuracy, wrong-or-unsupported answers, correct and false refusals, citation evidence, verified sentences, latency |

"LLM only" asks the same model the same question with no documents: the *Base LLM* row of
the poster's evaluation table. *Wrong or unsupported answers* (among answers actually given)
is the honest equivalent of the poster's *hallucination rate*.

### Results so far (test split, retrieval only)

The full end-to-end run needs a Groq key and has not been run yet. Retrieval and the floor,
on the 22 answerable and 13 unanswerable test questions:

| Retrieval | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| dense only | 55% | 73% | 0.655 |
| **BM25 only** | **82%** | **95%** | **0.867** |
| hybrid (RRF) | 68% | 91% | 0.784 |
| hybrid + rerank (current pipeline) | 73% | 86% | 0.788 |

Relevance floor: 22/22 answerable questions let through, **7/13** unanswerable stopped.

**What this says**, with the caveat that 22 questions give wide intervals (Hit@1 of 73% is
52–87%):

1. **Keyword search is the strongest retriever for this corpus.** DIU questions are full of
   exact identifiers (`CSE316 66_F`, `KT-501(A)`) that an English embedding model does not
   understand: dense search did not rank rt-11's evidence in its top 30 at all. Equal-weight
   fusion then lets that weak ranking drag BM25's correct first result down, and the
   reranker only partly recovers it. The losses are all routine lookups by course code.
2. **The floor cannot stop near-domain questions.** "Late registration fine for Spring 2027"
   scores 0.99 against the Fall 2026 notice. Six of 13 unanswerable questions depend on the
   model's own refusal, which the full run will measure.
3. **Calibrating on 13 dev questions overfits.** `calibrate` on dev suggests a floor of 0.968;
   on test that would wrongly refuse 4 of 22 answerable questions (reg-02 scores 0.107). The
   floor stays at 0.05 until the dev split is larger.

None of these findings were used to change the retrieval settings, so the test numbers above
remain an honest measurement. Fixes should be tried on an enlarged dev split first (see
Known limits).

### Filling in the poster's evaluation table

```bash
python cli.py evaluate --split test --delay 2
```

Copy the *End to end* table from the report: **LLM only** replaces *Base LLM*, **RAG**
replaces *Base LLM + RAG*. The poster's IFT rows need a fine-tuned model and are not
measured here. Report the confidence intervals alongside the percentages.

---

## What the pipeline does

```
Sources/*.pdf .docx .jpg .png .txt
   │
   ├─ sha256 ─────────────► unchanged? skip (no re-embedding)
   │
   ├─ EXTRACT      per page: text layer │ tables │ OCR fallback
   ├─ CLEAN        encoding artefacts, running headers, page numbers
   ├─ QUALITY GATE too little text? → quarantine, never index
   ├─ CHUNK        structure-aware, every chunk gets a context header
   ├─ EMBED        bge-small-en-v1.5 (configurable)
   └─ INDEX        Chroma (dense) + BM25 (lexical) + SQLite (truth)
```

Retrieval fuses dense and lexical results with Reciprocal Rank Fusion. At query time:

```
question
   │
   ├─ HYBRID SEARCH   dense + BM25, fused by RRF  → 30 candidates
   ├─ RERANK          cross-encoder reads query+chunk together → top 5
   │                  (small recency bonus breaks near-ties)
   ├─ ABSTAIN?        below the relevance floor → never call the LLM
   ├─ GENERATE        Llama, constrained to the numbered passages
   └─ VERIFY          strip invalid citations, check each cited sentence
                      against the passage it claims
                                    ↓
                    answer + citations + confidence
```

---

## Design decisions, and why


### Chroma, not FAISS

Your architecture diagram listed "FAISS / Chroma". FAISS cannot delete vectors by
metadata. Since DIU documents are revised every semester, a revised routine would
leave its previous version in the index permanently — and the assistant would cite
a superseded document with full confidence. Chroma supports delete-by-metadata, so
re-ingesting a file **replaces** its chunks.

### Tables become sentences, not text

A class routine is a relational table. Extracted naively it reads:

```
KT-201 KT-201 CSE228(68_D) MIS KT-201 CSE315(67_B) AS KT-201 ...
```

No embedding model matches that against *"where is CSE315 on Saturday?"*. Each cell
is instead expanded into a self-contained fact:

> Course CSE315 section 67_B is scheduled on Saturday from 11:30-01:00 in room KT-201 taught by AS.

This one transformation is the difference between the routine being unusable and
being the most answerable document in the corpus. Your 15-page routine produced
**154 such facts**.

### Every chunk carries a context header

A fact like *"Course CSE315 ... on Saturday"* contains no evidence that it belongs to
the CSE routine for Fall 2026, so a question mentioning either term would never
retrieve it. Each chunk is therefore prefixed with its document, semester, version
and section path:

```
Class Routine for CSE Program | Fall 2026 | V1.1 > Saturday
```

The header is embedded *and* shown to the answer model, so the generated answer
inherits the context too.

### Quarantine instead of silent failure

Your `notice-for-late-registration-for-fall-2026.pdf` has a **zero-character text
layer** — it is a scan. A naive pipeline indexes it as empty and loses it without
any error. Here, every page with too little text is sent to OCR; if a document still
yields too little text, it is **quarantined** with a written reason in
`data/quarantine/` and never indexed.

This is the trustworthiness requirement applied at ingestion: an empty or garbled
chunk that reaches retrieval becomes a confident, well-formatted, **wrong** citation.

### Mis-aligned table columns are dropped, not guessed

The routine grid spans page boundaries — Saturday's header is on page 1 and its
remaining rooms continue on page 2 with no headers at all. Routine context is
carried across pages. As a safeguard, a cell is only emitted as a class fact if the
course column actually contains a course code (`[A-Z]{2,4}\d{3}`). If column mapping
drifts, the row is **rejected and counted**, not turned into a plausible-looking
wrong schedule. The count is surfaced as an ingestion warning.

### A fast reranker, chosen by measurement

The first reranker, `BAAI/bge-reranker-base`, took **20–30 seconds per question** on this
laptop's CPU, which is unusable in a chat interface. Four options were benchmarked on the
same 15 questions, 30 candidates each:

| Reranker | Median time | CSE315 67_B Saturday fact | Weakest real question | Strongest junk* |
|---|---|---|---|---|
| bge-reranker-base | 26.5 s | rank 2 | 0.545 | 0.502 |
| bge-reranker-base, 15 candidates | 10.8 s | rank 2 | 0.514 | 0.502 |
| **ms-marco-MiniLM-L6-v2** | **3.7 s** | **rank 1** | 0.168 | **0.006** |
| ms-marco-MiniLM-L12-v2 | 7.2 s | rank 1 | 0.299 | 0.030 |

\*excluding the hostel-fee question discussed below

`cross-encoder/ms-marco-MiniLM-L6-v2` is 7× faster, ranks the right chunk first, and separates
answerable from unanswerable questions far more clearly: bge's margin was 0.502 vs 0.545,
MiniLM's is two orders of magnitude. Answers in the web UI now take about 4–5 seconds plus
the LLM call. Both rerankers are English-only.

### Three layers of abstention, because one is not enough

After re-indexing, `python cli.py calibrate` on 7 answerable and 7 unanswerable questions:

| Threshold | Real questions missed | Junk reaching the LLM |
|---|---|---|
| 0.0005 | 0/7 | 2/7 |
| **0.05** (configured) | **0/7** | **1/7** |
| 0.47 | 1/7 | 1/7 |

The weakest real question scores 0.234 and six of seven junk questions score at most
0.002, so `min_relevance: 0.05` sits well inside the gap. But one question defeats any
threshold: *"What is the DIU hostel fee for 2026?"* scores **0.96**, higher than some real
questions, because it looks exactly like a DIU question and retrieval finds DIU-shaped
text. No score threshold can catch that.

The score scale belongs to the reranker, so changing `retrieval.reranker_model` means
re-running `calibrate`.

So there are three layers:

1. **Relevance floor** — obvious out-of-domain questions never reach the LLM.
2. **Model sentinel** — the LLM sees the passages and emits `INSUFFICIENT_CONTEXT`
   when they do not contain the answer. This is what catches the hostel-fee case.
3. **Post-generation verification** — see below.

Re-run `python cli.py calibrate` as the corpus grows; the right floor depends on
what the index actually contains.

### Verification: checking facts, not similarity

The failure that matters is not a model refusing to answer. It is a **fabricated fact
wearing a valid citation**: fluent, well-formatted, and wrong.

After generation, citations pointing at passages that were not retrieved are stripped.
Then every cited sentence is checked against the passages it cites.

The first version compared embeddings of the sentence and the passage. **The evaluation
in Part 4 showed that this does not work.** On the question set, a correct answer scored
0.53–0.81 against its own evidence and 0.48–0.70 against the *wrong* routine passage; at
the 0.35 threshold it accepted all 49 mismatched pairs, and no threshold separates the
two ranges. Every routine passage looks alike to an embedding model.

The check now works on **terms that carry a digit**: dates, fees, times, course codes,
sections and rooms. Each must appear (after normalising `8:30 AM to 10:00 AM` to
`8:30-10:00` and so on) in a cited passage or its source header. Measured on the same
pairs:

| Sentence contains | Correct sentence accepted | Mismatched source rejected |
|---|---|---|
| dates, fees, times, codes or rooms | 20/20 | **33/35** |
| none of these (similarity fallback) | 8/8 | 0/14 |

The two misses were a bare year and a room-and-time shared by another class. Sentences
without such terms **cannot be verified automatically**, and the system says so rather
than pretending: the web UI shows e.g. *"2/2 verified · 1 not checkable"*, flagged sentences
are underlined with the missing terms in the tooltip, and only verified sentences earn full
credit in the confidence score.

`python cli.py ask --debug` shows the check per sentence:

```
Result       | Check           | Cites | Not in source | Sentence
verified     | terms           | 1     |               | The last date for late registration is 17 Sep...
FLAGGED      | terms           | 1     | 700           | The late fine is Tk. 700/= ...
unverifiable | similarity 0.61 | 1     |               | The form must be approved by the department...
```

An answer that cites nothing is flagged too, and its confidence is cut by 60%.

### Idempotent, continuous ingestion

`doc_id` is derived from the file path, not its content. Re-running `ingest`:

| Situation | Behaviour |
|---|---|
| File unchanged | Skipped — no re-parsing, no re-embedding |
| File revised | Old chunks deleted from Chroma, SQLite **and** BM25; new ones indexed |
| File deleted | Pruned from the index (nothing cites a file nobody can open) |
| File unreadable | Quarantined with a reason |
| Embedding model changed | Ingestion **refuses to run** until `--reset` |

That last rule matters: vectors from two different models are not comparable, and
mixing them degrades retrieval in a way that produces no error message.

---

## Verified on your sample documents

| Document | Format | Result |
|---|---|---|
| CSE class routine, Fall 2026 | 15-page table PDF | 154 chunks; day/time/room/course/section/teacher recovered; `Effective From 2026-09-09`, `V1.1` |
| Late registration notice | **scanned image**, 0-char text layer | OCR'd; `Date: 12 September 2026` parsed; category `Registration` |
| Improvement examination form | table-heavy DOCX | 7 chunks; category `Examination`, semester `Summer 2026` |

Total: 3 documents, 17 pages, 162 chunks. `python -m pytest tests/ -q` → 13 passed.

Example — a student question answered from the scanned notice:

```
$ python cli.py search "What is the last date for late registration with a fine?"

1. Daffodil International University (DIU) | p.1 | Registration | 2026-09-12 | RRF 0.0328
   The last date for late registration is 17 September 2026 with a late fine of Tk. 500/=.
```

---

## Known limits

These are boundaries of the current stage, not defects:

1. **Retrieval under-uses keyword matches.** BM25 alone beats the hybrid pipeline on the
   test split (see Evaluation). Candidates to try, on an enlarged dev split: weighting BM25
   above dense in the fusion, a BM25-first path when the question contains a course code,
   and smaller routine chunks (`chunking.table_rows_per_chunk`).
2. **The question set is small.** 48 questions give wide confidence intervals, and 13 dev
   questions are too few to calibrate on. Grow both splits as documents are added;
   `eval-feedback` turns rated chat answers into drafts.
3. **Sentences without dates, fees, times or codes cannot be verified** automatically. They
   are shown as "not checkable" rather than "supported".
4. **No query rewriting yet.** Each question is retrieved on its own, so a follow-up such
   as *"what about Sunday?"* loses the context of the previous question.
5. **DOCX has no page numbers.** Citations for `.docx` fall back to the section path.
6. **English only.** `bge-small-en-v1.5` and the MiniLM reranker are English models. Bengali
   OCR is installed (`ocr.langs: eng+ben`), but Bangla retrieval needs `BAAI/bge-m3`
   (`dim: 1024`, re-run with `--reset`) and a multilingual reranker, which is much slower on CPU.
7. **Metadata heuristics are keyword-based.** Check `python cli.py docs` after each bulk
   ingestion; documents with no date are flagged there.
8. **No user accounts.** Chat history and feedback are shared by everyone using the server.
   Keep it on localhost or behind `api.admin_token` until authentication is added.

---

## A note on Instruction Fine-Tuning

Your poster pairs IFT with RAG. Worth deciding deliberately before Part 3:

Fine-tuning writes facts into model weights. Your corpus changes every semester, and
a model that memorised Fall 2025's registration deadline will state it confidently in
Fall 2026 — and no citation layer can detect that, because the claim never came from
a retrieved document.

Recommendation: keep the base model frozen for facts, and use IFT **only** to teach
behaviour — citation formatting, and saying *"I don't know"* when retrieval returns
nothing relevant. That preserves the accuracy gain in your evaluation table while
keeping every factual claim traceable to a document. Abstention is the feature most
worth adding to the architecture diagram; it is what separates a trustworthy
assistant from a fluent one.

---

## Roadmap

- **Part 1 — Ingestion & retrieval** ✅ *complete*
- **Part 2 — Reranking & grounded answering** ✅ *complete* — cross-encoder reranking,
  recency tie-breaking, three-layer abstention, citation validation, grounding verification
- **Part 3 — Interface** ✅ *complete* — FastAPI backend, React UI, uploads, history,
  feedback, dashboard
- **Part 4 — Evaluation** ✅ *complete*: verified question set with dev/test split,
  retrieval, floor and end-to-end metrics with confidence intervals, a closed-book baseline,
  and a fact-level grounding check that replaced the similarity check the evaluation disproved
- **Next:** run the full evaluation with a Groq key; improve keyword-heavy retrieval on an
  enlarged dev split; query rewriting for follow-up questions; user accounts

---

## Project layout

```
config.yaml              all tunable settings
cli.py                   doctor | ingest | ask | search | serve | evaluate | eval-check | eval-feedback | docs | stats | calibrate | models
diu/
  config.py              config loading
  models.py              Block, DocMeta, Chunk  (Chunk.citation() = the UI source card)
  registry.py            SQLite: documents, chunks, index signature
  clean.py               conservative text normalisation
  chunk.py               structure-aware chunking + context headers
  embed.py               embedding model wrapper
  store.py               Chroma + BM25 + RRF fusion
  ingest.py              orchestration, quality gate, pruning
  rerank.py              cross-encoder reranking + recency tie-break
  retrieve.py            query -> hybrid search -> rerank -> top-k
  answer.py              grounded generation, citations, abstention, verification
  llm.py                 LLM provider behind a thin interface
  textnorm.py            normalisation shared by grounding checks and evaluation
  evaluation.py          question sets, metrics, closed-book baseline, reports
  extract/
    dispatch.py          routes by file type
    pdf.py               text layer, tables, per-page OCR fallback
    docx_.py             preserves paragraph/table order
    image.py             standalone image OCR
    tables.py            routine grid -> class facts
    meta.py              titles, dates, semester, category
    ocr.py               pluggable OCR with confidence scoring
  api/
    app.py               FastAPI routes, serves the built UI
    service.py           AssistantService: models loaded once, index read/write lock
    history.py           conversations and feedback (data/app/app.sqlite)
eval/
  questions.yaml         48 verified questions (dev/test), and how to extend them
  results/               one folder per evaluation run: report.md, summary.json, items.jsonl
frontend/                React + TypeScript + Vite
  src/App.tsx            state and view routing
  src/components/        chat, sources panel, documents, upload, dashboard, history, settings
data/
  registry/              SQLite registry
  vectordb/              Chroma + BM25 index
  quarantine/            documents that could not be read, with reasons
  app/                   conversations and feedback
tests/
  test_pipeline.py       ingestion regression tests
  test_answer.py         answering guard-rail tests (LLM stubbed, no key needed)
  test_api.py            HTTP API tests on a temporary corpus
  test_evaluation.py     scoring rules, question-set validation, report
```
