"""Evaluation: does the assistant retrieve, answer, cite and refuse correctly?

The question set (eval/questions.yaml) holds two kinds of item:

* answerable   - a gold document, evidence strings that the right passage must
                 contain, and answer facts the final answer must state.
* unanswerable - nothing in the corpus answers it; the correct behaviour is to
                 refuse. Near-domain and adversarial questions ("Who teaches
                 CSE315 on Friday?") matter most here, because they look
                 answerable.

Three layers are measured separately, so a bad number points at a stage:

1. Retrieval (no LLM): is the evidence passage found, and how high is it
   ranked, for dense-only, BM25-only, hybrid and hybrid+rerank retrieval?
2. The relevance floor (no LLM): which questions it stops before generation.
3. End to end (needs the LLM): answer accuracy, refusals, citation
   correctness and grounding, against a closed-book baseline where the same
   model answers with no documents.

Answers are scored by facts, not by text overlap with a reference answer. An
LLM answer can be worded any way; what matters is whether it states
"17 September 2026", so each fact lists acceptable forms and both sides are
normalised (times, ordinals, citation markers, punctuation) before matching.

Every rate is reported with a Wilson 95% interval. With a few dozen questions
the intervals are wide, and the report says so rather than implying precision
it does not have.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from .registry import Registry
from .textnorm import contains_term, normalize

VALID_TYPES = {"answerable", "unanswerable"}
VALID_SPLITS = {"dev", "test"}
RETRIEVAL_VARIANTS = ("dense", "sparse", "hybrid", "hybrid+rerank")

CLOSED_BOOK_SYSTEM = (
    "You are an assistant for students and staff of Daffodil International University "
    "(DIU) in Bangladesh. Answer the question directly and briefly. If you do not know "
    "the answer, say \"I don't know\"."
)

# Phrases that mark a closed-book reply as a refusal rather than an answer.
REFUSAL_RE = re.compile(
    r"\bi\s+(?:do\s*n[o']?t|don'?t)\s+know\b|\bi'?m\s+not\s+sure\b|\bnot\s+sure\b|"
    r"\b(?:do\s+not|don'?t)\s+have\s+(?:any\s+|specific\s+|access\s+|the\s+|enough\s+)?"
    r"(?:information|details|data|access)\b|\bno\s+(?:specific\s+)?information\b|"
    r"\b(?:cannot|can'?t|unable\s+to)\s+(?:provide|find|answer|confirm|determine)\b",
    re.I,
)


class EvalSetError(ValueError):
    """The question file is malformed. `problems` lists every issue found."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    id: str
    split: str
    type: str
    question: str
    category: str = ""
    difficulty: str = ""
    gold_docs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    answer_facts: list[list[str]] = field(default_factory=list)
    reference_answer: str = ""
    reviewed: bool = False
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type == "answerable"


def load_eval_set(path: str | Path) -> tuple[list[EvalItem], dict[str, Any]]:
    """Parse and validate a question file. Raises EvalSetError listing all problems."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    problems: list[str] = []
    items: list[EvalItem] = []
    seen: set[str] = set()

    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise EvalSetError(["the file has no 'items' list"])

    for index, raw in enumerate(raw_items):
        where = f"item {index + 1}"
        if not isinstance(raw, dict):
            problems.append(f"{where}: not a mapping")
            continue
        item_id = str(raw.get("id") or "").strip()
        where = f"item '{item_id}'" if item_id else where
        if not item_id:
            problems.append(f"{where}: missing id")
        elif item_id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(item_id)

        kind = raw.get("type")
        split = raw.get("split")
        question = str(raw.get("question") or "").strip()
        if kind not in VALID_TYPES:
            problems.append(f"{where}: type must be one of {sorted(VALID_TYPES)}")
        if split not in VALID_SPLITS:
            problems.append(f"{where}: split must be one of {sorted(VALID_SPLITS)}")
        if not question:
            problems.append(f"{where}: missing question")

        facts: list[list[str]] = []
        for fact in raw.get("answer_facts") or []:
            options = [fact] if isinstance(fact, (str, int, float)) else list(fact or [])
            options = [str(o).strip() for o in options if str(o).strip()]
            if not options:
                problems.append(f"{where}: an answer fact has no acceptable forms")
            facts.append(options)

        gold_docs = [str(d) for d in raw.get("gold_docs") or []]
        evidence = [str(e) for e in raw.get("evidence") or []]
        if kind == "answerable":
            if not gold_docs:
                problems.append(f"{where}: answerable items need gold_docs")
            if not facts:
                problems.append(f"{where}: answerable items need answer_facts")
        if kind == "unanswerable" and (facts or evidence):
            problems.append(f"{where}: unanswerable items must not have answer_facts or evidence")

        items.append(EvalItem(
            id=item_id, split=str(split), type=str(kind), question=question,
            category=str(raw.get("category") or ""),
            difficulty=str(raw.get("difficulty") or ""),
            gold_docs=gold_docs, evidence=evidence, answer_facts=facts,
            reference_answer=str(raw.get("reference_answer") or ""),
            reviewed=bool(raw.get("reviewed", False)),
            notes=str(raw.get("notes") or ""),
        ))

    if problems:
        raise EvalSetError(problems)
    meta = {
        "path": str(path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest()[:16],
        "version": data.get("version"),
        "description": data.get("description", ""),
    }
    return items, meta


def select(items: Iterable[EvalItem], split: str = "all") -> list[EvalItem]:
    return [i for i in items if split == "all" or i.split == split]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def normalize_answer(text: str) -> str:
    """Canonical form for fact matching; shared with grounding verification."""
    return normalize(text)


def fact_present(normalized_answer: str, options: list[str]) -> bool:
    return any(contains_term(normalized_answer, option) for option in options)


def facts_found(answer: str, item: EvalItem) -> list[bool]:
    normalized = normalize_answer(answer)
    return [fact_present(normalized, options) for options in item.answer_facts]


def doc_matches(item: EvalItem, filename: str) -> bool:
    name = (filename or "").lower()
    return any(g.lower() in name for g in item.gold_docs)


def evidence_in(item: EvalItem, text: str) -> list[bool]:
    haystack = _squash(text or "")
    return [_squash(e) in haystack for e in item.evidence]


def is_relevant(item: EvalItem, text: str, filename: str) -> bool:
    """A passage counts if it holds any evidence string (or, when an item has
    none, simply comes from a gold document)."""
    if not doc_matches(item, filename):
        return False
    return any(evidence_in(item, text)) if item.evidence else True


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval: sound for small n and for rates near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson(successes, n)
    return {"k": successes, "n": n, "rate": (successes / n) if n else None,
            "ci95": [round(low, 4), round(high, 4)]}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


# ---------------------------------------------------------------------------
# Consistency check against the index
# ---------------------------------------------------------------------------

def check_eval_set(items: list[EvalItem], registry: Registry) -> tuple[list[str], list[str]]:
    """Confirm each answerable item is actually answerable from the index.

    Errors: a gold document is missing or not indexed, or an evidence string
    appears in none of its chunks (the question would be unanswerable, or the
    evidence was mistyped). Warnings: an answer fact appears nowhere in the gold
    document's text or metadata, which is legitimate only when the fact lives
    in metadata the LLM sees in source headers (e.g. an effective date).
    """
    errors: list[str] = []
    warnings: list[str] = []
    docs = registry.list_documents()
    chunks_by_doc: dict[str, list[str]] = {}
    for row in registry.all_chunks():
        chunks_by_doc.setdefault(row["doc_id"], []).append(row["text"])

    for item in items:
        if not item.answerable:
            continue
        gold = [d for d in docs if doc_matches(item, d["filename"])]
        if not gold:
            errors.append(f"{item.id}: no document matches gold_docs {item.gold_docs}")
            continue
        indexed = [d for d in gold if d["status"] == "indexed"]
        if not indexed:
            errors.append(f"{item.id}: gold document is not indexed ({gold[0]['status']})")
            continue
        texts = [t for d in indexed for t in chunks_by_doc.get(d["doc_id"], [])]
        for evidence, found in zip(item.evidence, (
            any(_squash(e) in _squash(t) for t in texts) for e in item.evidence
        )):
            if not found:
                errors.append(f"{item.id}: evidence not found in the gold document: {evidence!r}")
        corpus = normalize_answer(" ".join(texts) + " " + " ".join(
            " ".join(str(d.get(k) or "") for k in ("title", "semester", "version_label",
                                                     "issued_date", "effective_from"))
            for d in indexed
        ))
        for options in item.answer_facts:
            if not fact_present(corpus, options):
                warnings.append(f"{item.id}: none of {options} appears in the gold document text")
    return errors, warnings


# ---------------------------------------------------------------------------
# Layer 1 and 2: retrieval and the relevance floor
# ---------------------------------------------------------------------------

def _first_relevant_rank(item: EvalItem, hits: list[dict[str, Any]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        meta = hit.get("meta") or {}
        if is_relevant(item, hit.get("text") or "", meta.get("filename", "")):
            return rank
    return None


def _evidence_recall(item: EvalItem, hits: list[dict[str, Any]], k: int) -> float:
    if not item.evidence:
        found = any(doc_matches(item, (h.get("meta") or {}).get("filename", "")) for h in hits[:k])
        return 1.0 if found else 0.0
    covered = [False] * len(item.evidence)
    for hit in hits[:k]:
        meta = hit.get("meta") or {}
        if not doc_matches(item, meta.get("filename", "")):
            continue
        for i, present in enumerate(evidence_in(item, hit.get("text") or "")):
            covered[i] = covered[i] or present
    return sum(covered) / len(covered)


def run_retrieval(
    items: list[EvalItem],
    retriever,
    registry: Registry,
    progress: Callable[[int, int, EvalItem], None] | None = None,
) -> list[dict[str, Any]]:
    """Rank of the evidence passage for each retrieval variant, per item."""
    candidates = retriever.candidates
    min_relevance = float(retriever.cfg.retrieval.get("min_relevance", 0.0))
    records = []
    for index, item in enumerate(items, start=1):
        if progress:
            progress(index, len(items), item)
        record: dict[str, Any] = {"id": item.id, "split": item.split, "type": item.type,
                                  "question": item.question, "variants": {}}

        started = time.perf_counter()
        vector = retriever.embedder.encode_query(item.question)
        embed_s = time.perf_counter() - started

        runs: dict[str, tuple[list[dict[str, Any]], float]] = {}
        t = time.perf_counter()
        runs["dense"] = (retriever.store.dense_search(vector, candidates), time.perf_counter() - t + embed_s)

        t = time.perf_counter()
        sparse = retriever.store.sparse_search(item.question, candidates)
        lookup = registry.get_chunks([h["chunk_id"] for h in sparse])
        for hit in sparse:
            row = lookup.get(hit["chunk_id"], {})
            hit["text"] = row.get("text", "")
            hit["meta"] = {"filename": row.get("filename", "")}
        runs["sparse"] = (sparse, time.perf_counter() - t)

        t = time.perf_counter()
        hybrid = retriever.store.hybrid_search(item.question, vector, k=candidates, candidates=candidates)
        runs["hybrid"] = (hybrid, time.perf_counter() - t + embed_s)

        t = time.perf_counter()
        reranked = retriever.reranker.rerank(item.question, [dict(h) for h in hybrid], top_k=candidates)
        runs["hybrid+rerank"] = (reranked, time.perf_counter() - t + runs["hybrid"][1])

        for name, (hits, seconds) in runs.items():
            entry: dict[str, Any] = {"latency_s": round(seconds, 3)}
            if item.answerable:
                rank = _first_relevant_rank(item, hits)
                doc_rank = next((r for r, h in enumerate(hits, start=1)
                                 if doc_matches(item, (h.get("meta") or {}).get("filename", ""))), None)
                entry.update({
                    "rank": rank,
                    "doc_rank": doc_rank,
                    "recall@5": _evidence_recall(item, hits, 5),
                })
            record["variants"][name] = entry

        top = reranked[0].get("rerank_score", 0.0) if reranked else 0.0
        record["top_relevance"] = round(float(top), 4)
        record["passes_floor"] = bool(reranked) and top >= min_relevance
        records.append(record)
    return records


def summarize_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in records if r["type"] == "answerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    variants: dict[str, Any] = {}
    for name in RETRIEVAL_VARIANTS:
        ranks = [r["variants"][name].get("rank") for r in answerable]
        n = len(ranks)

        def hit_at(k: int) -> dict[str, Any]:
            return rate(sum(1 for x in ranks if x is not None and x <= k), n)

        variants[name] = {
            "hit@1": hit_at(1),
            "hit@3": hit_at(3),
            "hit@5": hit_at(5),
            "recall@5": (statistics.mean(r["variants"][name]["recall@5"] for r in answerable)
                         if answerable else None),
            "mrr": (statistics.mean(1 / x if x else 0.0 for x in ranks) if n else None),
            "doc_hit@5": rate(sum(1 for r in answerable
                                  if (r["variants"][name].get("doc_rank") or 99) <= 5), n),
            "latency_p50_s": _percentile([r["variants"][name]["latency_s"] for r in records], 0.5),
        }
    return {
        "variants": variants,
        "floor": {
            "answerable_passed": rate(sum(r["passes_floor"] for r in answerable), len(answerable)),
            "unanswerable_stopped": rate(sum(not r["passes_floor"] for r in unanswerable),
                                         len(unanswerable)),
        },
    }


# ---------------------------------------------------------------------------
# Layer 3: end to end
# ---------------------------------------------------------------------------

def _outcome(item: EvalItem, answered: bool, found: list[bool]) -> str:
    if not item.answerable:
        return "answered" if answered else "refused"
    if not answered:
        return "refused"
    if found and all(found):
        return "correct"
    return "incomplete" if any(found) else "wrong"


def run_rag(
    items: list[EvalItem],
    answerer,
    registry: Registry,
    delay_s: float = 0.0,
    progress: Callable[[int, int, EvalItem], None] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(items, start=1):
        if progress:
            progress(index, len(items), item)
        started = time.perf_counter()
        answer = answerer.ask(item.question)
        elapsed = time.perf_counter() - started

        answered = not answer.abstained
        found = facts_found(answer.text, item) if answered else [False] * len(item.answer_facts)
        chunks = registry.get_chunks([c.chunk_id for c in answer.citations])

        cited_evidence = None
        doc_precision = None
        if answered and item.answerable and answer.citations:
            cited = [chunks.get(c.chunk_id, {}) for c in answer.citations]
            cited_evidence = any(is_relevant(item, row.get("text", ""), row.get("filename", ""))
                                 for row in cited)
            doc_precision = (sum(doc_matches(item, c.filename) for c in answer.citations)
                             / len(answer.citations))

        records.append({
            "id": item.id, "split": item.split, "type": item.type,
            "category": item.category, "difficulty": item.difficulty,
            "question": item.question, "reference_answer": item.reference_answer,
            "outcome": _outcome(item, answered, found),
            "answer": answer.text, "abstained": answer.abstained,
            "abstain_reason": answer.reason, "llm_called": answer.llm_called,
            "facts": [{"options": o, "found": f} for o, f in zip(item.answer_facts, found)],
            "citations": [{"n": c.n, "title": c.title, "filename": c.filename,
                           "page": c.page, "chunk_id": c.chunk_id} for c in answer.citations],
            "cited_evidence": cited_evidence,
            "citation_doc_precision": doc_precision,
            "cited_sentences": len(answer.support),
            "supported_sentences": sum(1 for s in answer.support if s.supported),
            # Only term checks verify anything; similarity-only sentences are unverifiable.
            "checked_sentences": sum(1 for s in answer.support if s.method == "terms"),
            "verified_sentences": sum(1 for s in answer.support
                                      if s.method == "terms" and s.supported),
            "flagged_terms": sorted({t for s in answer.support for t in s.missing_terms}),
            "uncited_answer": answered and not answer.citations,
            "warnings": answer.warnings,
            "confidence": answer.confidence,
            "top_relevance": answer.top_score,
            "latency_s": round(elapsed, 3),
        })
        if delay_s:
            time.sleep(delay_s)
    return records


def run_closed_book(
    items: list[EvalItem],
    client,
    delay_s: float = 0.0,
    progress: Callable[[int, int, EvalItem], None] | None = None,
) -> list[dict[str, Any]]:
    """The same LLM with no documents: what the poster calls "Base LLM"."""
    records = []
    for index, item in enumerate(items, start=1):
        if progress:
            progress(index, len(items), item)
        started = time.perf_counter()
        text = client.complete(CLOSED_BOOK_SYSTEM, item.question)
        elapsed = time.perf_counter() - started
        refused = is_refusal(text)
        found = facts_found(text, item)
        # A reply that hedges but still states the gold facts is an answer.
        answered = not refused or (item.answerable and bool(found) and all(found))
        records.append({
            "id": item.id, "split": item.split, "type": item.type,
            "category": item.category, "difficulty": item.difficulty,
            "question": item.question, "reference_answer": item.reference_answer,
            "outcome": _outcome(item, answered, found),
            "answer": text, "abstained": not answered,
            "facts": [{"options": o, "found": f} for o, f in zip(item.answer_facts, found)],
            "latency_s": round(elapsed, 3),
        })
        if delay_s:
            time.sleep(delay_s)
    return records


def summarize_answers(records: list[dict[str, Any]], with_citations: bool) -> dict[str, Any]:
    answerable = [r for r in records if r["type"] == "answerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    count = lambda rows, outcome: sum(1 for r in rows if r["outcome"] == outcome)  # noqa: E731

    answered = [r for r in records if not r["abstained"]]
    unreliable = (count(answerable, "wrong") + count(answerable, "incomplete")
                  + count(unanswerable, "answered"))
    fact_recalls = [
        sum(f["found"] for f in r["facts"]) / len(r["facts"]) for r in answerable if r["facts"]
    ]
    summary: dict[str, Any] = {
        "accuracy": rate(count(answerable, "correct"), len(answerable)),
        "fact_recall": statistics.mean(fact_recalls) if fact_recalls else None,
        "false_refusal": rate(count(answerable, "refused"), len(answerable)),
        "correct_refusal": rate(count(unanswerable, "refused"), len(unanswerable)),
        "answered_unanswerable": rate(count(unanswerable, "answered"), len(unanswerable)),
        "unreliable_answers": rate(unreliable, len(answered)),
        "overall_success": rate(count(answerable, "correct") + count(unanswerable, "refused"),
                                len(records)),
        "outcomes": {o: sum(1 for r in records if r["outcome"] == o)
                     for o in ("correct", "incomplete", "wrong", "refused", "answered")},
        "latency_p50_s": _percentile([r["latency_s"] for r in records], 0.5),
        "latency_p95_s": _percentile([r["latency_s"] for r in records], 0.95),
    }
    if with_citations:
        attempted = [r for r in answerable if not r["abstained"]]
        with_cites = [r for r in attempted if r["cited_evidence"] is not None]
        precisions = [r["citation_doc_precision"] for r in with_cites]
        summary.update({
            "citation_has_evidence": rate(sum(1 for r in with_cites if r["cited_evidence"]),
                                          len(attempted)),
            "citation_doc_precision": statistics.mean(precisions) if precisions else None,
            "verified_sentences": rate(sum(r["verified_sentences"] for r in records),
                                       sum(r["checked_sentences"] for r in records)),
            "unverifiable_sentences": rate(
                sum(r["cited_sentences"] - r["checked_sentences"] for r in records),
                sum(r["cited_sentences"] for r in records)),
            "uncited_answers": rate(sum(1 for r in answered if r["uncited_answer"]), len(answered)),
        })
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(value: dict[str, Any] | float | None, ci: bool = True) -> str:
    if value is None:
        return "–"
    if isinstance(value, dict):
        if not value["n"]:
            return "–"
        text = f"{value['rate'] * 100:.1f}%"
        if ci:
            low, high = value["ci95"]
            text += f" ({low * 100:.0f}–{high * 100:.0f})"
        return text
    return f"{value * 100:.1f}%"


def _frac(value: dict[str, Any]) -> str:
    return f"{value['k']}/{value['n']}" if value and value["n"] else "–"


def render_report(result: dict[str, Any]) -> str:
    meta = result["meta"]
    lines = [
        "# DIU Academic Assistant — evaluation report",
        "",
        f"Run: {meta['started_at']} · question set `{meta['eval_set']['path']}` "
        f"(sha256 {meta['eval_set']['sha256']}) · split **{meta['split']}**",
        "",
        f"- Questions: **{meta['n_items']}** ({meta['n_answerable']} answerable, "
        f"{meta['n_unanswerable']} unanswerable)",
        f"- Human-reviewed: **{meta['n_reviewed']} of {meta['n_items']}**"
        + ("" if meta["n_reviewed"] == meta["n_items"]
           else " — gold answers not yet reviewed by a person are provisional"),
        f"- Embedding `{meta['config']['embedding']}` · reranker `{meta['config']['reranker']}` · "
        f"relevance floor {meta['config']['min_relevance']} · answer model "
        f"`{meta['config']['answer_model'] if meta['mode'] == 'full' else 'not run'}`",
        f"- Documents indexed: {meta['index']['documents']} · chunks: {meta['index']['chunks']}",
        "",
        "Percentages show a Wilson 95% confidence interval in brackets. With this many "
        "questions the intervals are wide: differences whose intervals overlap may be noise.",
        "",
        "## 1. Retrieval (no LLM)",
        "",
        "Rank of the first passage containing the evidence, over answerable questions.",
        "",
        "| Retrieval | Hit@1 | Hit@3 | Hit@5 | Evidence recall@5 | MRR | Right document in top 5 | Median time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    retrieval = result["retrieval"]
    for name in RETRIEVAL_VARIANTS:
        v = retrieval["variants"][name]
        lines.append(
            f"| {name} | {_pct(v['hit@1'])} | {_pct(v['hit@3'])} | {_pct(v['hit@5'])} | "
            f"{_pct(v['recall@5'])} | {v['mrr']:.3f} | {_pct(v['doc_hit@5'], ci=False)} | "
            f"{v['latency_p50_s']:.2f}s |" if v["mrr"] is not None else f"| {name} | – | – | – | – | – | – | – |"
        )
    floor = retrieval["floor"]
    lines += [
        "",
        "## 2. Relevance floor (no LLM)",
        "",
        f"Questions whose best reranked passage scores below {meta['config']['min_relevance']} "
        "are refused without calling the model.",
        "",
        "| | Rate | Count |",
        "|---|---|---|",
        f"| Answerable questions let through | {_pct(floor['answerable_passed'])} | {_frac(floor['answerable_passed'])} |",
        f"| Unanswerable questions stopped | {_pct(floor['unanswerable_stopped'])} | {_frac(floor['unanswerable_stopped'])} |",
        "",
        "Unanswerable questions that pass the floor must be caught by the model's own "
        "refusal (`INSUFFICIENT_CONTEXT`), measured in section 3.",
        "",
    ]

    if meta["mode"] != "full":
        lines += ["## 3. End to end", "",
                  "Not run (`--retrieval-only`, or no LLM configured).", ""]
    else:
        systems = [("RAG (hybrid + rerank + verification)", result["rag"]["summary"])]
        if result.get("closed_book"):
            systems.insert(0, ("LLM only, no documents", result["closed_book"]["summary"]))
        lines += [
            "## 3. End to end",
            "",
            "| System | Accuracy | Wrong or unsupported answers | Correct refusals | False refusals | Overall success | Median time |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, s in systems:
            lines.append(
                f"| {label} | {_pct(s['accuracy'])} | {_pct(s['unreliable_answers'])} | "
                f"{_pct(s['correct_refusal'])} | {_pct(s['false_refusal'])} | "
                f"{_pct(s['overall_success'])} | {s['latency_p50_s']:.2f}s |"
            )
        rag = result["rag"]["summary"]
        lines += [
            "",
            "Citations (RAG only):",
            "",
            "| | Value |",
            "|---|---|",
            f"| Attempted answers whose citations include the evidence passage | {_pct(rag['citation_has_evidence'])} |",
            f"| Citations pointing at a gold document (mean precision) | {_pct(rag['citation_doc_precision'])} |",
            f"| Cited sentences whose dates, fees, times and codes were all found in the cited source | {_pct(rag['verified_sentences'])} |",
            f"| Cited sentences with no such terms (cannot be verified automatically) | {_pct(rag['unverifiable_sentences'])} |",
            f"| Answers with no citation | {_pct(rag['uncited_answers'])} |",
            "",
            "### Failures",
            "",
        ]
        failures = [r for r in result["rag"]["records"] if r["outcome"] not in ("correct", "refused")
                    or (r["type"] == "answerable" and r["outcome"] == "refused")]
        if not failures:
            lines.append("None.")
        for r in failures:
            missing = [f["options"][0] for f in r["facts"] if not f["found"]]
            lines += [
                f"- **{r['id']}** ({r['type']}, {r['outcome']}): {r['question']}",
                f"  - Expected: {r['reference_answer'] or 'a refusal'}"
                + (f" · missing: {', '.join(missing)}" if missing else ""),
                f"  - Got: {(r['answer'] or '').strip()[:300]}"
                + (f" · refusal reason: {r['abstain_reason']}" if r["abstained"] else ""),
            ]
    lines += [
        "",
        "## Definitions",
        "",
        "- **Hit@k** — the evidence passage is among the top k retrieved. **MRR** — mean of 1/rank "
        "of the first evidence passage (0 if not in the 30 candidates). **Evidence recall@5** — share "
        "of an item's evidence strings found in the top 5 (matters for questions needing two passages).",
        "- **Accuracy** — answerable questions answered with every gold fact stated. Facts list "
        "acceptable forms; times, dates, ordinals, punctuation and citation markers are normalised "
        "before matching.",
        "- **Wrong or unsupported answers** — of all answers given (not refusals): answers missing "
        "gold facts, plus answers to unanswerable questions. This is the closest honest equivalent "
        "of the poster's \"hallucination rate\".",
        "- **Correct refusals** — unanswerable questions the system declined. **False refusals** — "
        "answerable questions it declined.",
        "- **Overall success** — correct answers plus correct refusals, over all questions.",
        "- **LLM only** — the same model and question with no retrieved documents. A reply counts as "
        "a refusal when it says it does not know or lacks information.",
        "",
    ]
    return "\n".join(lines)


def write_results(result: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k not in ("rag", "closed_book")}
    summary["retrieval"] = {k: v for k, v in result["retrieval"].items() if k != "records"}
    for system in ("rag", "closed_book"):
        if result.get(system):
            summary[system] = result[system]["summary"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    with open(out / "items.jsonl", "w", encoding="utf-8") as fh:
        for record in result["retrieval"]["records"]:
            fh.write(json.dumps({"layer": "retrieval", **record}, default=str) + "\n")
        for system in ("rag", "closed_book"):
            for record in (result.get(system) or {}).get("records", []):
                fh.write(json.dumps({"layer": system, **record}, default=str) + "\n")

    (out / "report.md").write_text(render_report(result), encoding="utf-8")
    return out


def feedback_to_candidates(rated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn rated chat answers into draft eval items for a person to complete."""
    drafts = []
    for row in rated:
        response = row["response"]
        filenames = sorted({c.get("filename", "") for c in response.get("citations", []) if c.get("filename")})
        note = f"rated {row['rating']} in the web UI on {row['created_at']}"
        if row.get("comment"):
            note += f": {row['comment']}"
        drafts.append({
            "id": f"fb-{row['id'][:8]}",
            "split": "test",
            "type": "answerable" if not response.get("abstained") else "unanswerable",
            "category": row.get("category") or "",
            "question": row["question"],
            "gold_docs": filenames,
            "evidence": [],
            "answer_facts": [],
            "reference_answer": response.get("answer", ""),
            "reviewed": False,
            "notes": note + ". TODO: confirm the type, add evidence and answer_facts from the source.",
        })
    return drafts


def run_started_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "EvalItem", "EvalSetError", "load_eval_set", "select", "check_eval_set",
    "normalize_answer", "fact_present", "facts_found", "is_refusal", "is_relevant",
    "wilson", "rate", "run_retrieval", "summarize_retrieval", "run_rag",
    "run_closed_book", "summarize_answers", "render_report", "write_results",
    "feedback_to_candidates", "run_started_at",
]
