"""Grounded answer generation with citations, abstention and verification.

Three mechanisms keep answers tied to documents, applied in order:

1. Abstention before generation. If reranking finds nothing above the relevance
   floor, the LLM is never called. A model handed irrelevant context will still
   produce a fluent answer, so the cheapest way to avoid a confident wrong
   answer is not to ask.

2. Constrained generation. The model sees only numbered retrieved passages, is
   required to cite one per factual sentence, and is told to emit a sentinel
   when the passages do not contain the answer.

3. Verification after generation. Citations pointing at passages that were not
   retrieved are stripped, and every cited sentence is checked for semantic
   support against the passage it claims. Unsupported sentences are flagged.

Step 3 is what the "Citation & Verification" box in the architecture diagram
does, and it is what makes a hallucination visible rather than invisible.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from .config import Config
from .llm import LLMUnavailable, get_client
from .retrieve import RetrievalResult, Retriever
from .textnorm import checkable_terms, contains_term, human_dates, normalize

ABSTAIN_SENTINEL = "INSUFFICIENT_CONTEXT"
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

# Citation styles models produce instead of plain [1]: gpt-oss writes 【1】 or
# 【1†L3-L5】, others use full-width ［1］ or grouped [1, 2].
_BRACKETED_RE = re.compile(r"[【［]\s*(\d{1,2})(?:\s*†[^】］]*)?\s*[】］]")
_GROUPED_RE = re.compile(r"\[\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})+)\s*\]")


def normalize_citations(text: str) -> str:
    """Rewrite other citation styles as [n] so they are parsed and verified."""
    text = _BRACKETED_RE.sub(lambda m: f"[{m.group(1)}]", text)
    return _GROUPED_RE.sub(
        lambda m: "".join(f"[{n.strip()}]" for n in re.split(r"[,;]", m.group(1))), text
    )


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

SYSTEM_PROMPT = """\
You are the DIU Academic Assistant for Daffodil International University.
You answer questions for students and staff using only official university documents.

RULES, in order of importance:

1. Use ONLY the numbered sources provided. Never use outside knowledge about DIU
   or any other university, and never infer a fact that is not written down.
2. If the sources do not contain the answer, reply with exactly this and nothing
   else: INSUFFICIENT_CONTEXT
3. End every factual sentence with the citation it came from, written exactly
   like [1] or [2][3]: plain square brackets and the source number only.
4. Copy dates, deadlines, fee amounts, room numbers, course codes and section
   labels exactly as written in the sources. Never round, reformat or estimate them.
5. If two sources disagree, say so explicitly, cite both, and point out which is
   more recent using the dates given.
6. Be direct and brief. Answer the question asked; do not add advice, filler, or
   invitations to contact anyone unless a source says so.
7. Write in the language the question was asked in.
"""

USER_TEMPLATE = """\
SOURCES
{sources}

QUESTION
{question}

Answer using only the sources above, citing each factual sentence."""


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    filename: str
    title: str
    page: int
    section: str
    source_path: str
    doc_date: str
    category: str
    score: float

    def label(self) -> str:
        parts = [self.title]
        if self.page:
            parts.append(f"page {self.page}")
        if self.doc_date:
            parts.append(self.doc_date)
        return " - ".join(str(p) for p in parts if p)


@dataclass
class SentenceSupport:
    sentence: str
    cited: list[int]
    similarity: float
    supported: bool
    # "terms": every digit-bearing term was looked up in the cited passages.
    # "similarity": the sentence has no such terms, so only embedding
    # similarity was available - a much weaker check.
    method: str = "similarity"
    missing_terms: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    support: list[SentenceSupport] = field(default_factory=list)
    confidence: float = 0.0
    top_score: float = 0.0
    n_retrieved: int = 0
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    # False when the answer was withheld before generation, so no model ran.
    llm_called: bool = False

    @property
    def confidence_band(self) -> str:
        if self.abstained:
            return "none"
        if self.confidence >= 0.70:
            return "high"
        if self.confidence >= 0.45:
            return "medium"
        return "low"


def _format_sources(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        meta = hit.get("meta") or {}
        header_bits = [meta.get("title", "Untitled")]
        if meta.get("page_start"):
            header_bits.append(f"page {meta['page_start']}")
        doc_date = meta.get("issued_date") or meta.get("effective_from") or ""
        if doc_date:
            header_bits.append(f"dated {doc_date}")
        if meta.get("semester"):
            header_bits.append(str(meta["semester"]))
        blocks.append(
            f"[{index}] ({' | '.join(str(b) for b in header_bits)})\n"
            f"{(hit.get('text') or '').strip()}"
        )
    return "\n\n".join(blocks)


def _build_citations(hits: list[dict[str, Any]], used: set[int]) -> list[Citation]:
    citations = []
    for index, hit in enumerate(hits, start=1):
        if index not in used:
            continue
        meta = hit.get("meta") or {}
        citations.append(
            Citation(
                n=index,
                chunk_id=hit.get("chunk_id", ""),
                doc_id=str(meta.get("doc_id", "")),
                filename=str(meta.get("filename", "")),
                title=str(meta.get("title", "Untitled")),
                page=int(meta.get("page_start") or 0),
                section=str(meta.get("section_path", "")),
                source_path=str(meta.get("source_path", "")),
                doc_date=str(
                    meta.get("issued_date") or meta.get("effective_from") or ""
                ),
                category=str(meta.get("category", "")),
                score=float(hit.get("rerank_score", 0.0)),
            )
        )
    return citations


class Answerer:
    def __init__(self, cfg: Config, retriever: Retriever | None = None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)
        self.min_relevance = float(cfg.retrieval.get("min_relevance", 0.05))
        self.min_support = float(cfg.answer.get("min_support", 0.35))
        self.verify = bool(cfg.answer.get("verify_grounding", True))
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_client(self.cfg)
        return self._client

    # -- verification ---------------------------------------------------------

    @staticmethod
    def _evidence_text(hit: dict[str, Any]) -> str:
        """What the model saw for one source: its header and its passage.

        The header carries document metadata (dates, semester, version) that
        is not repeated in the passage, so a correct sentence quoting it must
        not be flagged.
        """
        meta = hit.get("meta") or {}
        header = " ".join(str(part) for part in (
            meta.get("title", ""), meta.get("semester", ""), meta.get("version_label", ""),
            human_dates(str(meta.get("issued_date") or "")),
            human_dates(str(meta.get("effective_from") or "")),
        ) if part)
        return f"{header} {hit.get('text') or ''}"

    def _check_support(
        self, answer_text: str, hits: list[dict[str, Any]]
    ) -> list[SentenceSupport]:
        """Check each cited sentence against the passages it cites.

        Dates, fees, times, course codes, sections and rooms all carry digits,
        and a sentence stating one must find it verbatim (after normalisation)
        in a cited passage. Embedding similarity cannot do this job: measured on
        the evaluation set, a correct answer scored 0.53-0.81 against its own
        evidence and 0.48-0.70 against the wrong routine passage, so no
        threshold separates them. Similarity is kept only for sentences with no
        digit-bearing terms, where nothing better is available.
        """
        sentences = [s.strip() for s in SENTENCE_RE.split(answer_text) if s.strip()]
        cited_sentences = [
            (s, [int(n) for n in CITATION_RE.findall(s)]) for s in sentences
        ]
        cited_sentences = [(s, ns) for s, ns in cited_sentences if ns]
        if not cited_sentences:
            return []

        embedder = self.retriever.embedder
        clean = [CITATION_RE.sub("", s).strip() for s, _ in cited_sentences]
        sentence_vectors = embedder.encode_documents(clean, show_progress=False)
        chunk_vectors = embedder.encode_documents(
            [(h.get("text") or "") for h in hits], show_progress=False
        )
        evidence = [normalize(self._evidence_text(h)) for h in hits]

        results = []
        for (sentence, numbers), text, vector in zip(cited_sentences, clean, sentence_vectors):
            valid = [n for n in numbers if 1 <= n <= len(hits)]
            best = max((float(np.dot(vector, chunk_vectors[n - 1])) for n in valid), default=0.0)
            terms = checkable_terms(text)
            if terms and valid:
                missing = [t for t in terms
                           if not any(contains_term(evidence[n - 1], t) for n in valid)]
                supported, method = not missing, "terms"
            else:
                missing, method = [], "similarity"
                supported = bool(valid) and best >= self.min_support
            results.append(
                SentenceSupport(
                    sentence=sentence,
                    cited=numbers,
                    similarity=round(best, 3),
                    supported=supported,
                    method=method,
                    missing_terms=missing,
                )
            )
        return results

    # -- main entry point -----------------------------------------------------

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        k: int | None = None,
        today: date | None = None,
    ) -> Answer:
        started = time.perf_counter()
        retrieval: RetrievalResult = self.retriever.retrieve(
            question, k=k, filters=filters, today=today
        )
        hits = retrieval.hits

        if not hits or retrieval.top_score < self.min_relevance:
            return Answer(
                question=question,
                text=(
                    "I could not find this in the DIU documents I have indexed, "
                    "so I cannot answer it reliably. Try rephrasing, or check "
                    "whether the relevant notice or policy has been added to the "
                    "knowledge base."
                ),
                abstained=True,
                reason=(
                    "no candidates retrieved" if not hits
                    else f"best relevance {retrieval.top_score:.3f} is below the "
                         f"floor of {self.min_relevance:.2f}"
                ),
                top_score=retrieval.top_score,
                n_retrieved=len(hits),
                elapsed_s=time.perf_counter() - started,
            )

        prompt = USER_TEMPLATE.format(
            sources=_format_sources(hits), question=question
        )
        try:
            raw = self.client.complete(SYSTEM_PROMPT, prompt)
        except LLMUnavailable:
            raise
        raw = normalize_citations(raw)

        warnings: list[str] = []

        if ABSTAIN_SENTINEL in raw:
            return Answer(
                question=question,
                text=(
                    "The documents I retrieved do not actually answer this "
                    "question. I would rather say so than guess."
                ),
                abstained=True,
                reason="the model judged the retrieved passages insufficient",
                top_score=retrieval.top_score,
                n_retrieved=len(hits),
                elapsed_s=time.perf_counter() - started,
                llm_called=True,
            )

        # Strip citations pointing at passages that were never retrieved.
        valid = set(range(1, len(hits) + 1))
        invented = {int(n) for n in CITATION_RE.findall(raw)} - valid
        if invented:
            warnings.append(
                "removed citation(s) to non-existent sources: "
                + ", ".join(f"[{n}]" for n in sorted(invented))
            )
            raw = CITATION_RE.sub(
                lambda m: "" if int(m.group(1)) in invented else m.group(0), raw
            )

        used = {int(n) for n in CITATION_RE.findall(raw)}
        if not used:
            warnings.append(
                "the answer cites no source; treat it as unverified"
            )

        support = self._check_support(raw, hits) if self.verify else []
        unsupported = [s for s in support if not s.supported]
        if unsupported:
            missing = sorted({t for s in unsupported for t in s.missing_terms})
            detail = f" (not found in the cited source: {', '.join(missing)})" if missing else ""
            warnings.append(
                f"{len(unsupported)} of {len(support)} cited sentences are not backed "
                f"by the source they cite{detail}"
            )

        # A sentence whose terms were found in its source earns full credit. One
        # that only looks similar to its source earns half: similarity cannot
        # tell a true sentence from a plausible false one about the same topic.
        credit = {"terms": 1.0, "similarity": 0.5}
        supported_share = (
            sum(credit.get(item.method, 0.5) for item in support if item.supported) / len(support)
            if support else 0.0
        )
        # Relevance of what was retrieved, and how much of the answer checked out.
        confidence = 0.6 * retrieval.top_score + 0.4 * supported_share
        if not used:
            confidence *= 0.4

        return Answer(
            question=question,
            text=raw.strip(),
            citations=_build_citations(hits, used),
            support=support,
            confidence=round(confidence, 3),
            top_score=round(retrieval.top_score, 3),
            n_retrieved=len(hits),
            elapsed_s=time.perf_counter() - started,
            warnings=warnings,
            llm_called=True,
        )
