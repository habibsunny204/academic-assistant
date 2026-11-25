"""Tests for grounded answering.

The LLM is replaced with a stub so the guard rails can be tested deterministically
and without an API key. What is being tested here is not the model's writing --
it is whether the system catches the model when it misbehaves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diu.answer import Answerer
from diu.config import load_config
from diu.embed import Embedder
from diu.retrieve import RetrievalResult

NOTICE_TEXT = (
    "Notice for Late Registration | Fall 2026\n\n"
    "The last date for late registration is 17 September 2026 with a late fine "
    "of Tk. 500/=. Registration will be allowed till 24 September 2026 with a "
    "late fine of Tk. 1000/=."
)
ROUTINE_TEXT = (
    "Class Routine for CSE Program | Fall 2026 | V1.1 > Saturday\n\n"
    "Course CSE315 section 67_B is scheduled on Saturday from 11:30-01:00 in "
    "room KT-201 taught by AS."
)


def _hits():
    return [
        {
            "chunk_id": "c1", "text": NOTICE_TEXT, "rerank_score": 0.69,
            "meta": {"title": "Notice for Late Registration", "page_start": 1,
                     "issued_date": "2026-09-12", "category": "Registration",
                     "source_path": "Sources/notice.pdf", "section_path": ""},
        },
        {
            "chunk_id": "c2", "text": ROUTINE_TEXT, "rerank_score": 0.61,
            "meta": {"title": "Class Routine for CSE Program", "page_start": 1,
                     "effective_from": "2026-09-09", "category": "Class Routine",
                     "source_path": "Sources/routine.pdf",
                     "section_path": "Saturday"},
        },
    ]


class FakeRetriever:
    def __init__(self, cfg, hits, top_score):
        self.embedder = Embedder(cfg)
        self._hits = hits
        self._top = top_score

    def retrieve(self, query, k=None, filters=None, today=None):
        return RetrievalResult(query=query, hits=self._hits,
                               n_candidates=len(self._hits), top_score=self._top)


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def _answerer(cfg, hits, top_score, reply):
    answerer = Answerer(cfg, retriever=FakeRetriever(cfg, hits, top_score))
    llm = FakeLLM(reply)
    answerer._client = llm
    return answerer, llm


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# -- abstention --------------------------------------------------------------

def test_low_relevance_abstains_without_calling_the_llm(cfg):
    """The cheapest way to avoid a confident wrong answer is not to ask."""
    # Relative to the configured floor: absolute scores depend on the reranker.
    below_floor = float(cfg.retrieval["min_relevance"]) * 0.5
    answerer, llm = _answerer(cfg, _hits(), below_floor, "The fee is 5000 Taka. [1]")
    result = answerer.ask("What is the capital of France?")

    assert result.abstained
    assert llm.calls == 0
    assert result.confidence_band == "none"


def test_empty_retrieval_abstains(cfg):
    answerer, llm = _answerer(cfg, [], 0.0, "anything")
    result = answerer.ask("Anything at all")

    assert result.abstained
    assert llm.calls == 0


def test_model_sentinel_is_honoured(cfg):
    """Retrieval can pass the floor while the passages still lack the answer."""
    answerer, _ = _answerer(cfg, _hits(), 0.69, "INSUFFICIENT_CONTEXT")
    result = answerer.ask("What is the DIU hostel fee for 2026?")

    assert result.abstained
    assert "insufficient" in result.reason.lower()


# -- citation integrity ------------------------------------------------------

def test_citations_to_nonexistent_sources_are_stripped(cfg):
    """A model citing [9] when only two passages exist must not be believed."""
    reply = ("The late fine is Tk. 500/= [1]. The hostel fee is Tk. 20000 [9].")
    answerer, _ = _answerer(cfg, _hits(), 0.69, reply)
    result = answerer.ask("What is the late fine?")

    assert "[9]" not in result.text
    assert any("non-existent" in w for w in result.warnings)
    assert [c.n for c in result.citations] == [1]


@pytest.mark.parametrize("marker", ["【1】", "【1†L2-L4】", "［1］"])
def test_other_citation_styles_are_recognised(cfg, marker):
    """gpt-oss cites as 【1】; missing that made correct answers look uncited."""
    reply = f"The last date for late registration is 17 September 2026{marker}."
    answerer, _ = _answerer(cfg, _hits(), 0.69, reply)
    result = answerer.ask("When is late registration?")

    assert [c.n for c in result.citations] == [1]
    assert "[1]" in result.text
    assert result.support and result.support[0].supported


def test_grouped_citations_are_split():
    from diu.answer import normalize_citations

    assert normalize_citations("Both apply [1, 2].") == "Both apply [1][2]."


def test_uncited_answer_is_flagged_and_penalised(cfg):
    answerer, _ = _answerer(cfg, _hits(), 0.69, "The late fine is Tk. 500.")
    result = answerer.ask("What is the late fine?")

    assert result.citations == []
    assert any("cites no source" in w for w in result.warnings)
    assert result.confidence < 0.35


def test_citations_carry_what_the_ui_renders(cfg):
    answerer, _ = _answerer(
        cfg, _hits(), 0.69,
        "The last date for late registration is 17 September 2026 [1].",
    )
    result = answerer.ask("When is late registration?")

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.title == "Notice for Late Registration"
    assert citation.page == 1
    assert citation.doc_date == "2026-09-12"
    assert "page 1" in citation.label()


# -- grounding verification --------------------------------------------------

def test_supported_sentence_passes_verification(cfg):
    reply = ("The last date for late registration is 17 September 2026 with a "
             "late fine of Tk. 500 [1].")
    answerer, _ = _answerer(cfg, _hits(), 0.69, reply)
    result = answerer.ask("When is the last date for late registration?")

    assert result.support
    assert all(s.supported for s in result.support)
    assert not any("not backed" in w for w in result.warnings)


def test_invented_claim_attached_to_a_real_source_is_caught(cfg):
    """The dangerous failure: a fabricated fact wearing a valid citation."""
    reply = ("Students receive a free laptop and a dormitory room upon "
             "enrolment in the faculty of marine biology [1].")
    answerer, _ = _answerer(cfg, _hits(), 0.69, reply)
    result = answerer.ask("What do students receive?")

    assert result.support
    assert not result.support[0].supported
    assert any("not backed" in w for w in result.warnings)


def test_wrong_number_with_a_valid_citation_is_caught(cfg):
    """Similarity alone passed this: the sentence reads exactly like its source."""
    answerer, _ = _answerer(cfg, _hits(), 0.69, "The late fine is Tk. 700/= until 17 September 2026 [1].")
    result = answerer.ask("What is the late fine?")

    assert not result.support[0].supported
    assert result.support[0].method == "terms"
    assert result.support[0].missing_terms == ["700"]
    assert any("700" in w for w in result.warnings)


def test_right_fact_cited_to_the_wrong_source_is_caught(cfg):
    """The routine fact is true, but passage [1] is the notice, not the routine."""
    answerer, _ = _answerer(cfg, _hits(), 0.69,
                            "CSE315 section 67_B meets in room KT-201 from 11:30 to 1:00 PM [1].")
    result = answerer.ask("Where is CSE315 67_B?")
    assert not result.support[0].supported

    answerer, _ = _answerer(cfg, _hits(), 0.69,
                            "CSE315 section 67_B meets in room KT-201 from 11:30 to 1:00 PM [2].")
    assert answerer.ask("Where is CSE315 67_B?").support[0].supported


def test_facts_from_the_source_header_count_as_support(cfg):
    """The effective date is document metadata, shown to the model in the header."""
    answerer, _ = _answerer(cfg, _hits(), 0.69,
                            "The Fall 2026 routine is effective from 9 September 2026 [2].")
    result = answerer.ask("When is the routine effective?")
    assert result.support[0].supported


def test_confidence_is_higher_for_a_grounded_answer(cfg):
    grounded = ("The last date for late registration is 17 September 2026 with "
                "a late fine of Tk. 500/= [1].")
    invented = ("Students receive a free laptop in the faculty of marine "
                "biology [1].")

    good, _ = _answerer(cfg, _hits(), 0.69, grounded)
    bad, _ = _answerer(cfg, _hits(), 0.69, invented)

    assert (good.ask("When is late registration?").confidence
            > bad.ask("When is late registration?").confidence)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
