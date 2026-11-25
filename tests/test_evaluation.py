"""Tests for the evaluation harness.

An evaluation that scores wrongly is worse than none: it produces confident
numbers for a thesis. These tests pin the scoring rules - fact matching,
outcome labels, citation checks and the report - using stub answerers and a
hand-built registry, so they run in about a second with no models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diu.answer import Answer, Citation, SentenceSupport
from diu.evaluation import (
    EvalItem, EvalSetError, check_eval_set, fact_present, facts_found,
    feedback_to_candidates, is_refusal, is_relevant, load_eval_set, normalize_answer,
    render_report, run_closed_book, run_rag, summarize_answers, wilson,
)
from diu.models import Chunk, DocMeta
from diu.registry import Registry

ROOT = Path(__file__).resolve().parent.parent
NOTICE = "notice-for-late-registration-for-fall-2026-524065d2ca.pdf"
NOTICE_TEXT = ("Notice | Fall 2026\n\nThe last date for late registration is 17 September 2026\n"
               "with a late fine of Tk. 500/=.")


def _item(**overrides) -> EvalItem:
    base = dict(
        id="reg-01", split="test", type="answerable",
        question="When is the last date for late registration?",
        gold_docs=["notice-for-late-registration-for-fall-2026"],
        evidence=["The last date for late registration is 17 September 2026"],
        answer_facts=[["17 September 2026", "September 17, 2026"]],
        reference_answer="17 September 2026.",
    )
    base.update(overrides)
    return EvalItem(**base)


# -- fact matching -----------------------------------------------------------

@pytest.mark.parametrize("answer, fact", [
    ("The class runs 8:30 AM to 10:00 AM [1].", "08:30-10:00"),
    ("It is from 01:00 - 02:30 in the afternoon.", "01:00-02:30"),
    ("Room KT-318 (A), on Thursday.", "KT-318(A)"),
    ("The deadline is 17th September, 2026 [1].", "17 September 2026"),
    ("A late fine of Tk. 500/= applies.", "500"),
    ("It is **V1.1** of the routine.", "V1.1"),
])
def test_equivalent_forms_match(answer, fact):
    assert fact_present(normalize_answer(answer), [fact])


@pytest.mark.parametrize("answer, fact", [
    ("The class starts at 11:30.", "V1.1"),        # "1.1" must not match inside "11:30"
    ("The fine is Tk. 5000.", "500"),              # no partial-number matches
    ("Section 67_B meets on Saturday.", "67"),
    ("Taught by HHPX.", "HHP"),
])
def test_near_misses_do_not_match(answer, fact):
    assert not fact_present(normalize_answer(answer), [fact])


def test_every_fact_is_scored_separately():
    item = _item(answer_facts=[["KT-201"], ["11:30-01:00"], ["MTF"]])
    assert facts_found("Room KT-201 at 11:30-1:00 PM.", item) == [True, True, False]


def test_evidence_matching_ignores_line_breaks_and_case():
    item = _item()
    assert is_relevant(item, NOTICE_TEXT, NOTICE)
    assert not is_relevant(item, NOTICE_TEXT, "some-other-document.pdf")


# -- the shipped question set --------------------------------------------------

def test_shipped_question_set_is_valid_and_balanced():
    items, meta = load_eval_set(ROOT / "eval" / "questions.yaml")
    assert len({i.id for i in items}) == len(items)
    for split in ("dev", "test"):
        part = [i for i in items if i.split == split]
        assert any(i.answerable for i in part) and any(not i.answerable for i in part)
    assert meta["sha256"]


def test_malformed_sets_report_every_problem(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"items": [
        {"id": "a", "split": "test", "type": "answerable", "question": "Q?"},
        {"id": "a", "split": "train", "type": "maybe", "question": ""},
        {"id": "u", "split": "dev", "type": "unanswerable", "question": "Q?",
         "answer_facts": ["x"]},
    ]}), encoding="utf-8")
    with pytest.raises(EvalSetError) as caught:
        load_eval_set(path)
    text = " ".join(caught.value.problems)
    for expected in ("need gold_docs", "need answer_facts", "duplicate id", "split must be",
                     "type must be", "missing question", "must not have answer_facts"):
        assert expected in text


# -- statistics ------------------------------------------------------------------

def test_wilson_interval_is_sane_for_small_samples():
    low, high = wilson(9, 10)
    assert 0.55 < low < 0.6 and 0.98 < high <= 1.0
    assert wilson(0, 0) == (0.0, 0.0)
    low, high = wilson(0, 12)
    assert low == 0.0 and 0.2 < high < 0.3


# -- consistency check against the index ---------------------------------------

@pytest.fixture()
def registry(tmp_path) -> Registry:
    reg = Registry(tmp_path / "registry.sqlite")
    meta = DocMeta(doc_id="d_notice", source_path=f"Sources/{NOTICE}", filename=NOTICE,
                   sha256="x", title="Notice for Late Registration for Fall 2026",
                   semester="Fall 2026")
    reg.upsert_document(meta, "indexed", 1, [])
    reg.replace_chunks("d_notice", [Chunk(chunk_id="c1", doc_id="d_notice", ord=0,
                                          text=NOTICE_TEXT, page_start=1, page_end=1)])
    return reg


def test_check_flags_evidence_missing_from_the_index(registry):
    good = _item()
    typo = _item(id="reg-02", evidence=["The last date for late registration is 18 September 2026"])
    missing = _item(id="reg-03", gold_docs=["hostel-notice"])
    errors, _ = check_eval_set([good, typo, missing], registry)
    assert not any(e.startswith("reg-01") for e in errors)
    assert any(e.startswith("reg-02") and "evidence not found" in e for e in errors)
    assert any(e.startswith("reg-03") and "no document matches" in e for e in errors)


# -- end-to-end scoring ------------------------------------------------------------

class StubAnswerer:
    def __init__(self, answers: dict[str, Answer]):
        self.answers = answers

    def ask(self, question: str) -> Answer:
        return self.answers[question]


def _citation(chunk_id="c1", filename=NOTICE) -> Citation:
    return Citation(n=1, chunk_id=chunk_id, doc_id="d_notice", filename=filename,
                    title="Notice", page=1, section="", source_path="", doc_date="2026-09-12",
                    category="Registration", score=0.99)


def test_rag_outcomes_and_citation_checks(registry):
    correct = _item(id="a1", question="q1")
    wrong = _item(id="a2", question="q2")
    refused = _item(id="a3", question="q3")
    junk_refused = _item(id="u1", type="unanswerable", question="q4", gold_docs=[],
                         evidence=[], answer_facts=[])
    junk_answered = _item(id="u2", type="unanswerable", question="q5", gold_docs=[],
                          evidence=[], answer_facts=[])
    support = [SentenceSupport("s", [1], 0.8, True, "terms"),
               SentenceSupport("t", [1], 0.6, False, "terms", ["20"]),
               SentenceSupport("u", [1], 0.7, True, "similarity")]
    answers = {
        "q1": Answer("q1", "The last date is 17 September 2026 [1].", citations=[_citation()],
                     support=support[:1], llm_called=True),
        "q2": Answer("q2", "The last date is 20 September 2026 [1].",
                     citations=[_citation(chunk_id="c-other", filename="other.pdf")], support=support[1:],
                     llm_called=True),
        "q3": Answer("q3", "I could not find this.", abstained=True, reason="floor"),
        "q4": Answer("q4", "I could not find this.", abstained=True, reason="floor"),
        "q5": Answer("q5", "The hostel fee is Tk. 20000.", llm_called=True),
    }
    items = [correct, wrong, refused, junk_refused, junk_answered]
    records = run_rag(items, StubAnswerer(answers), registry)
    outcomes = {r["id"]: r["outcome"] for r in records}
    assert outcomes == {"a1": "correct", "a2": "wrong", "a3": "refused",
                        "u1": "refused", "u2": "answered"}
    by_id = {r["id"]: r for r in records}
    assert by_id["a1"]["cited_evidence"] is True
    assert by_id["a2"]["cited_evidence"] is False
    assert by_id["a2"]["citation_doc_precision"] == 0.0

    summary = summarize_answers(records, with_citations=True)
    assert summary["accuracy"]["k"] == 1 and summary["accuracy"]["n"] == 3
    assert summary["false_refusal"]["k"] == 1
    assert summary["correct_refusal"]["k"] == 1 and summary["correct_refusal"]["n"] == 2
    # Answers given: a1, a2, u2. Unreliable among them: a2 (wrong) and u2 (answered junk).
    assert (summary["unreliable_answers"]["k"], summary["unreliable_answers"]["n"]) == (2, 3)
    # Term-checked sentences: s (verified) and t (flagged). u was similarity-only.
    assert (summary["verified_sentences"]["k"], summary["verified_sentences"]["n"]) == (1, 2)
    assert (summary["unverifiable_sentences"]["k"], summary["unverifiable_sentences"]["n"]) == (1, 3)
    assert by_id["a2"]["flagged_terms"] == ["20"]
    assert summary["uncited_answers"]["k"] == 1  # u2 answered without citations


class StubClient:
    model = "stub"

    def __init__(self, replies: dict[str, str]):
        self.replies = replies

    def complete(self, system: str, user: str) -> str:
        return self.replies[user]


def test_closed_book_refusals_and_hedged_answers():
    items = [
        _item(id="a1", question="q1"),
        _item(id="a2", question="q2"),
        _item(id="u1", type="unanswerable", question="q3", gold_docs=[], evidence=[],
              answer_facts=[]),
        _item(id="u2", type="unanswerable", question="q4", gold_docs=[], evidence=[],
              answer_facts=[]),
    ]
    client = StubClient({
        "q1": "I'm not sure, but it is usually 17 September 2026.",   # hedged, yet correct
        "q2": "I don't know the exact date.",
        "q3": "I don't have information about hostel fees.",
        "q4": "The hostel fee is around Tk. 15,000 per month.",
    })
    records = run_closed_book(items, client)
    assert [r["outcome"] for r in records] == ["correct", "refused", "refused", "answered"]
    assert is_refusal("Sorry, I cannot provide that information.")
    assert not is_refusal("The fine is Tk. 500.")


def test_report_renders_every_section(registry):
    item = _item(question="q1")
    answer = Answer("q1", "It is 17 September 2026 [1].", citations=[_citation()],
                    support=[SentenceSupport("s", [1], 0.9, True, "terms")], llm_called=True)
    records = run_rag([item], StubAnswerer({"q1": answer}), registry)
    variant = {"latency_s": 0.1, "rank": 1, "doc_rank": 1, "recall@5": 1.0}
    retrieval_records = [{"id": "reg-01", "split": "test", "type": "answerable",
                          "question": "q1", "variants": {v: dict(variant) for v in
                                                         ("dense", "sparse", "hybrid", "hybrid+rerank")},
                          "top_relevance": 0.9, "passes_floor": True}]
    from diu.evaluation import summarize_retrieval

    result = {
        "meta": {"started_at": "now", "mode": "full", "split": "test",
                 "eval_set": {"path": "eval/questions.yaml", "sha256": "abc"},
                 "n_items": 1, "n_answerable": 1, "n_unanswerable": 0, "n_reviewed": 0,
                 "index": {"documents": 1, "chunks": 1},
                 "config": {"embedding": "e", "reranker": "r", "min_relevance": 0.05,
                            "answer_model": "m"}},
        "retrieval": {**summarize_retrieval(retrieval_records), "records": retrieval_records},
        "rag": {"summary": summarize_answers(records, True), "records": records},
    }
    report = render_report(result)
    for heading in ("## 1. Retrieval", "## 2. Relevance floor", "## 3. End to end",
                    "## Definitions", "not yet reviewed"):
        assert heading in report or heading.replace("not yet", "provisional") in report


def test_feedback_becomes_draft_items():
    rated = [{"id": "abcdef1234", "question": "When is late registration?", "category": None,
              "created_at": "2026-09-16", "rating": "down", "comment": "wrong date",
              "response": {"answer": "It is 20 September.", "abstained": False,
                           "citations": [{"filename": NOTICE}]}}]
    draft = feedback_to_candidates(rated)[0]
    assert draft["id"] == "fb-abcdef12"
    assert draft["reviewed"] is False
    assert draft["gold_docs"] == [NOTICE]
    assert "wrong date" in draft["notes"]
