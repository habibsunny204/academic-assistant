"""HTTP API tests.

Each test module builds a throwaway corpus in a temp directory, indexes it with
the real embedding and reranking models, and swaps only the LLM for a stub. That
exercises the real retrieval path end to end without an API key and without
touching the project's own index.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diu.api import create_app
from diu.api.service import AssistantService
from diu.config import Config, load_config
from diu.ingest import ingest
from diu.llm import LLMUnavailable

NOTICE = """Daffodil International University
NOTICE
Date: 12 September 2026

Students are requested to complete course registration for the Fall 2026 semester.
The last date for late registration is 17 September 2026 with a late fine of Tk. 500.
Registration will be allowed until 24 September 2026 with a late fine of Tk. 1000.
After that, registration will not be permitted.
"""

EXAM_POLICY = """Improvement Examination Policy, Summer 2026

A student may apply for an improvement examination for a course in which the grade
obtained is C+ or below. The improvement examination form must be submitted to the
department office with clearance from accounts before the application deadline.
"""


class FakeLLM:
    name = "fake"
    model = "fake-llama"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return ("The last date for late registration is 17 September 2026 with a "
                "late fine of Tk. 500 [1].")


def _config(tmp: Path, admin_token: str = "") -> Config:
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["paths"].update({
        "sources": str(tmp / "Sources"),
        "registry_db": str(tmp / "data" / "registry.sqlite"),
        "vector_dir": str(tmp / "data" / "vectordb"),
        "quarantine": str(tmp / "data" / "quarantine"),
        "app_db": str(tmp / "data" / "app.sqlite"),
        "frontend_dist": str(tmp / "no-frontend"),
    })
    cfg["api"]["admin_token"] = admin_token
    return cfg


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    sources = tmp / "Sources"
    sources.mkdir()
    (sources / "notice-late-registration-fall-2026.txt").write_text(NOTICE, encoding="utf-8")
    (sources / "improvement-examination-policy-summer-2026.txt").write_text(
        EXAM_POLICY, encoding="utf-8")

    cfg = _config(tmp)
    report = ingest(cfg)
    assert report.counts().get("indexed") == 2

    service = AssistantService(cfg)
    llm = FakeLLM()
    service.answerer._client = llm
    app = create_app(cfg, service=service, warm_up=False)
    with TestClient(app) as client:
        yield {"client": client, "service": service, "llm": llm, "cfg": cfg, "tmp": tmp}


# -- status ------------------------------------------------------------------

def test_health_reports_ready_index(env):
    body = env["client"].get("/api/health").json()
    assert body["status"] == "ok"
    assert body["index"]["state"] == "ready"
    assert body["index"]["documents"] >= 2
    assert body["llm"]["configured"] is True


def test_categories_come_from_the_index(env):
    names = {c["name"] for c in env["client"].get("/api/categories").json()}
    assert "Registration" in names


# -- asking ------------------------------------------------------------------

def test_ask_returns_citations_the_ui_can_render(env):
    response = env["client"].post(
        "/api/ask", json={"question": "What is the last date for late registration?"})
    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is False
    assert "17 September 2026" in body["answer"]
    citation = body["citations"][0]
    assert citation["n"] == 1
    assert citation["doc_id"].startswith("d_")
    assert citation["url"] == f"/api/documents/{citation['doc_id']}/file"
    assert citation["file_type"] == "TXT"
    assert citation["doc_date"] == "2026-09-12"
    details = body["details"]
    assert details["llm_called"] is True
    assert details["model"] == "fake-llama"
    assert details["confidence_band"] in {"high", "medium", "low"}
    assert body["message_id"] and body["conversation_id"]


def test_follow_up_joins_the_same_conversation(env):
    client = env["client"]
    first = client.post("/api/ask", json={"question": "When does late registration end?"}).json()
    second = client.post("/api/ask", json={
        "question": "What is the late registration fine?",
        "conversation_id": first["conversation_id"],
    }).json()
    assert second["conversation_id"] == first["conversation_id"]

    convo = client.get(f"/api/conversations/{first['conversation_id']}").json()
    assert [m["message_id"] for m in convo["messages"]] == [
        first["message_id"], second["message_id"]]


def test_out_of_domain_question_abstains_without_the_llm(env):
    before = env["llm"].calls
    body = env["client"].post(
        "/api/ask", json={"question": "What is the capital of France?"}).json()
    assert body["abstained"] is True
    assert body["citations"] == []
    assert body["details"]["llm_called"] is False
    assert body["details"]["model"] is None
    assert env["llm"].calls == before


def test_blank_question_is_rejected(env):
    response = env["client"].post("/api/ask", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_question"


def test_missing_llm_returns_setup_error_not_a_crash(env, monkeypatch):
    service = env["service"]
    saved = service.answerer._client
    service.answerer._client = None

    def unavailable(cfg):
        raise LLMUnavailable("GROQ_API_KEY is not set.")

    monkeypatch.setattr("diu.answer.get_client", unavailable)
    try:
        response = env["client"].post(
            "/api/ask", json={"question": "What is the last date for late registration?"})
    finally:
        service.answerer._client = saved
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "llm_unavailable"


def test_category_filter_excludes_other_categories(env):
    body = env["client"].post("/api/ask", json={
        "question": "What is the last date for late registration?",
        "category": "Examination",
    }).json()
    assert all(c["category"] == "Examination" for c in body["citations"])


# -- feedback ----------------------------------------------------------------

def test_feedback_is_stored_with_the_message(env):
    client = env["client"]
    answer = client.post(
        "/api/ask", json={"question": "What is the late registration fine?"}).json()
    assert client.post("/api/feedback", json={
        "message_id": answer["message_id"], "rating": "down",
        "comment": "missing the second deadline"}).status_code == 204

    convo = client.get(f"/api/conversations/{answer['conversation_id']}").json()
    assert convo["messages"][-1]["feedback"] == "down"
    assert client.get("/api/health").json()["feedback"]["down"] >= 1


def test_feedback_for_unknown_message_is_404(env):
    response = env["client"].post(
        "/api/feedback", json={"message_id": "nope", "rating": "up"})
    assert response.status_code == 404


def test_deleting_a_conversation(env):
    client = env["client"]
    answer = client.post("/api/ask", json={"question": "Late registration fine?"}).json()
    assert client.delete(f"/api/conversations/{answer['conversation_id']}").status_code == 204
    assert client.get(f"/api/conversations/{answer['conversation_id']}").status_code == 404


# -- documents ---------------------------------------------------------------

def test_documents_list_and_file_download(env):
    client = env["client"]
    docs = client.get("/api/documents").json()
    notice = next(d for d in docs if d["filename"].startswith("notice"))
    assert notice["status"] == "indexed"

    response = client.get(notice["url"])
    assert response.status_code == 200
    assert "17 September 2026" in response.text
    assert "inline" in response.headers["content-disposition"]


def test_unknown_document_is_404(env):
    assert env["client"].get("/api/documents/d_doesnotexist/file").status_code == 404


def test_file_outside_sources_is_never_served(env, tmp_path):
    """A registry row pointing outside Sources/ must not become a file read."""
    service = env["service"]
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    doc = service.registry.list_documents()[0]
    with service.registry._connect() as conn:
        conn.execute("UPDATE documents SET source_path = ? WHERE doc_id = ?",
                     (str(secret), doc["doc_id"]))
    try:
        response = env["client"].get(f"/api/documents/{doc['doc_id']}/file")
    finally:
        with service.registry._connect() as conn:
            conn.execute("UPDATE documents SET source_path = ? WHERE doc_id = ?",
                         (doc["source_path"], doc["doc_id"]))
    assert response.status_code == 404


def test_upload_rejects_unsupported_types(env):
    response = env["client"].post(
        "/api/documents", files={"file": ("virus.exe", b"MZ...", "application/octet-stream")})
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_type"


def test_upload_indexes_the_file_and_makes_it_answerable(env):
    client = env["client"]
    text = ("Hostel Notice, Fall 2026. The monthly hostel seat rent for the Fall 2026 "
            "semester is Tk. 4500 for a shared room. Applications close on "
            "30 September 2026. Contact the hostel office in the Daffodil Smart City.")
    response = client.post(
        "/api/documents",
        files={"file": ("../../hostel-notice-fall-2026.txt", text.encode(), "text/plain")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Directory components are stripped: the file lands inside Sources/uploads/.
    assert body["filename"] == "hostel-notice-fall-2026.txt"
    assert body["status"] == "indexed"
    assert (env["tmp"] / "Sources" / "uploads" / "hostel-notice-fall-2026.txt").is_file()

    # Available to retrieval immediately, without restarting the server.
    retrieved = env["service"].retriever.retrieve("What is the hostel seat rent?", k=3)
    assert any("Tk. 4500" in (h.get("text") or "") for h in retrieved.hits)

    duplicate = client.post(
        "/api/documents",
        files={"file": ("hostel-notice-fall-2026.txt", text.encode(), "text/plain")})
    assert duplicate.status_code == 409

    replaced = client.post(
        "/api/documents",
        files={"file": ("hostel-notice-fall-2026.txt", text.encode(), "text/plain")},
        data={"replace": "true"})
    assert replaced.status_code == 200


def test_reindex_reports_unchanged_files(env):
    body = env["client"].post("/api/reindex", json={}).json()
    assert body["counts"].get("unchanged", 0) >= 2


def test_admin_token_protects_write_endpoints(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg = Config(cfg)
    cfg["api"]["admin_token"] = "s3cret"
    app = create_app(cfg, service=env["service"], warm_up=False)
    with TestClient(app) as client:
        assert client.post("/api/reindex", json={}).status_code == 401
        assert client.post("/api/reindex", json={},
                           headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert client.post("/api/reindex", json={},
                           headers={"X-Admin-Token": "s3cret"}).status_code == 200
        # Reading stays open.
        assert client.get("/api/documents").status_code == 200


def test_unbuilt_frontend_shows_a_hint_not_a_404(env):
    response = env["client"].get("/")
    assert response.status_code == 200
    assert "npm run build" in response.text
