"""FastAPI application.

Routes are thin: validate the request, call AssistantService, map ServiceError
to an HTTP status. Errors always have the shape
    {"detail": {"code": "<stable_code>", "message": "<human text>"}}
so the frontend can branch on `code` (e.g. show setup instructions for
`llm_unavailable`) without parsing messages.

When frontend/dist exists, the built React app is served from "/" so a single
process serves the whole application.
"""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Config, load_config
from .service import AssistantService, ServiceError


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    category: str | None = None
    conversation_id: str | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class ReindexRequest(BaseModel):
    force: bool = False


def _raise(exc: ServiceError) -> None:
    raise HTTPException(status_code=exc.status,
                        detail={"code": exc.code, "message": exc.message})


def create_app(
    cfg: Config | None = None,
    service: AssistantService | None = None,
    warm_up: bool = True,
) -> FastAPI:
    cfg = cfg or load_config()
    state: dict[str, AssistantService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = service or AssistantService(cfg)
        if warm_up:
            await run_in_threadpool(svc.warm_up)
        state["service"] = svc
        yield

    app = FastAPI(title="DIU Academic Assistant", version="0.3.0", lifespan=lifespan)

    origins = list(cfg.api.get("cors_origins", []) or [])
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins,
            allow_methods=["*"], allow_headers=["*"],
        )

    def get_service() -> AssistantService:
        return state["service"]

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        expected = str(cfg.api.get("admin_token") or "")
        if not expected:
            return
        if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
            raise HTTPException(
                status_code=401,
                detail={"code": "admin_token_required",
                        "message": "This action requires a valid admin token."},
            )

    @app.exception_handler(ServiceError)
    async def service_error_handler(_, exc: ServiceError):
        return JSONResponse(status_code=exc.status,
                            content={"detail": {"code": exc.code,
                                                "message": exc.message}})

    # -- read endpoints -------------------------------------------------------

    @app.get("/api/health")
    def health(svc: AssistantService = Depends(get_service)):
        return svc.health()

    @app.get("/api/categories")
    def categories(svc: AssistantService = Depends(get_service)):
        return svc.categories()

    @app.post("/api/ask")
    def ask(body: AskRequest, svc: AssistantService = Depends(get_service)):
        return svc.ask(body.question, body.category or None, body.conversation_id)

    @app.get("/api/conversations")
    def conversations(svc: AssistantService = Depends(get_service)):
        return svc.history.list_conversations()

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: str, svc: AssistantService = Depends(get_service)):
        found = svc.history.get_conversation(conversation_id)
        if not found:
            _raise(ServiceError(404, "conversation_not_found", "No such conversation."))
        return found

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str,
                            svc: AssistantService = Depends(get_service)):
        if not svc.history.delete_conversation(conversation_id):
            _raise(ServiceError(404, "conversation_not_found", "No such conversation."))

    @app.post("/api/feedback", status_code=204)
    def feedback(body: FeedbackRequest, svc: AssistantService = Depends(get_service)):
        if not svc.history.set_feedback(body.message_id, body.rating, body.comment):
            _raise(ServiceError(404, "message_not_found", "No such message."))

    @app.get("/api/documents")
    def documents(svc: AssistantService = Depends(get_service)):
        return svc.documents()

    @app.get("/api/documents/{doc_id}/file")
    def document_file(doc_id: str, svc: AssistantService = Depends(get_service)):
        path, media_type = svc.document_file(doc_id)
        # inline, so PDFs open in the browser and honour #page=N from citations
        return FileResponse(path, media_type=media_type, filename=path.name,
                            content_disposition_type="inline")

    # -- write endpoints ------------------------------------------------------

    @app.post("/api/documents", dependencies=[Depends(require_admin)])
    async def upload(
        file: UploadFile = File(...),
        replace: bool = Form(False),
        svc: AssistantService = Depends(get_service),
    ):
        # Read one byte past the limit so oversized uploads are rejected
        # without buffering an arbitrarily large body.
        data = await file.read(svc.max_upload_bytes + 1)
        return await run_in_threadpool(svc.upload, file.filename or "", data, replace)

    @app.post("/api/reindex", dependencies=[Depends(require_admin)])
    def reindex(body: ReindexRequest | None = None,
                svc: AssistantService = Depends(get_service)):
        return svc.reindex(force=bool(body and body.force))

    # -- frontend -------------------------------------------------------------

    dist = Path(cfg.paths.get("frontend_dist", "frontend/dist"))
    index_html = dist / "index.html"

    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail={"code": "not_found",
                                                         "message": "Unknown endpoint."})
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
            return FileResponse(candidate)
        if index_html.is_file():
            return FileResponse(index_html)
        return HTMLResponse(
            "<h1>DIU Academic Assistant API</h1>"
            "<p>The web UI has not been built. Run <code>npm run build</code> in "
            "<code>frontend/</code>, or use the dev server on port 5173. "
            "API docs: <a href='/docs'>/docs</a></p>"
        )

    return app
