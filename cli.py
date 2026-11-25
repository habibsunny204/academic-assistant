"""DIU Academic Assistant - command line interface.

    python cli.py doctor            check the environment before ingesting
    python cli.py ingest            index everything under Sources/
    python cli.py ask "..."         grounded answer with citations
    python cli.py search "..."      hybrid search over the index
    python cli.py serve             web UI + HTTP API
    python cli.py docs | stats | calibrate | models
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from diu.answer import Answerer
from diu.config import load_config
from diu.embed import Embedder
from diu.llm import LLMUnavailable, get_client
from diu.ingest import ingest as run_ingest
from diu.registry import Registry
from diu.store import VectorStore

app = typer.Typer(add_completion=False, help="DIU Academic Assistant CLI")
console = Console()

STATUS_STYLE = {
    "indexed": "green",
    "unchanged": "dim",
    "quarantined": "yellow",
    "failed": "red",
}


@app.command()
def doctor() -> None:
    """Check that OCR, the embedding model and the index are usable."""
    cfg = load_config()
    console.print(Panel.fit("DIU Academic Assistant - environment check",
                            style="bold cyan"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")

    sources = Path(cfg.paths["sources"])
    table.add_row(
        "Sources folder",
        f"[green]{sources}[/green]" if sources.exists()
        else f"[red]missing: {sources}[/red]",
    )

    backend = cfg.ocr.get("backend", "none")
    if backend == "tesseract":
        cmd = Path(cfg.ocr.get("tesseract_cmd", ""))
        table.add_row(
            "Tesseract binary",
            f"[green]{cmd}[/green]" if cmd.exists()
            else f"[red]not found at {cmd}[/red]",
        )
        tessdata = Path(cfg.ocr.get("tessdata_dir", ""))
        langs = (
            sorted(p.stem for p in tessdata.glob("*.traineddata"))
            if tessdata.exists() else []
        )
        wanted = cfg.ocr.get("langs", "eng").split("+")
        missing = [lang for lang in wanted if lang not in langs]
        table.add_row(
            "OCR languages",
            f"[green]available: {', '.join(langs) or 'none'}[/green]"
            + (f" [red](missing: {', '.join(missing)})[/red]" if missing else ""),
        )
    else:
        table.add_row("OCR backend", "[yellow]disabled - scanned pages will be "
                                     "quarantined[/yellow]")

    embedder = Embedder(cfg)
    table.add_row("Embedding model", f"{embedder.model_name} ({embedder.dim}-dim)")

    registry = Registry(cfg.paths["registry_db"])
    recorded = registry.get_index_meta("embedding_signature")
    if recorded is None:
        table.add_row("Index", "[dim]empty - run 'python cli.py ingest'[/dim]")
    elif recorded != embedder.signature():
        table.add_row("Index", f"[red]built with {recorded}, config says "
                               f"{embedder.signature()} - needs --reset[/red]")
    else:
        stats = registry.stats()
        table.add_row("Index", f"[green]{stats['chunks']} chunks, "
                               f"{sum(stats['by_status'].values())} documents[/green]")
    console.print(table)


@app.command()
def ingest(
    path: str = typer.Option(None, "--path", "-p",
                             help="Ingest a single file or folder instead of Sources/"),
    force: bool = typer.Option(False, "--force", "-f",
                               help="Re-process files even if unchanged"),
    reset: bool = typer.Option(False, "--reset",
                               help="Wipe the index and rebuild from scratch"),
) -> None:
    """Extract, chunk, embed and index documents."""
    cfg = load_config()

    def progress(index: int, total: int, file_path: Path) -> None:
        console.print(f"[dim]({index}/{total})[/dim] {file_path.name}")

    try:
        report = run_ingest(cfg, target=path, force=force, reset=reset,
                            progress_callback=progress)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    table = Table(title="Ingestion result", show_header=True, header_style="bold")
    table.add_column("Status")
    table.add_column("Document")
    table.add_column("Category")
    table.add_column("Chunks", justify="right")
    table.add_column("Notes")

    for result in report.results:
        notes = result.reason
        if result.warnings:
            extra = "; ".join(result.warnings[:2])
            notes = f"{notes} {extra}".strip()
        table.add_row(
            f"[{STATUS_STYLE.get(result.status, 'white')}]{result.status}[/]",
            (result.title or Path(result.path).name)[:44],
            result.category,
            str(result.n_chunks) if result.n_chunks else "-",
            notes[:60],
        )
    console.print(table)

    for path_removed in report.pruned:
        console.print(f"[yellow]pruned (source file gone):[/yellow] {path_removed}")

    counts = ", ".join(f"{k}: {v}" for k, v in sorted(report.counts().items()))
    console.print(Panel.fit(
        f"{counts}\nvectors: {report.vector_count}  |  BM25 entries: {report.bm25_size}",
        title="Index", style="bold green"))


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language question"),
    k: int = typer.Option(5, "--k", help="Number of results"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    doc_type: str = typer.Option(None, "--type", "-t", help="Filter by document type"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid | dense | sparse"),
) -> None:
    """Search the index and show the citation each result would produce."""
    cfg = load_config()
    registry = Registry(cfg.paths["registry_db"])
    embedder = Embedder(cfg)

    recorded = registry.get_index_meta("embedding_signature")
    if recorded is None:
        console.print("[red]Index is empty. Run 'python cli.py ingest' first.[/red]")
        raise typer.Exit(code=1)

    store = VectorStore(cfg, embedder.signature())

    filters = {}
    if category:
        filters["category"] = category
    if doc_type:
        filters["doc_type"] = doc_type
    where = None
    if len(filters) == 1:
        where = filters
    elif len(filters) > 1:
        where = {"$and": [{key: value} for key, value in filters.items()]}

    if mode == "sparse":
        hits = store.sparse_search(query, k, where)
        ids = [h["chunk_id"] for h in hits]
        if ids:
            got = store.collection.get(ids=ids, include=["documents", "metadatas"])
            lookup = {cid: i for i, cid in enumerate(got["ids"])}
            for hit in hits:
                i = lookup.get(hit["chunk_id"])
                if i is not None:
                    hit["text"] = got["documents"][i]
                    hit["meta"] = got["metadatas"][i]
    elif mode == "dense":
        hits = store.dense_search(embedder.encode_query(query), k, where)
    else:
        hits = store.hybrid_search(query, embedder.encode_query(query),
                                   k=k, where=where)

    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    console.print(Panel.fit(f"[bold]{query}[/bold]", title=f"{mode} search"))
    for position, hit in enumerate(hits, start=1):
        meta = hit.get("meta") or {}
        score = hit.get("rrf")
        score_text = (
            f"RRF {score:.4f}" if score is not None
            else f"score {hit.get('score', 0):.3f}"
        )
        date_text = meta.get("issued_date") or meta.get("effective_from") or "no date"
        header = (
            f"[bold cyan]{position}.[/bold cyan] {meta.get('title', '?')} "
            f"[dim]| p.{meta.get('page_start', '?')} | {meta.get('category', '')} "
            f"| {date_text} | {score_text}[/dim]"
        )
        body = (hit.get("text") or "").strip()
        if len(body) > 420:
            body = body[:420] + " ..."
        console.print(header)
        console.print(f"   [dim]{meta.get('source_path', '')}[/dim]")
        for line in body.split("\n"):
            console.print(f"   {line}")
        console.print()


@app.command()
def docs(status: str = typer.Option(None, "--status", "-s")) -> None:
    """List documents in the registry."""
    cfg = load_config()
    registry = Registry(cfg.paths["registry_db"])
    rows = registry.list_documents(status)
    if not rows:
        console.print("[yellow]No documents. Run 'python cli.py ingest'.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    for column in ("Status", "Title", "Type", "Category", "Semester",
                   "Date", "Pages", "Chunks"):
        table.add_column(column, justify="right" if column in ("Pages", "Chunks")
                         else "left")
    for row in rows:
        table.add_row(
            f"[{STATUS_STYLE.get(row['status'], 'white')}]{row['status']}[/]",
            (row["title"] or "")[:40],
            row["doc_type"] or "",
            row["category"] or "",
            row["semester"] or "",
            row["issued_date"] or row["effective_from"] or "",
            str(row["n_pages"]),
            str(row["n_chunks"]),
        )
    console.print(table)

    for row in rows:
        warnings = json.loads(row["warnings"] or "[]")
        if warnings:
            console.print(f"[yellow]{row['title']}:[/yellow] "
                          + "; ".join(warnings[:3]))


@app.command()
def stats() -> None:
    """Corpus statistics, matching the dataset table in the project poster."""
    cfg = load_config()
    registry = Registry(cfg.paths["registry_db"])
    data = registry.stats()

    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for status, count in sorted(data["by_status"].items()):
        table.add_row(f"Documents ({status})", str(count))
    table.add_row("Pages", str(data["pages"]))
    table.add_row("Chunks", str(data["chunks"]))
    console.print(table)

    if data["by_category"]:
        by_category = Table(title="By category", show_header=True,
                            header_style="bold")
        by_category.add_column("Category")
        by_category.add_column("Documents", justify="right")
        for name, count in data["by_category"].items():
            by_category.add_row(name, str(count))
        console.print(by_category)


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question about DIU"),
    k: int = typer.Option(None, "--k", help="Passages to ground the answer on"),
    category: str = typer.Option(None, "--category", "-c", help="Restrict to a category"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
    debug: bool = typer.Option(False, "--debug", help="Show per-sentence grounding"),
) -> None:
    """Answer a question using only indexed documents, with citations."""
    cfg = load_config()
    registry = Registry(cfg.paths["registry_db"])
    if registry.get_index_meta("embedding_signature") is None:
        console.print("[red]Index is empty. Run 'python cli.py ingest' first.[/red]")
        raise typer.Exit(code=1)

    answerer = Answerer(cfg)
    try:
        result = answerer.ask(question, filters={"category": category} if category else None, k=k)
    except LLMUnavailable as exc:
        console.print(Panel(str(exc), title="LLM not configured", style="red"))
        raise typer.Exit(code=1)

    console.print(Panel.fit(f"[bold]{result.question}[/bold]", style="cyan"))

    if result.abstained:
        console.print(Panel(result.text, title="No answer", style="yellow"))
        console.print(f"[dim]Reason: {result.reason}[/dim]")
        return

    body = result.text
    for citation in result.citations:
        body = body.replace(f"[{citation.n}]", f"[bold cyan][{citation.n}][/bold cyan]")
    console.print(Panel(body, title="Answer", style="green"))

    if show_sources and result.citations:
        table = Table(title="Sources", show_header=True, header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Document")
        table.add_column("Page", justify="right")
        table.add_column("Date")
        table.add_column("Category")
        table.add_column("Relevance", justify="right")
        for citation in result.citations:
            table.add_row(
                str(citation.n), citation.title[:42],
                str(citation.page) if citation.page else "-",
                citation.doc_date or "-", citation.category,
                f"{citation.score:.2f}",
            )
        console.print(table)

    band_style = {"high": "green", "medium": "yellow", "low": "red"}
    console.print(
        f"[{band_style.get(result.confidence_band, 'white')}]"
        f"Confidence: {result.confidence_band} ({result.confidence:.2f})[/] "
        f"[dim]| retrieved {result.n_retrieved} | top relevance "
        f"{result.top_score:.2f} | {result.elapsed_s:.2f}s[/dim]"
    )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    if debug and result.support:
        table = Table(title="Grounding check", show_header=True, header_style="bold")
        table.add_column("Result")
        table.add_column("Check")
        table.add_column("Cites")
        table.add_column("Not in source")
        table.add_column("Sentence")
        for item in result.support:
            if item.method == "terms":
                verdict = "[green]verified[/green]" if item.supported else "[red]FLAGGED[/red]"
            else:
                verdict = "[yellow]unverifiable[/yellow]" if item.supported else "[red]FLAGGED[/red]"
            table.add_row(
                verdict,
                "terms" if item.method == "terms" else f"similarity {item.similarity:.2f}",
                ",".join(str(n) for n in item.cited),
                ", ".join(item.missing_terms),
                item.sentence[:60],
            )
        console.print(table)


@app.command()
def models() -> None:
    """List the models the configured provider currently serves."""
    cfg = load_config()
    try:
        client = get_client(cfg)
    except LLMUnavailable as exc:
        console.print(Panel(str(exc), title="LLM not configured", style="red"))
        raise typer.Exit(code=1)
    for name in client.list_models():
        marker = "  <- configured" if name == client.model else ""
        console.print(f"{name}{marker}")


@app.command()
def calibrate(
    questions: str = typer.Option(None, "--file", "-f",
                                  help="An eval question set (eval/questions.yaml), or YAML "
                                       "with 'in_domain' and 'out_of_domain' lists"),
    split: str = typer.Option("dev", "--split",
                              help="Which split of an eval question set to use: dev | test | all"),
) -> None:
    """Suggest an abstention threshold from questions you can and cannot answer.

    Re-run this whenever the corpus grows: the right floor depends on what the
    index actually contains. Calibrate on the dev split and report on test.
    """
    import yaml

    from diu.evaluation import EvalSetError, load_eval_set, select
    from diu.retrieve import Retriever

    cfg = load_config()
    if questions:
        data = yaml.safe_load(Path(questions).read_text(encoding="utf-8")) or {}
        if "items" in data:
            try:
                items, _ = load_eval_set(questions)
            except EvalSetError as exc:
                console.print("[red]Invalid question set:[/red]\n  " + "\n  ".join(exc.problems))
                raise typer.Exit(code=1)
            chosen = select(items, split)
            in_domain = [i.question for i in chosen if i.answerable]
            out_domain = [i.question for i in chosen if not i.answerable]
            console.print(f"Using the [bold]{split}[/bold] split: {len(in_domain)} answerable, "
                          f"{len(out_domain)} unanswerable questions.")
        else:
            in_domain = data.get("in_domain", [])
            out_domain = data.get("out_of_domain", [])
    else:
        console.print("[yellow]No --file given; using the built-in sample set. "
                      "Write your own for meaningful numbers.[/yellow]")
        in_domain = [
            "What is the fine for late registration?",
            "When is the last date for late registration?",
            "Who teaches CSE315 section 67_B?",
            "Which room is PHY101 held in on Tuesday?",
            "What happens if I do not complete course registration?",
        ]
        out_domain = [
            "What is the capital of France?",
            "How do I renew my passport in Bangladesh?",
            "Who is the Vice Chancellor of DIU?",
            "What are the library opening hours?",
            "How do I apply for a student visa to Canada?",
        ]

    if not in_domain or not out_domain:
        console.print("[red]Need both in_domain and out_of_domain questions.[/red]")
        raise typer.Exit(code=1)

    retriever = Retriever(cfg)
    with console.status("Scoring questions..."):
        scored_in = [(retriever.retrieve(q, k=5).top_score, q) for q in in_domain]
        scored_out = [(retriever.retrieve(q, k=5).top_score, q) for q in out_domain]

    table = Table(title="Relevance scores", show_header=True, header_style="bold")
    table.add_column("Kind")
    table.add_column("Score", justify="right")
    table.add_column("Question")
    for score, q in sorted(scored_in):
        table.add_row("[green]answerable[/green]", f"{score:.4f}", q[:56])
    for score, q in sorted(scored_out, reverse=True):
        table.add_row("[yellow]unanswerable[/yellow]", f"{score:.4f}", q[:56])
    console.print(table)

    grid = Table(title="Threshold trade-off", show_header=True, header_style="bold")
    grid.add_column("Threshold", justify="right")
    grid.add_column("Real questions missed", justify="right")
    grid.add_column("Junk reaching the LLM", justify="right")
    # Candidate thresholds sit between neighbouring observed scores, so the
    # scan fits whatever scale the reranker uses (bge clusters near 0.5,
    # MiniLM spans 0.0-1.0). Geometric midpoints suit scores that span decades.
    scores = sorted({round(s, 4) for s, _ in scored_in + scored_out})
    candidates: list[tuple[float, float]] = []  # (threshold, gap width)
    for low, high in zip(scores, scores[1:]):
        mid = (low * high) ** 0.5 if low > 0 else (low + high) / 2
        width = high / low if low > 0 else float("inf")
        candidates.append((mid, width))
    current = float(cfg.retrieval.get("min_relevance", 0.0))

    def outcome(threshold: float) -> tuple[int, int]:
        missed = sum(1 for s, _ in scored_in if s < threshold)
        leaked = sum(1 for s, _ in scored_out if s >= threshold)
        return missed, leaked

    best, best_key = current, None
    for threshold, width in candidates:
        missed, leaked = outcome(threshold)
        # Missing a real question is the worse failure: it makes the assistant
        # useless, while leaked junk still faces the model's own abstention.
        # Among equal costs, prefer the widest gap between scores.
        key = (missed * 2 + leaked, -width)
        if best_key is None or key < best_key:
            best, best_key = threshold, key
        grid.add_row(f"{threshold:.4f}", f"{missed}/{len(scored_in)}",
                     f"{leaked}/{len(scored_out)}")
    missed, leaked = outcome(current)
    grid.add_row(f"[bold]{current:.4f} (current)[/bold]",
                 f"{missed}/{len(scored_in)}", f"{leaked}/{len(scored_out)}")
    console.print(grid)
    console.print(Panel.fit(
        f"Suggested retrieval.min_relevance: [bold]{best:.3f}[/bold]\n"
        f"Currently set to {cfg.retrieval.get('min_relevance')}",
        style="cyan"))


@app.command()
def serve(
    host: str = typer.Option(None, "--host", help="Defaults to api.host in config.yaml"),
    port: int = typer.Option(None, "--port", help="Defaults to api.port in config.yaml"),
) -> None:
    """Run the web UI and HTTP API."""
    import uvicorn

    from diu.api import create_app

    cfg = load_config()
    host = host or cfg.api.get("host", "127.0.0.1")
    port = port or int(cfg.api.get("port", 8000))
    if host not in ("127.0.0.1", "localhost") and not cfg.api.get("admin_token"):
        console.print("[yellow]warning: serving beyond localhost with no api.admin_token "
                      "- anyone who can reach this port can upload and re-index "
                      "documents.[/yellow]")
    if not (Path(cfg.paths["frontend_dist"]) / "index.html").is_file():
        console.print("[yellow]The web UI is not built: run 'npm install' and "
                      "'npm run build' in frontend/. The API still works.[/yellow]")
    console.print(f"Loading models, then serving on [bold]http://{host}:{port}[/bold] "
                  f"(API docs at /docs)")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


DEFAULT_EVAL_SET = "eval/questions.yaml"


def _load_eval_or_exit(path: str):
    from diu.evaluation import EvalSetError, load_eval_set

    try:
        return load_eval_set(path)
    except FileNotFoundError:
        console.print(f"[red]No question set at {path}.[/red]")
        raise typer.Exit(code=1)
    except EvalSetError as exc:
        console.print("[red]Invalid question set:[/red]\n  " + "\n  ".join(exc.problems))
        raise typer.Exit(code=1)


@app.command("eval-check")
def eval_check(
    questions: str = typer.Option(DEFAULT_EVAL_SET, "--set", "-s", help="Question set YAML"),
) -> None:
    """Check that every answerable question is answerable from the current index."""
    from diu.evaluation import check_eval_set

    cfg = load_config()
    items, meta = _load_eval_or_exit(questions)
    registry = Registry(cfg.paths["registry_db"])
    errors, warnings = check_eval_set(items, registry)

    table = Table(show_header=True, header_style="bold", title=f"{questions}")
    table.add_column("Split")
    table.add_column("Answerable", justify="right")
    table.add_column("Unanswerable", justify="right")
    table.add_column("Reviewed", justify="right")
    for split in ("dev", "test"):
        part = [i for i in items if i.split == split]
        table.add_row(split, str(sum(i.answerable for i in part)),
                      str(sum(not i.answerable for i in part)),
                      f"{sum(i.reviewed for i in part)}/{len(part)}")
    console.print(table)

    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for error in errors:
        console.print(f"[red]error:[/red] {error}")
    unreviewed = sum(not i.reviewed for i in items)
    if unreviewed:
        console.print(f"[yellow]{unreviewed} items are not yet marked reviewed: true.[/yellow]")
    if errors:
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/green] - {len(items)} items consistent with the index "
                  f"(set sha256 {meta['sha256']}).")


@app.command()
def evaluate(
    questions: str = typer.Option(DEFAULT_EVAL_SET, "--set", "-s", help="Question set YAML"),
    split: str = typer.Option("test", "--split", help="test | dev | all"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only",
                                        help="Skip the LLM: retrieval and relevance floor only"),
    baseline: bool = typer.Option(True, "--baseline/--no-baseline",
                                  help="Also run the LLM without documents, for comparison"),
    limit: int = typer.Option(None, "--limit", help="Evaluate only the first N questions"),
    delay: float = typer.Option(0.0, "--delay", help="Seconds to wait between LLM calls"),
    out: str = typer.Option(None, "--out", help="Output folder (default eval/results/<time>)"),
) -> None:
    """Measure retrieval, refusals, answer accuracy and citations on a question set."""
    from datetime import datetime

    from diu.evaluation import (
        check_eval_set, run_closed_book, run_rag, run_retrieval, run_started_at, select,
        summarize_answers, summarize_retrieval, write_results,
    )

    if split not in ("test", "dev", "all"):
        console.print("[red]--split must be test, dev or all.[/red]")
        raise typer.Exit(code=1)
    cfg = load_config()
    started_at = run_started_at()
    items, set_meta = _load_eval_or_exit(questions)
    registry = Registry(cfg.paths["registry_db"])
    if registry.get_index_meta("embedding_signature") is None:
        console.print("[red]Index is empty. Run 'python cli.py ingest' first.[/red]")
        raise typer.Exit(code=1)

    errors, _ = check_eval_set(items, registry)
    if errors:
        console.print("[red]The question set does not match the index; run "
                      "'python cli.py eval-check'.[/red]")
        raise typer.Exit(code=1)

    chosen = select(items, split)[: limit or None]
    if not chosen:
        console.print("[red]No questions selected.[/red]")
        raise typer.Exit(code=1)

    answerer = Answerer(cfg)
    retriever = answerer.retriever
    mode = "retrieval" if retrieval_only else "full"
    if mode == "full":
        try:
            client = answerer.client
        except LLMUnavailable as exc:
            console.print(Panel(f"{exc}\n\nOr run: python cli.py evaluate --retrieval-only",
                                title="LLM not configured", style="red"))
            raise typer.Exit(code=1)

    def progress(label: str):
        def report(index: int, total: int, item) -> None:
            console.print(f"[dim]{label} ({index}/{total})[/dim] {item.id}: {item.question[:70]}")
        return report

    stats = registry.stats()
    result: dict = {
        "meta": {
            "started_at": started_at,
            "mode": mode,
            "split": split,
            "eval_set": set_meta,
            "n_items": len(chosen),
            "n_answerable": sum(i.answerable for i in chosen),
            "n_unanswerable": sum(not i.answerable for i in chosen),
            "n_reviewed": sum(i.reviewed for i in chosen),
            "index": {"documents": stats["by_status"].get("indexed", 0), "chunks": stats["chunks"]},
            "config": {
                "embedding": retriever.embedder.signature(),
                "reranker": retriever.reranker.model_name,
                "candidates": retriever.candidates,
                "final_k": retriever.final_k,
                "min_relevance": answerer.min_relevance,
                "min_support": answerer.min_support,
                "answer_model": cfg.answer.get("model"),
            },
        },
    }

    records = run_retrieval(chosen, retriever, registry, progress("retrieval"))
    result["retrieval"] = {**summarize_retrieval(records), "records": records}

    if mode == "full":
        rag_records = run_rag(chosen, answerer, registry, delay, progress("RAG"))
        result["rag"] = {"summary": summarize_answers(rag_records, with_citations=True),
                         "records": rag_records}
        if baseline:
            book = run_closed_book(chosen, client, delay, progress("LLM only"))
            result["closed_book"] = {"summary": summarize_answers(book, with_citations=False),
                                     "records": book}
        result["meta"]["config"]["answer_model"] = getattr(client, "model", cfg.answer.get("model"))

    folder = out or str(Path("eval") / "results" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = write_results(result, folder)

    def pct(value) -> str:
        if value is None or (isinstance(value, dict) and not value["n"]):
            return "-"
        if isinstance(value, dict):
            lo, hi = value["ci95"]
            return f"{value['rate'] * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]"
        return f"{value * 100:.0f}%"

    table = Table(title=f"Retrieval ({split}: {result['meta']['n_answerable']} answerable)",
                  header_style="bold")
    for column in ("Variant", "Hit@1", "Hit@5", "MRR", "Median time"):
        table.add_column(column, justify="left" if column == "Variant" else "right")
    for name, v in result["retrieval"]["variants"].items():
        table.add_row(name, pct(v["hit@1"]), pct(v["hit@5"]),
                      f"{v['mrr']:.3f}" if v["mrr"] is not None else "-",
                      f"{v['latency_p50_s']:.2f}s")
    console.print(table)
    floor = result["retrieval"]["floor"]
    console.print(f"Relevance floor: answerable let through {pct(floor['answerable_passed'])}, "
                  f"unanswerable stopped {pct(floor['unanswerable_stopped'])}")

    if mode == "full":
        table = Table(title="End to end", header_style="bold")
        for column in ("System", "Accuracy", "Wrong/unsupported answers", "Correct refusals",
                       "False refusals"):
            table.add_column(column, justify="left" if column == "System" else "right")
        systems = [("RAG", result["rag"]["summary"])]
        if result.get("closed_book"):
            systems.insert(0, ("LLM only", result["closed_book"]["summary"]))
        for label, s in systems:
            table.add_row(label, pct(s["accuracy"]), pct(s["unreliable_answers"]),
                          pct(s["correct_refusal"]), pct(s["false_refusal"]))
        console.print(table)

    if result["meta"]["n_reviewed"] < result["meta"]["n_items"]:
        console.print(f"[yellow]{result['meta']['n_items'] - result['meta']['n_reviewed']} of "
                      f"{result['meta']['n_items']} questions are not human-reviewed; treat "
                      f"these numbers as provisional.[/yellow]")
    console.print(Panel.fit(f"Report: {path / 'report.md'}\nPer-question detail: {path / 'items.jsonl'}",
                            style="green"))


@app.command("eval-feedback")
def eval_feedback(
    out: str = typer.Option("eval/feedback_candidates.yaml", "--out",
                            help="Where to write draft questions"),
) -> None:
    """Turn answers rated in the web UI into draft evaluation questions."""
    import yaml

    from diu.api.history import HistoryStore
    from diu.evaluation import feedback_to_candidates

    cfg = load_config()
    rated = HistoryStore(cfg.paths["app_db"]).rated_messages()
    if not rated:
        console.print("[yellow]No rated answers yet. Use the thumbs up/down buttons in the "
                      "web UI first.[/yellow]")
        return
    drafts = feedback_to_candidates(rated)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Draft questions from answers rated in the web UI.\n"
              "# Complete gold_docs, evidence and answer_facts from the source document,\n"
              "# then move finished items into eval/questions.yaml.\n")
    target.write_text(header + yaml.safe_dump({"items": drafts}, sort_keys=False,
                                              allow_unicode=True, width=100), encoding="utf-8")
    down = sum(1 for r in rated if r["rating"] == "down")
    console.print(f"[green]Wrote {len(drafts)} drafts[/green] ({down} rated not helpful) to {target}")


if __name__ == "__main__":
    app()
