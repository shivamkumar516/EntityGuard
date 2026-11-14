"""EntityGuard web application: FastAPI backend (Section 47) + Gradio interface at /ui (Sections 40, 46).

Run locally from the repo root:
    uvicorn src.api.app:api --reload
then open http://localhost:8000/ui  (interface)  or  http://localhost:8000/docs  (API docs).

It needs only saved artifacts in outputs/ (see src/api/store.py). Another folder can be used
with the environment variable EG_OUT_DIR.

Routes (Section 47):
    GET  /records/dblp            GET  /records/scholar
    POST /blocking/run            POST /matching/run
    GET  /matches                 GET  /pair/{dblp_id}/{scholar_id}
    POST /review                  GET  /evaluation
DBLP ids always have three parts (conf/vldb/Name), so /pair/conf/vldb/Name/<scholar id> is
split after the third part; Scholar ids may themselves contain "/". The same lookup is also
available as GET /pair?dblp_id=...&scholar_id=... .
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .store import DEFAULT_OUT_DIR, Store


class Review(BaseModel):
    dblp_id: str
    scholar_id: str
    decision: str
    note: str = ""
    reviewer: str = "api"


def create_app(out_dir=None, review_csv=None, with_ui=True):
    store = Store(out_dir or os.environ.get("EG_OUT_DIR", DEFAULT_OUT_DIR), review_csv=review_csv)
    app = FastAPI(title="EntityGuard API",
                  description="Entity resolution between DBLP and Google Scholar. Interface at /ui.")
    app.state.store = store

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui" if with_ui else "/docs")

    @app.get("/records/dblp")
    def records_dblp(offset: int = 0, limit: int = Query(20, le=500)):
        return store.records("dblp", offset, limit)

    @app.get("/records/scholar")
    def records_scholar(offset: int = 0, limit: int = Query(20, le=500)):
        return store.records("scholar", offset, limit)

    @app.post("/blocking/run")
    def blocking_run(scholar_fraction: float = 1.0, passes: str = ""):
        """Re-run the reference blocking. passes: comma-separated subset of
        title_prefix,rare_tokens,char_prefix,rare_char6 (default: all)."""
        try:
            return store.run_blocking(scholar_fraction, [p for p in passes.split(",") if p] or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(503, f"dataset files not available: {e}")

    @app.post("/matching/run")
    def matching_run(threshold: float = None):
        try:
            return store.run_matching(store.cfg.threshold if threshold is None else threshold)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/matches")
    def matches(band_name: str = None, limit: int = Query(50, le=1000), min_score: float = None):
        return store.matches(band_name, limit, min_score=min_score).astype(object).where(lambda d: d.notna(), None) \
            .to_dict("records")

    def _pair(dblp_id, scholar_id):
        p = store.pair(dblp_id, scholar_id)
        if p is None:
            raise HTTPException(404, "pair not found: unknown dblp_id or scholar_id")
        return p

    @app.get("/pair")
    def pair_query(dblp_id: str, scholar_id: str):
        return _pair(dblp_id, scholar_id)

    @app.get("/pair/{ids:path}")
    def pair_path(ids: str):
        parts = ids.split("/")
        if len(parts) < 4:
            raise HTTPException(404, "use /pair/<dblp id with 3 parts>/<scholar id>")
        return _pair("/".join(parts[:3]), "/".join(parts[3:]))

    @app.post("/review")
    def review(r: Review):
        try:
            return {"saved": True, **store.save_review(r.dblp_id, r.scholar_id, r.decision, r.note, r.reviewer)}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/evaluation")
    def evaluation():
        return store.evaluation()

    if with_ui:
        import gradio as gr
        from .ui import build_ui
        app = gr.mount_gradio_app(app, build_ui(store), path="/ui")
    return app


def _create_default():
    try:
        return create_app()
    except FileNotFoundError as e:          # still importable before the artifacts exist
        missing = FastAPI(title="EntityGuard API (artifacts missing)")

        @missing.get("/")
        def info():
            return {"error": str(e)}
        return missing


api = _create_default()
