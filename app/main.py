import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .config import ON_VERCEL, settings
from .embed import CachedEmbedder, FastEmbedder
from .index import JobIndex
from .ingest import load_jobs, save_jobs
from .llm import LLM, ChatLLM
from .memory import Memory
from .profile import extract_profile
from .rank import match_jobs
from .sources import fetch_all

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
REFRESH_INTERVAL_SECONDS = 30 * 60


class ProfileRequest(BaseModel):
    session_id: str
    resume_text: str = Field(min_length=30, max_length=20000)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class FeedbackRequest(BaseModel):
    session_id: str
    job_id: str
    verdict: str = Field(pattern="^(saved|dismissed)$")


def _build_index() -> JobIndex:
    from qdrant_client import QdrantClient

    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    elif settings.qdrant_path == ":memory:":
        client = QdrantClient(":memory:")
    else:
        Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=settings.qdrant_path)
    embedder = FastEmbedder(settings.embed_model, settings.model_dir)
    if Path(settings.vectors_path).exists():
        # Vectors made at build time: startup then needs no model run for known jobs.
        embedder = CachedEmbedder.from_file(embedder, settings.vectors_path)
    return JobIndex(embedder, client)


def create_app(llm: LLM | None = None, index: JobIndex | None = None, memory: Memory | None = None) -> FastAPI:
    """Pass llm/index/memory to inject test doubles; otherwise they are built from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.memory = memory or Memory(settings.db_path)
        app.state.llm = llm
        if llm is None and settings.llm_api_key:
            app.state.llm = ChatLLM(settings.llm_base_url, settings.llm_api_key, settings.models)
        app.state.index = index or _build_index()
        app.state.last_refresh = 0.0
        app.state.last_sync = None
        if index is None and Path(settings.jobs_path).exists():
            jobs, complete = load_jobs(settings.jobs_path)
            app.state.last_sync = app.state.index.sync(jobs, complete)
        yield

    app = FastAPI(title="Job Match Agent", lifespan=lifespan)

    def session(session_id: str) -> str:
        if not app.state.memory.session_exists(session_id):
            raise HTTPException(404, "unknown session")
        return session_id

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "llm": app.state.llm is not None, "jobs": app.state.index.count()}

    @app.get("/api/stats")
    def stats():
        return {
            "total": app.state.index.count(),
            "by_source": app.state.index.sources(),
            "last_sync": app.state.last_sync,
        }

    @app.post("/api/session")
    def new_session():
        return {"session_id": app.state.memory.create_session()}

    @app.post("/api/profile")
    def profile(body: ProfileRequest):
        sid = session(body.session_id)
        result = extract_profile(app.state.llm, body.resume_text)
        app.state.memory.save_profile(sid, result)  # the resume text itself is not stored
        return result

    @app.get("/api/matches")
    def matches(session_id: str, k: int = 8):
        sid = session(session_id)
        found_profile = app.state.memory.get_profile(sid)
        if found_profile is None:
            raise HTTPException(400, "send a resume to /api/profile first")
        prefs = app.state.memory.get_preferences(sid)
        results = match_jobs(
            found_profile,
            prefs,
            app.state.index,
            app.state.llm,
            k=max(1, min(k, 20)),
            exclude_ids=app.state.memory.job_ids(sid, "dismissed"),
        )
        return {"preferences": prefs, "matches": results}

    @app.post("/api/chat")
    def chat(body: ChatRequest):
        sid = session(body.session_id)
        result = run_agent(app.state.llm, app.state.memory, app.state.index, sid, body.message)
        return {**result, "preferences": app.state.memory.get_preferences(sid)}

    @app.post("/api/feedback")
    def feedback(body: FeedbackRequest):
        sid = session(body.session_id)
        if app.state.index.get(body.job_id) is None:
            raise HTTPException(404, "unknown job")
        app.state.memory.set_feedback(sid, body.job_id, body.verdict)
        return {"ok": True}

    @app.post("/api/refresh")
    def refresh(x_admin_token: str = Header("")):
        """Re-download the job sources and sync the index. At most once every 30 minutes."""
        if settings.admin_token and x_admin_token != settings.admin_token:
            raise HTTPException(403, "admin token required")
        if ON_VERCEL and not settings.admin_token:
            # A refresh there would only change one short-lived instance's memory.
            raise HTTPException(403, "refresh is off on this deployment; the index is rebuilt at each deploy")
        waited = time.monotonic() - app.state.last_refresh
        if app.state.last_refresh and waited < REFRESH_INTERVAL_SECONDS:
            raise HTTPException(429, f"refreshed {int(waited)}s ago; try again later")
        app.state.last_refresh = time.monotonic()
        jobs, complete, errors = fetch_all()
        app.state.last_sync = app.state.index.sync(jobs, complete)
        if jobs and not ON_VERCEL:  # the disk is read-only there
            save_jobs(settings.jobs_path, jobs, complete)
        return {"sync": app.state.last_sync, "errors": errors}

    return app


app = create_app()
