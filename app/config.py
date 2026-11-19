import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Paths are anchored to the project folder, so the app works from any working directory.
ROOT = Path(__file__).resolve().parent.parent

# Vercel sets VERCEL=1. Its disk is read-only except /tmp, and /tmp is wiped when an
# instance stops. So there the vector index lives in memory (it is rebuilt from the
# prebuilt vectors at startup) and the SQLite file goes to /tmp.
ON_VERCEL = bool(os.environ.get("VERCEL"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Any OpenAI-compatible chat endpoint works. Groq is the default.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = Field("", validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY"))
    # Tried in order: if the first model keeps failing, the next one is used.
    llm_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b"

    # Empty QDRANT_URL means an on-disk local index at QDRANT_PATH.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    # ":memory:" keeps the index in memory instead of on disk.
    qdrant_path: str = ":memory:" if ON_VERCEL else str(ROOT / "data" / "qdrant")

    db_path: str = "/tmp/app.db" if ON_VERCEL else str(ROOT / "data" / "app.db")
    jobs_path: str = str(ROOT / "data" / "jobs.json")
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Written by `python -m app.prebuild`. Both are optional: without them the model is
    # downloaded on first use and every job is embedded at startup.
    vectors_path: str = str(ROOT / "data" / "vectors.json")
    model_dir: str = str(ROOT / "data" / "model")

    # If set, POST /api/refresh requires the header "X-Admin-Token".
    admin_token: str = ""

    @property
    def models(self) -> list[str]:
        return [m.strip() for m in self.llm_models.split(",") if m.strip()]


settings = Settings()
