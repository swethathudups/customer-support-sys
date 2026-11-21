"""Environment-based application configuration (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[2] == project root (holds _KB/ and _DB/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "Customer Support API"
    env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = True
    api_v1_prefix: str = "/api/v1"

    # Database — default: the repo's seeded SQLite DB
    database_url: str = ""

    # Auth
    api_keys: str = "public:dev-public-key,agent:dev-agent-key"
    agent_service_key: str = "dev-agent-key"

    # Rate limiting
    rate_limit_per_minute: int = 120

    # External dependency resilience
    http_timeout_seconds: float = 10.0
    http_max_retries: int = 2
    circuit_breaker_fail_threshold: int = 5
    circuit_breaker_reset_seconds: float = 30.0

    # Agent / LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 1024
    agent_max_steps: int = 8
    agent_refund_approval_limit_cents: int = 10000  # $100.00
    agent_planner: str = "auto"                 # auto | llm | rules  (auto: llm iff ANTHROPIC_API_KEY set)
    agent_tool_api_base_url: str = ""           # empty -> call the in-process app over ASGI
    agent_run_timeout_seconds: float = 90.0
    agent_pending_ttl_minutes: int = 15         # how long a "please confirm" stays actionable
    agent_history_messages: int = 12            # prior messages given to the planner
    agent_kb_top_k: int = 3
    llm_refusal_fallback: bool = True           # server-side fallback on safety refusals

    # Paths (not overridable via env; derived)
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    @property
    def kb_dir(self) -> Path:
        return self.project_root / "_KB"

    @property
    def db_file(self) -> Path:
        return self.project_root / "_DB" / "ecommerce.db"

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        # Absolute path -> sqlite+aiosqlite:////abs/path (note 4 slashes for abs).
        return f"sqlite+aiosqlite:///{self.db_file}"

    @property
    def is_sqlite(self) -> bool:
        return self.effective_database_url.startswith("sqlite")

    def api_key_map(self) -> dict[str, str]:
        """Return {key -> principal_name} for all configured API keys."""
        out: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            name, key = pair.split(":", 1)
            out[key.strip()] = name.strip()
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
