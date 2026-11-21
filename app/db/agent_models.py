"""ORM models for the agent's own state (sessions, messages, runs, steps, tool calls).

Kept separate from the commerce schema in _DB/. Created idempotently at startup
(`ensure_agent_schema`) and portable to PostgreSQL. The agent never writes to
commerce tables directly — only through the Tool APIs.
"""
from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Text
from sqlalchemy import String as Str
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AgentSession(Base):
    __tablename__ = "agent_sessions"
    session_id: Mapped[str] = mapped_column(Str(64), primary_key=True)
    customer_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[str] = mapped_column(Str(40))
    updated_at: Mapped[str] = mapped_column(Str(40))
    # JSON: pending_action, last_order_number, clarifications
    state_json: Mapped[str] = mapped_column(Text, default="{}")


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("agent_sessions.session_id"), index=True)
    run_id: Mapped[str | None] = mapped_column(Str(40), nullable=True)
    role: Mapped[str] = mapped_column(Str(16))  # human | ai
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Str(40))


class AgentRun(Base):
    __tablename__ = "agent_runs"
    run_id: Mapped[str] = mapped_column(Str(40), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("agent_sessions.session_id"), index=True)
    customer_id: Mapped[int] = mapped_column(Integer, index=True)
    request_id: Mapped[str | None] = mapped_column(Str(64), nullable=True)
    status: Mapped[str] = mapped_column(Str(24))  # running|completed|awaiting_confirmation|escalated|failed
    intent: Mapped[str | None] = mapped_column(Str(48), nullable=True)
    planner: Mapped[str | None] = mapped_column(Str(24), nullable=True)
    user_message: Mapped[str] = mapped_column(Text)
    final_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(Str(48), nullable=True)
    started_at: Mapped[str] = mapped_column(Str(40))
    finished_at: Mapped[str | None] = mapped_column(Str(40), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    meta_json: Mapped[str] = mapped_column(Text, default="{}")


class AgentStep(Base):
    __tablename__ = "agent_steps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(Str(24))
    name: Mapped[str] = mapped_column(Str(64))
    status: Mapped[str] = mapped_column(Str(16))  # ok | blocked | error
    summary: Mapped[str] = mapped_column(Text)
    data_json: Mapped[str] = mapped_column(Text, default="{}")
    started_at: Mapped[str] = mapped_column(Str(40))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


class AgentToolCall(Base):
    __tablename__ = "agent_tool_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(Str(64))
    method: Mapped[str | None] = mapped_column(Str(8), nullable=True)
    path: Mapped[str | None] = mapped_column(Str(200), nullable=True)
    is_write: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(Str(16))  # ok | error | blocked
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Str(48), nullable=True)
    request_json: Mapped[str] = mapped_column(Text, default="{}")        # redacted
    response_summary: Mapped[str] = mapped_column(Text, default="{}")    # redacted + truncated
    idempotency_key: Mapped[str | None] = mapped_column(Str(100), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[str] = mapped_column(Str(40))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


Index("ix_agent_runs_session_started", AgentRun.session_id, AgentRun.started_at)

_ready_for: AsyncEngine | None = None  # strong ref: id() can be reused after dispose


async def ensure_agent_schema(engine: AsyncEngine) -> None:
    """Create the agent tables if missing (idempotent; once per engine)."""
    global _ready_for
    if _ready_for is engine:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _ready_for = engine
