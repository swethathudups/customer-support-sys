"""Schemas for the Agentic APIs (chat + observability)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RunStatus = Literal["running", "completed", "awaiting_confirmation", "escalated", "failed"]


class ChatRequest(BaseModel):
    customer_id: int = Field(..., ge=1, examples=[1], description="Customer the chat is for.")
    session_id: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{1,64}$", examples=["SESSION1001"])
    message: str = Field(..., min_length=1, max_length=2000, examples=["Where is my latest order?"])

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class PendingConfirmation(BaseModel):
    action_id: str
    tool: str
    order_number: str | None = None
    description: str
    risk: str
    instructions: str = 'Reply "yes" to confirm or "no" to cancel.'


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    message: str
    intent: str | None = None
    status: RunStatus
    escalated: bool = False
    ticket_number: str | None = None
    pending_confirmation: PendingConfirmation | None = None
    tools_used: list[str] = []
    citations: list[str] = []
    policy_ids: list[str] = []
    planner: str
    duration_ms: int
    request_id: str | None = None


class SessionOut(BaseModel):
    session_id: str
    customer_id: int
    created_at: str
    updated_at: str
    message_count: int
    run_count: int
    awaiting_confirmation: bool


class MessageOut(BaseModel):
    message_id: int
    role: Literal["human", "ai"]
    content: str
    run_id: str | None = None
    created_at: str


class StepOut(BaseModel):
    seq: int
    type: str = Field(..., description="guardrail|intent|decision|retrieval|tool_call|validation|"
                                       "confirmation|escalation|response|error")
    name: str
    status: str
    summary: str
    data: dict[str, Any] = {}
    started_at: str
    duration_ms: int


class ToolCallOut(BaseModel):
    seq: int
    tool_name: str
    method: str | None = None
    path: str | None = None
    is_write: bool
    status: str
    http_status: int | None = None
    error_code: str | None = None
    request: dict[str, Any] = {}
    response_summary: dict[str, Any] = {}
    idempotency_key: str | None = None
    attempts: int
    started_at: str
    duration_ms: int


class RunOut(BaseModel):
    run_id: str
    session_id: str
    status: RunStatus
    intent: str | None = None
    planner: str | None = None
    user_message: str
    final_response: str | None = None
    escalated: bool
    error_code: str | None = None
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    step_count: int
    tool_call_count: int
    request_id: str | None = None


class RunTraceOut(RunOut):
    steps: list[StepOut]
    tool_calls: list[ToolCallOut]


class DependencyHealth(BaseModel):
    name: str
    status: Literal["up", "down", "configured", "not_configured"]
    detail: str | None = None


class AgentHealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    planner: str
    llm_model: str | None = None
    dependencies: list[DependencyHealth]
