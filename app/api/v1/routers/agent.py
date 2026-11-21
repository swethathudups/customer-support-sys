"""Category 2 - Agentic APIs: the only surface the frontend uses.

The frontend talks to the agent here; the agent (not the frontend) calls the
Tool APIs. Observability endpoints expose structured events + audit metadata,
never chain-of-thought, credentials or raw customer records.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import AgentSvc, CurrentPrincipal
from app.core.correlation import get_request_id
from app.core.pagination import PageParams, page_params
from app.schemas.agent import (
    AgentHealthOut,
    ChatRequest,
    ChatResponse,
    MessageOut,
    RunOut,
    RunTraceOut,
    SessionOut,
    StepOut,
    ToolCallOut,
)
from app.schemas.common import Page, make_page

router = APIRouter(prefix="/agent", tags=["agent"])
Paging = Annotated[PageParams, Depends(page_params)]


@router.post("/chat", response_model=ChatResponse, summary="Send a message to the support agent")
async def chat(body: ChatRequest, svc: AgentSvc, principal: CurrentPrincipal) -> ChatResponse:
    """Runs one agent turn: guardrails -> intent -> tool/KB selection -> (confirmation) -> reply.

    Refund/return/cancel actions are never executed on a question: the agent first
    explains eligibility, then asks for explicit confirmation (`status=awaiting_confirmation`);
    the customer's next message ("yes"/"no") resolves it.
    """
    return await svc.chat(body, principal, get_request_id())


@router.get("/health", response_model=AgentHealthOut, summary="Agent + dependency health")
async def agent_health(svc: AgentSvc) -> AgentHealthOut:
    return await svc.health()


@router.get("/sessions/{session_id}", response_model=SessionOut, summary="Session info")
async def get_session(session_id: str, svc: AgentSvc, principal: CurrentPrincipal) -> SessionOut:
    return await svc.get_session(session_id, principal)


@router.get("/sessions/{session_id}/messages", response_model=Page[MessageOut],
            summary="Conversation history (oldest first)")
async def session_messages(session_id: str, svc: AgentSvc, principal: CurrentPrincipal,
                           page: Paging) -> Page[MessageOut]:
    items, total = await svc.list_messages(session_id, principal, page.limit, page.offset)
    return make_page(items, total, page.limit, page.offset)


@router.get("/sessions/{session_id}/traces", response_model=Page[RunTraceOut],
            summary="Execution traces for every run in the session (newest first)")
async def session_traces(session_id: str, svc: AgentSvc, principal: CurrentPrincipal,
                         page: Paging) -> Page[RunTraceOut]:
    items, total = await svc.list_traces(session_id, principal, page.limit, page.offset)
    return make_page(items, total, page.limit, page.offset)


@router.get("/runs/{run_id}", response_model=RunOut, summary="Run status + metadata")
async def get_run(run_id: str, svc: AgentSvc, principal: CurrentPrincipal) -> RunOut:
    return await svc.get_run(run_id, principal)


@router.get("/runs/{run_id}/steps", response_model=list[StepOut], summary="Structured execution steps")
async def run_steps(run_id: str, svc: AgentSvc, principal: CurrentPrincipal) -> list[StepOut]:
    return await svc.get_run_steps(run_id, principal)


@router.get("/runs/{run_id}/tools", response_model=list[ToolCallOut], summary="Tool calls made in the run")
async def run_tools(run_id: str, svc: AgentSvc, principal: CurrentPrincipal) -> list[ToolCallOut]:
    return await svc.get_run_tools(run_id, principal)
