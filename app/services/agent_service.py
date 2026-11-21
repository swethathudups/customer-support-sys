"""Agent application service: runs a chat turn end-to-end and serves the trace.

Responsibilities (kept out of routers and out of the graph):
  * authorization (caller must be allowed to act for the customer / see the session)
  * session + history management, run lifecycle, one-shot trace persistence
  * choosing the planner, enforcing the run timeout, safe failure handling
"""
from __future__ import annotations

import asyncio
import json
import uuid

from app.agent.graph import AgentDeps, AgentState, build_graph
from app.agent.kb import article_count
from app.agent.planner.base import Planner
from app.agent.planner.rules import RuleBasedPlanner
from app.agent.recorder import RunRecorder, now_iso
from app.agent.redact import mask_text
from app.agent.tools.client import BREAKER, ToolApiClient, probe_tool_api
from app.agent.tools.executor import ToolExecutor
from app.core.config import settings
from app.core.exceptions import ForbiddenError, NotFoundError, UpstreamError
from app.core.logging import get_logger
from app.core.security import Principal, ensure_owns_customer
from app.db.agent_models import AgentRun, AgentSession
from app.db.session import ping
from app.repositories.agent_repo import AgentRepository
from app.schemas.agent import (
    AgentHealthOut,
    ChatRequest,
    ChatResponse,
    DependencyHealth,
    MessageOut,
    PendingConfirmation,
    RunOut,
    RunTraceOut,
    SessionOut,
    StepOut,
    ToolCallOut,
)

logger = get_logger("app.agent.service")

_SAFE_FAILURE = ("I'm sorry - something went wrong on my side and I couldn't finish that. "
                 "Nothing was changed. Please try again, or ask for a human to take over.")


def select_planner() -> Planner:
    mode = settings.agent_planner.lower()
    if mode == "llm" or (mode == "auto" and settings.anthropic_api_key):
        from app.agent.planner.llm import LLMPlanner  # lazy: only needs the SDK in LLM mode
        return LLMPlanner()
    return RuleBasedPlanner()


def _run_out(r: AgentRun) -> dict:
    return dict(run_id=r.run_id, session_id=r.session_id, status=r.status, intent=r.intent,
                planner=r.planner, user_message=r.user_message, final_response=r.final_response,
                escalated=r.escalated, error_code=r.error_code, started_at=r.started_at,
                finished_at=r.finished_at, duration_ms=r.duration_ms, step_count=r.step_count,
                tool_call_count=r.tool_call_count, request_id=r.request_id)


class AgentService:
    def __init__(self, repo: AgentRepository, planner: Planner | None = None) -> None:
        self.repo = repo
        self._planner = planner

    # ------------------------------------------------------------------ chat
    async def chat(self, req: ChatRequest, principal: Principal, request_id: str) -> ChatResponse:
        ensure_owns_customer(principal, req.customer_id)
        planner = self._planner or select_planner()
        session = await self.repo.get_session(req.session_id)
        if session and session.customer_id != req.customer_id:
            raise ForbiddenError("This session belongs to a different customer.", code="ownership_violation")

        run_id = f"run_{uuid.uuid4().hex[:16]}"
        rec = RunRecorder(run_id)
        human_text = mask_text(req.message)

        async with ToolApiClient(request_id) as client:
            executor = ToolExecutor(client, req.customer_id, rec)
            if session is None:
                # New session: make sure the customer really exists (through the Tool API).
                probe = await client.request("GET", f"/customers/{req.customer_id}",
                                             customer_id=req.customer_id)
                if probe.status_code == 404:
                    raise NotFoundError("Customer not found.", code="customer_not_found")
                if not probe.ok:
                    raise UpstreamError("The support backend is unavailable right now; please retry.")
                session = await self.repo.create_session(req.session_id, req.customer_id)

            ctx = json.loads(session.state_json or "{}")
            history = [{"role": m.role, "content": m.content} for m in
                       await self.repo.recent_messages(req.session_id, settings.agent_history_messages)]

            run = AgentRun(run_id=run_id, session_id=req.session_id, customer_id=req.customer_id,
                           request_id=request_id, status="running", planner=planner.name,
                           user_message=human_text, started_at=now_iso(), escalated=False,
                           step_count=0, tool_call_count=0)
            await self.repo.create_run(run)
            rec.step("intake", "chat_request", f"Received message ({len(human_text)} chars); "
                     f"planner={planner.name}; history={len(history)} msgs",
                     data={"session_id": req.session_id, "has_pending_action": bool(ctx.get("pending_action"))})

            graph = build_graph(AgentDeps(planner, executor, rec))
            init: AgentState = dict(customer_id=req.customer_id, session_id=req.session_id, run_id=run_id,
                                    message=req.message, history=history, session_ctx=ctx,
                                    observations=[], llm_turns=[], steps_taken=0)
            error_code: str | None = None
            try:
                async with asyncio.timeout(settings.agent_run_timeout_seconds):
                    final = await graph.ainvoke(init, config={"recursion_limit": 60})
            except Exception as exc:  # never leak internals; the trace keeps the exception *type* only
                error_code = "run_timeout" if isinstance(exc, TimeoutError) else "agent_error"
                logger.exception("agent_run_failed", extra={"run_id": run_id, "error_code": error_code})
                rec.step("error", "run_failed", f"{error_code}: {type(exc).__name__}", status="error")
                final = {**init, "reply": _SAFE_FAILURE, "status": "failed"}

        status = final.get("status", "completed")
        reply = final.get("reply") or _SAFE_FAILURE
        new_ctx = dict(final.get("session_ctx", ctx))
        if final.get("pending_changed"):
            new_ctx["pending_action"] = final.get("pending_action")
        if final.get("last_order_number"):
            new_ctx["last_order_number"] = final["last_order_number"]
        pending = new_ctx.get("pending_action")

        duration = rec.elapsed_ms
        await self.repo.finish_run(
            run=run, session=session, ctx=new_ctx, human_text=human_text, ai_text=reply,
            steps=rec.steps, tool_calls=rec.tool_calls, status=status, intent=final.get("intent"),
            final_response=reply, escalated=bool(final.get("escalated")), error_code=error_code,
            finished_at=now_iso(), duration_ms=duration)
        logger.info("agent_run_finished", extra={
            "run_id": run_id, "session_id": req.session_id, "customer_id": req.customer_id,
            "status": status, "intent": final.get("intent"), "duration_ms": duration,
            "steps": len(rec.steps), "tool_calls": len(rec.tool_calls), "error_code": error_code})

        ticket = final.get("ticket_number") or next(
            (o["data"]["ticket_number"] for o in final.get("observations", [])
             if o["tool"] == "create_support_ticket" and o["ok"]), None)
        used = list(dict.fromkeys(t["tool_name"] for t in rec.tool_calls if t["status"] != "blocked"))
        used += [s["name"] for s in rec.steps if s["type"] == "retrieval" and s["name"] not in used]
        return ChatResponse(
            run_id=run_id, session_id=req.session_id, message=reply, intent=final.get("intent"),
            status=status, escalated=bool(final.get("escalated")), ticket_number=ticket,
            pending_confirmation=(PendingConfirmation(
                action_id=pending["id"], tool=pending["tool"], order_number=pending.get("order_number"),
                description=pending["description"], risk=pending["risk"])
                if status == "awaiting_confirmation" and pending else None),
            tools_used=used, citations=final.get("citations", []), policy_ids=final.get("policy_ids", []),
            planner=planner.name, duration_ms=duration, request_id=request_id)

    # ------------------------------------------------------------------ observability
    async def _owned_session(self, session_id: str, principal: Principal) -> AgentSession:
        s = await self.repo.get_session(session_id)
        if not s:
            raise NotFoundError("Session not found.", code="session_not_found")
        ensure_owns_customer(principal, s.customer_id)
        return s

    async def _owned_run(self, run_id: str, principal: Principal) -> AgentRun:
        r = await self.repo.get_run(run_id)
        if not r:
            raise NotFoundError("Run not found.", code="run_not_found")
        ensure_owns_customer(principal, r.customer_id)
        return r

    async def get_session(self, session_id: str, principal: Principal) -> SessionOut:
        s = await self._owned_session(session_id, principal)
        msgs, runs = await self.repo.counts(session_id)
        ctx = json.loads(s.state_json or "{}")
        return SessionOut(session_id=s.session_id, customer_id=s.customer_id, created_at=s.created_at,
                          updated_at=s.updated_at, message_count=msgs, run_count=runs,
                          awaiting_confirmation=bool(ctx.get("pending_action")))

    async def list_messages(self, session_id: str, principal: Principal, limit: int, offset: int):
        await self._owned_session(session_id, principal)
        rows, total = await self.repo.list_messages(session_id, limit, offset)
        return [MessageOut(message_id=m.id, role=m.role, content=m.content, run_id=m.run_id,
                           created_at=m.created_at) for m in rows], total

    @staticmethod
    def _steps(rows) -> list[StepOut]:
        return [StepOut(seq=s.seq, type=s.type, name=s.name, status=s.status, summary=s.summary,
                        data=json.loads(s.data_json), started_at=s.started_at, duration_ms=s.duration_ms)
                for s in rows]

    @staticmethod
    def _tools(rows) -> list[ToolCallOut]:
        return [ToolCallOut(seq=t.seq, tool_name=t.tool_name, method=t.method, path=t.path,
                            is_write=t.is_write, status=t.status, http_status=t.http_status,
                            error_code=t.error_code, request=json.loads(t.request_json),
                            response_summary=json.loads(t.response_summary),
                            idempotency_key=t.idempotency_key, attempts=t.attempts,
                            started_at=t.started_at, duration_ms=t.duration_ms) for t in rows]

    async def list_traces(self, session_id: str, principal: Principal, limit: int, offset: int):
        await self._owned_session(session_id, principal)
        runs, total = await self.repo.list_runs(session_id, limit, offset)
        out = [RunTraceOut(**_run_out(r), steps=self._steps(await self.repo.steps(r.run_id)),
                           tool_calls=self._tools(await self.repo.tool_calls(r.run_id))) for r in runs]
        return out, total

    async def get_run(self, run_id: str, principal: Principal) -> RunOut:
        return RunOut(**_run_out(await self._owned_run(run_id, principal)))

    async def get_run_steps(self, run_id: str, principal: Principal) -> list[StepOut]:
        await self._owned_run(run_id, principal)
        return self._steps(await self.repo.steps(run_id))

    async def get_run_tools(self, run_id: str, principal: Principal) -> list[ToolCallOut]:
        await self._owned_run(run_id, principal)
        return self._tools(await self.repo.tool_calls(run_id))

    # ------------------------------------------------------------------ health
    async def health(self) -> AgentHealthOut:
        mode = settings.agent_planner.lower()
        llm_on = mode == "llm" or (mode == "auto" and bool(settings.anthropic_api_key))
        db_ok, api_ok = await ping(), await probe_tool_api()
        deps = [
            DependencyHealth(name="database", status="up" if db_ok else "down"),
            DependencyHealth(name="tool_api", status="up" if api_ok else "down",
                             detail=f"circuit_breaker={BREAKER.state}"),
            DependencyHealth(name="knowledge_base", status="up" if article_count() else "down",
                             detail=f"{article_count()} articles"),
            DependencyHealth(name="llm", status="configured" if llm_on else "not_configured",
                             detail=settings.llm_model if llm_on else "offline rule-based planner active"),
        ]
        ok = db_ok and api_ok and article_count() > 0 and BREAKER.state != "open"
        return AgentHealthOut(status="ok" if ok else "degraded", planner="llm" if llm_on else "rules",
                              llm_model=settings.llm_model if llm_on else None, dependencies=deps)
