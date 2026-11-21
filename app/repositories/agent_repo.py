"""Persistence for agent sessions, messages, runs, steps and tool calls (SQLAlchemy 2.0 ORM)."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.recorder import now_iso
from app.db.agent_models import AgentMessage, AgentRun, AgentSession, AgentStep, AgentToolCall


class AgentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ---- sessions / messages ----
    async def get_session(self, session_id: str) -> AgentSession | None:
        return await self.db.get(AgentSession, session_id)

    async def create_session(self, session_id: str, customer_id: int) -> AgentSession:
        now = now_iso()
        s = AgentSession(session_id=session_id, customer_id=customer_id, created_at=now,
                         updated_at=now, state_json="{}")
        self.db.add(s)
        await self.db.flush()
        return s

    async def recent_messages(self, session_id: str, limit: int) -> list[AgentMessage]:
        rows = (await self.db.execute(
            select(AgentMessage).where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.id.desc()).limit(limit))).scalars().all()
        return list(reversed(rows))

    async def list_messages(self, session_id: str, limit: int, offset: int) -> tuple[list[AgentMessage], int]:
        q = select(AgentMessage).where(AgentMessage.session_id == session_id)
        total = (await self.db.execute(
            select(func.count()).select_from(q.subquery()))).scalar_one()
        rows = (await self.db.execute(q.order_by(AgentMessage.id.asc()).limit(limit).offset(offset))
                ).scalars().all()
        return list(rows), total

    async def counts(self, session_id: str) -> tuple[int, int]:
        m = (await self.db.execute(select(func.count()).select_from(AgentMessage)
                                   .where(AgentMessage.session_id == session_id))).scalar_one()
        r = (await self.db.execute(select(func.count()).select_from(AgentRun)
                                   .where(AgentRun.session_id == session_id))).scalar_one()
        return m, r

    # ---- runs ----
    async def create_run(self, run: AgentRun) -> None:
        self.db.add(run)
        await self.db.commit()   # visible immediately as status=running; releases the connection

    async def get_run(self, run_id: str) -> AgentRun | None:
        return await self.db.get(AgentRun, run_id)

    async def list_runs(self, session_id: str, limit: int, offset: int) -> tuple[list[AgentRun], int]:
        q = select(AgentRun).where(AgentRun.session_id == session_id)
        total = (await self.db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        rows = (await self.db.execute(q.order_by(AgentRun.started_at.desc()).limit(limit).offset(offset))
                ).scalars().all()
        return list(rows), total

    async def steps(self, run_id: str) -> list[AgentStep]:
        return list((await self.db.execute(
            select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.seq))).scalars().all())

    async def tool_calls(self, run_id: str) -> list[AgentToolCall]:
        return list((await self.db.execute(
            select(AgentToolCall).where(AgentToolCall.run_id == run_id).order_by(AgentToolCall.seq))
        ).scalars().all())

    # ---- one-shot flush at the end of a run ----
    async def finish_run(self, *, run: AgentRun, session: AgentSession, ctx: dict, human_text: str,
                         ai_text: str, steps: list[dict], tool_calls: list[dict],
                         **run_fields: Any) -> None:
        for k, v in run_fields.items():
            setattr(run, k, v)
        run.step_count, run.tool_call_count = len(steps), len(tool_calls)
        self.db.add(run)
        for s in steps:
            self.db.add(AgentStep(run_id=run.run_id, **s))
        for t in tool_calls:
            self.db.add(AgentToolCall(run_id=run.run_id, **t))
        now = now_iso()
        self.db.add(AgentMessage(session_id=session.session_id, run_id=run.run_id, role="human",
                                 content=human_text, created_at=run.started_at))
        self.db.add(AgentMessage(session_id=session.session_id, run_id=run.run_id, role="ai",
                                 content=ai_text, created_at=now))
        session.state_json = json.dumps(ctx, default=str)
        session.updated_at = now
        self.db.add(session)
        await self.db.commit()
