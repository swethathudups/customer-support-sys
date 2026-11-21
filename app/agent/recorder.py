"""In-memory collector for a run's structured execution events.

Steps and tool calls are buffered during the run and flushed in one transaction
when it ends (see AgentService). Everything passes through redaction first, so
the persisted trace is audit metadata - never raw records or hidden reasoning.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from app.agent.redact import mask_text, to_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunRecorder:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.steps: list[dict] = []
        self.tool_calls: list[dict] = []
        self._t0 = time.monotonic()

    def step(self, type_: str, name: str, summary: str, *, status: str = "ok",
             data: Any = None, duration_ms: int = 0) -> int:
        seq = len(self.steps) + 1
        self.steps.append({
            "seq": seq, "type": type_, "name": name, "status": status,
            "summary": mask_text(summary)[:500], "data_json": to_json(data or {}),
            "started_at": now_iso(), "duration_ms": duration_ms,
        })
        return seq

    def tool_call(self, *, tool_name: str, method: str | None, path: str | None,
                  is_write: bool, status: str, http_status: int | None,
                  error_code: str | None, request: Any, response: Any,
                  idempotency_key: str | None, attempts: int, duration_ms: int) -> int:
        seq = len(self.tool_calls) + 1
        self.tool_calls.append({
            "seq": seq, "tool_name": tool_name, "method": method, "path": path,
            "is_write": is_write, "status": status, "http_status": http_status,
            "error_code": error_code, "request_json": to_json(request),
            "response_summary": to_json(response), "idempotency_key": idempotency_key,
            "attempts": attempts, "started_at": now_iso(), "duration_ms": duration_ms,
        })
        return seq

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)
