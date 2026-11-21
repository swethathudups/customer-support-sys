"""HTTP client the agent uses to call the Tool APIs.

* Authenticates as the *agent service principal* and scopes every call to the
  chat customer via X-Customer-Id - the tool layer then enforces ownership, so
  the agent can never read another customer's data.
* Timeouts, bounded retries with backoff (reads always; writes only when an
  Idempotency-Key makes replay safe) and a shared circuit breaker.
* Default transport is in-process ASGI (no network hop); set
  AGENT_TOOL_API_BASE_URL to call a remote deployment instead.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings
from app.core.correlation import REQUEST_ID_HEADER
from app.core.logging import get_logger
from app.core.resilience import CircuitBreaker
from app.core.security import API_KEY_HEADER, CUSTOMER_ID_HEADER

logger = get_logger("app.agent.client")

BREAKER = CircuitBreaker(settings.circuit_breaker_fail_threshold,
                         settings.circuit_breaker_reset_seconds)


@dataclass
class ApiResponse:
    ok: bool
    status_code: int | None
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = 1
    duration_ms: int = 0
    meta: dict = field(default_factory=dict)


class ToolApiClient:
    """Async context manager; one per agent run."""

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ToolApiClient":
        timeout = httpx.Timeout(settings.http_timeout_seconds)
        if settings.agent_tool_api_base_url:
            self._client = httpx.AsyncClient(base_url=settings.agent_tool_api_base_url, timeout=timeout)
        else:
            from app.main import app  # lazy: avoid import cycle
            self._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url="http://agent-tools", timeout=timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    async def request(self, method: str, path: str, *, customer_id: int,
                      params: dict | None = None, json: dict | None = None,
                      idempotency_key: str | None = None) -> ApiResponse:
        assert self._client is not None
        headers = {API_KEY_HEADER: settings.agent_service_key,
                   CUSTOMER_ID_HEADER: str(customer_id),
                   REQUEST_ID_HEADER: self._request_id}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if not BREAKER.allow():
            return ApiResponse(False, None, error_code="circuit_open",
                               error_message="Tool API temporarily unavailable.")
        retryable = method == "GET" or bool(idempotency_key)
        max_attempts = 1 + (settings.http_max_retries if retryable else 0)
        t0 = time.monotonic()
        last: ApiResponse | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                url = f"{settings.api_v1_prefix}{path}"
                r = await self._client.request(method, url, params=params, json=json, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ApiResponse(False, None, error_code="upstream_unreachable",
                                   error_message=type(exc).__name__, attempts=attempt)
                BREAKER.record_failure()
            else:
                if r.status_code >= 500:
                    last = ApiResponse(False, r.status_code, error_code="upstream_error",
                                       error_message=f"HTTP {r.status_code}", attempts=attempt)
                    BREAKER.record_failure()
                else:
                    BREAKER.record_success()
                    last = self._parse(r, attempt)
                    break
            if attempt < max_attempts:
                await asyncio.sleep(0.2 * 2 ** (attempt - 1))
        assert last is not None
        last.duration_ms = int((time.monotonic() - t0) * 1000)
        return last

    @staticmethod
    def _parse(r: httpx.Response, attempts: int) -> ApiResponse:
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.is_success:
            return ApiResponse(True, r.status_code, data=body, attempts=attempts)
        err = (body or {}).get("error", {}) if isinstance(body, dict) else {}
        return ApiResponse(False, r.status_code, error_code=err.get("code", "http_error"),
                           error_message=err.get("message", f"HTTP {r.status_code}"),
                           attempts=attempts, data=err.get("details"))


async def probe_tool_api() -> bool:
    """Unauthenticated liveness probe of the Tool API (used by /agent/health)."""
    try:
        async with ToolApiClient("health-probe") as c:
            assert c._client is not None
            r = await c._client.get("/health")
            return r.status_code == 200
    except Exception:
        return False
