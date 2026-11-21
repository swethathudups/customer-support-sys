"""Tool executor: the single choke point between the agent and the Tool APIs.

Every call passes the same gates, in order:
  1. tool exists                       2. arguments validate (Pydantic)
  3. order_number resolves to one of THIS customer's orders
  4. write tools: confirmation granted + prerequisite read already succeeded
  5. HTTP call as the agent principal, scoped to the customer (API re-checks ownership)
Each call is recorded (redacted) for the audit trace.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agent import kb
from app.agent.recorder import RunRecorder
from app.agent.tools.client import ToolApiClient
from app.agent.tools.registry import TOOLS, ToolSpec


@dataclass
class ToolResult:
    tool: str
    args: dict
    ok: bool
    data: Any = None
    error_code: str | None = None
    message: str | None = None
    blocked: bool = False          # stopped by a local gate; never reached the API
    order_number: str | None = None

    def for_llm(self) -> dict:
        """Compact JSON the LLM sees as the tool_result."""
        if self.ok:
            return {"ok": True, "data": self.data}
        return {"ok": False, "error_code": self.error_code, "message": self.message}


class ToolExecutor:
    def __init__(self, client: ToolApiClient, customer_id: int, recorder: RunRecorder) -> None:
        self._client = client
        self._customer_id = customer_id
        self._rec = recorder
        self._orders: dict[str, int] | None = None   # order_number -> order_id (this customer only)
        self.succeeded: set[tuple[str, str | None]] = set()   # (tool, order_number) that returned ok

    # ------------------------------------------------------------------
    async def execute(self, name: str, raw_args: dict | None, *, confirmed: bool = False,
                      idempotency_key: str | None = None) -> ToolResult:
        raw_args = raw_args or {}
        spec = TOOLS.get(name)
        if spec is None:
            return self._blocked(name, raw_args, "unknown_tool", f"No such tool: {name}")
        try:
            args = spec.input_model(**raw_args)
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:4])
            return self._blocked(name, raw_args, "invalid_arguments", msg, spec)
        arg_dict = args.model_dump()
        order_number = arg_dict.get("order_number")

        if spec.local:
            return self._run_local(spec, arg_dict)

        # Gate: prerequisite read must have succeeded for this order.
        if spec.prerequisite and (spec.prerequisite, order_number) not in self.succeeded:
            return self._blocked(name, arg_dict, "prerequisite_missing",
                                 f"Call {spec.prerequisite} for this order first.", spec, order_number)

        path_vars: dict[str, Any] = {"customer_id": self._customer_id}
        if "{order_id}" in spec.path:
            order_id = await self._resolve_order(order_number)
            if order_id is None:
                return self._blocked(name, arg_dict, "order_not_found",
                                     f"No order {order_number} found on this account.", spec, order_number)
            path_vars["order_id"] = order_id
        # Gate: confirmation for risky writes (checked last so a 'blocked' here means every
        # other gate already passed - the graph uses that to stage a confirmation prompt).
        if spec.requires_confirmation and not confirmed:
            return self._blocked(name, arg_dict, "confirmation_required",
                                 "The customer has not confirmed this action.", spec, order_number)
        if "{product_id}" in spec.path:
            path_vars["product_id"] = arg_dict["product_id"]
        if "{ticket_number}" in spec.path:
            path_vars["ticket_number"] = arg_dict["ticket_number"]
        path = spec.path.format(**path_vars)

        params = {k: arg_dict[k] for k in spec.query_fields if arg_dict.get(k) is not None} or None
        body = spec.body_fn(args) if spec.body_fn else None
        return await self._call(spec, path, arg_dict, params, body, idempotency_key, order_number)

    # ------------------------------------------------------------------
    def _run_local(self, spec: ToolSpec, args: dict) -> ToolResult:
        t0 = time.monotonic()
        hits = kb.search(args["query"])
        ms = int((time.monotonic() - t0) * 1000)
        self._rec.step("retrieval", spec.name,
                       f"KB search returned {len(hits)} article(s): "
                       f"{', '.join(h['policy_id'] for h in hits) or 'none'}",
                       data={"query": args["query"], "results": [
                           {"policy_id": h["policy_id"], "path": h["path"], "score": h["score"]}
                           for h in hits]}, duration_ms=ms)
        return ToolResult(spec.name, args, True, data=hits)

    async def _call(self, spec: ToolSpec, path: str, args: dict, params: dict | None,
                    body: dict | None, idem: str | None, order_number: str | None) -> ToolResult:
        resp = await self._client.request(spec.method, path, customer_id=self._customer_id,
                                          params=params, json=body, idempotency_key=idem)
        if resp.ok:
            self.succeeded.add((spec.name, order_number))
        self._rec.tool_call(
            tool_name=spec.name, method=spec.method, path=_mask_path(path), is_write=spec.is_write,
            status="ok" if resp.ok else "error", http_status=resp.status_code,
            error_code=resp.error_code, request={"args": args, "body_keys": sorted(body or {})},
            response=resp.data if resp.ok else {"error_code": resp.error_code, "message": resp.error_message},
            idempotency_key=idem, attempts=resp.attempts, duration_ms=resp.duration_ms)
        if resp.ok:
            return ToolResult(spec.name, args, True, data=resp.data, order_number=order_number)
        return ToolResult(spec.name, args, False, error_code=resp.error_code,
                          message=resp.error_message, order_number=order_number)

    def _blocked(self, name: str, args: dict, code: str, message: str,
                 spec: ToolSpec | None = None, order_number: str | None = None) -> ToolResult:
        self._rec.tool_call(tool_name=name, method=spec.method if spec else None, path=None,
                            is_write=bool(spec and spec.is_write), status="blocked", http_status=None,
                            error_code=code, request={"args": args}, response={"message": message},
                            idempotency_key=None, attempts=0, duration_ms=0)
        self._rec.step("validation", name, f"Blocked before execution: {code} - {message}",
                       status="blocked", data={"code": code})
        return ToolResult(name, args, False, error_code=code, message=message, blocked=True,
                          order_number=order_number)

    async def _resolve_order(self, order_number: str | None) -> int | None:
        if not order_number:
            return None
        if self._orders is None:
            resp = await self._client.request("GET", f"/customers/{self._customer_id}/orders",
                                              customer_id=self._customer_id, params={"limit": 100})
            self._rec.tool_call(tool_name="list_customer_orders", method="GET",
                                path=f"/customers/{{customer_id}}/orders", is_write=False,
                                status="ok" if resp.ok else "error", http_status=resp.status_code,
                                error_code=resp.error_code, request={"purpose": "resolve_order_number"},
                                response={"count": len(resp.data["items"])} if resp.ok else {},
                                idempotency_key=None, attempts=resp.attempts,
                                duration_ms=resp.duration_ms)
            if not resp.ok:
                return None
            self._orders = {o["order_number"]: o["order_id"] for o in resp.data["items"]}
        return self._orders.get(order_number)


def _mask_path(path: str) -> str:
    """Keep the endpoint shape, drop concrete ids from the audit trail."""
    parts = [("{id}" if p.isdigit() else p) for p in path.split("/")]
    return "/".join(parts)

