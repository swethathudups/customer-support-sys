"""Offline, deterministic planner (no LLM needed).

Not a fixed script per question: for each intent it inspects what has already
been observed this turn and picks the *next missing* step, branching on tool
results (e.g. only proposes a refund if the eligibility verdict allows it).
This keeps the whole system runnable/testable without an API key and doubles as
the reference behaviour the LLM planner is compared against.
"""
from __future__ import annotations

import re

from app.agent.intents import (
    extract_order_number,
    extract_ticket_number,
    mentions_latest,
    return_reason_from,
)
from app.agent.planner.base import Decision, ToolCallReq, kb_hits, last_obs

_INFRA_ERRORS = {"upstream_error", "upstream_unreachable", "circuit_open"}

ORDER_INTENTS = {"order_status", "order_tracking", "refund_question", "refund_request", "refund_status",
                 "return_question", "return_request", "cancel_question", "cancel_request",
                 "damaged_or_wrong_item", "payment_question", "delay_compensation"}


def _call(name: str, **args) -> Decision:
    return Decision("tool", calls=[ToolCallReq(name, args)], summary=f"Call {name}")


def _respond(text: str, summary: str, state: dict | None = None, **kw) -> Decision:
    text = re.sub(r"(\*\*\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)  # "...**ORD-1**. Shipped"
    d = Decision("respond", reply=text, summary=summary, **kw)
    if state is not None:
        hits = kb_hits(state["observations"])
        d.citations = list(dict.fromkeys(h["path"] for h in hits))
        d.policy_ids = list(dict.fromkeys(h["policy_id"] for h in hits))
    return d


def _money(m: dict | None) -> str:
    return m["display"] if m else "n/a"


def _summary_line(hit: dict) -> str:
    """Front-loaded '**Summary.**' sentence of a KB article."""
    for line in hit["excerpt"].splitlines():
        if line.startswith("**Summary.**"):
            return line.replace("**Summary.**", "").strip()
    first = next((ln for ln in hit["excerpt"].splitlines() if ln.strip() and not ln.startswith("#")), "")
    return first.strip()


class RuleBasedPlanner:
    name = "rules"

    async def decide(self, state: dict) -> Decision:
        obs: list[dict] = state["observations"]
        intent: str = state["intent"]

        # Guardrail GUARD-001: tool/infrastructure failure -> don't guess, hand off.
        if obs and not obs[-1]["ok"] and obs[-1].get("error_code") in _INFRA_ERRORS:
            return Decision("respond", summary="Tool failure - escalating instead of guessing",
                            escalate_reason="tool_failure")
        # A tool refused for a business/validation reason we cannot recover from.
        if obs and not obs[-1]["ok"] and obs[-1].get("error_code") == "order_not_found":
            return _respond("I couldn't find that order on your account. Could you double-check the "
                            "order number (it looks like ORD-123456)?", "Order not found for customer")

        # Any other failed call (4xx from the API, blocked by a local gate): don't guess.
        if obs and not obs[-1]["ok"] and obs[-1]["tool"] not in ("get_support_ticket", "create_support_ticket"):
            return Decision("respond", summary=f"Unrecoverable tool error {obs[-1].get('error_code')} - escalating",
                            escalate_reason="tool_failure")

        handler = getattr(self, f"_{intent}", None) or self._unknown
        return await handler(state)

    # ------------------------------------------------------------------ order resolution
    def _need_order(self, state: dict) -> tuple[str | None, Decision | None, bool]:
        """-> (order_number, decision_if_more_needed, assumed_latest)."""
        msg = state["message"]
        explicit = extract_order_number(msg)
        if explicit:
            return explicit, None, False
        obs = state["observations"]
        ctx_order = state["session_ctx"].get("last_order_number")
        if ctx_order and not mentions_latest(msg):
            return ctx_order, None, False
        listed = last_obs(obs, "list_customer_orders")
        if listed is None:
            return None, _call("list_customer_orders", limit=10), False
        if not listed["ok"]:
            return None, Decision("respond", summary="Could not list orders",
                                  escalate_reason="tool_failure"), False
        items = sorted(listed["data"]["items"], key=lambda o: o["placed_at"], reverse=True)
        if not items:
            return None, _respond("I don't see any orders on your account yet. If you ordered as a guest "
                                  "or with a different email, tell me the order number and I'll look.",
                                  "Customer has no orders"), False
        return items[0]["order_number"], None, True

    @staticmethod
    def _lead(order_number: str, assumed: bool) -> str:
        return (f"I'm looking at your most recent order, **{order_number}**. " if assumed else
                f"For order **{order_number}**: ")

    # ------------------------------------------------------------------ intents
    async def _greeting(self, state: dict) -> Decision:
        return _respond("Hi! I'm your support assistant. I can check orders and tracking, explain refund and "
                        "return eligibility, cancel unshipped orders, and answer policy questions. "
                        "What can I help with?", "Greeting")

    async def _policy_question(self, state: dict) -> Decision:
        obs = state["observations"]
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query=state["message"])
        hits = kb_hits(obs)
        if not hits:
            return await self._unknown(state)
        top = hits[0]
        extra = [h for h in hits[1:3] if h["score"] >= 0.6 * top["score"]]
        text = f"{_summary_line(top)} _(policy {top['policy_id']})_"
        for h in extra:
            text += f"\n\nAlso relevant: {_summary_line(h)} _(policy {h['policy_id']})_"
        return _respond(text + "\n\nWant me to check this against one of your orders?",
                        f"Answered from KB {top['policy_id']}", state)

    async def _unknown(self, state: dict) -> Decision:
        obs = state["observations"]
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query=state["message"])
        hits = kb_hits(obs)
        asked = state["session_ctx"].get("clarifications", 0)
        if hits and hits[0]["score"] >= 4.0:
            top = hits[0]
            return _respond(f"{_summary_line(top)} _(policy {top['policy_id']})_",
                            f"Answered from KB {top['policy_id']}", state)
        if asked >= 2:
            return Decision("respond", summary="Still unclear after 2 clarifying questions - escalating",
                            escalate_reason="unclear_request")
        return _respond("I want to make sure I get this right - could you tell me a bit more? For example, "
                        "is this about an order (share the order number), a refund or return, or a policy "
                        "question?", "Asked a clarifying question", clarification=True)

    async def _order_status(self, state: dict) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        d = last_obs(obs, "get_order_details", n)
        if d is None:
            return _call("get_order_details", order_number=n)
        o = d["data"]
        items = ", ".join(i["title"] for i in o.get("items", [])[:3]) or "your items"
        text = (f"{self._lead(n, assumed)}it's currently **{o['status'].replace('_', ' ')}**. "
                f"Placed {o['placed_at'][:10]}, total {_money(o['grand_total'])} ({items}).")
        if o.get("promised_delivery_date"):
            text += f" Promised delivery: {o['promised_delivery_date']}."
        return _respond(text + " Want tracking details too?", "Reported order status", state)

    async def _order_tracking(self, state: dict) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        if last_obs(obs, "get_order_tracking", n) is None:
            return _call("get_order_tracking", order_number=n)
        if last_obs(obs, "get_delivery_estimate", n) is None:
            return _call("get_delivery_estimate", order_number=n)
        t = last_obs(obs, "get_order_tracking", n)["data"]
        est = last_obs(obs, "get_delivery_estimate", n)
        lead = self._lead(n, assumed)
        if not t.get("tracking_number"):
            msg = est["data"]["message"] if est["ok"] else "It hasn't shipped yet."
            return _respond(f"{lead}there's no shipment yet. {msg}", "No shipment yet", state)
        text = (f"{lead}shipped via **{t['carrier']}**, tracking **{t['tracking_number']}** "
                f"(status: {t['shipment_status']}).")
        if t.get("events"):
            e = t["events"][-1]
            text += f" Latest scan: {e['description']} ({e['event_time'][:10]})."
        if est["ok"]:
            text += f" {est['data']['message']}"
        if t.get("tracking_url"):
            text += f"\n\nTrack it here: {t['tracking_url']}"
        return _respond(text, "Reported tracking", state)

    # ----- refunds
    async def _refund_flow(self, state: dict, request: bool) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        if last_obs(obs, "get_order_details", n) is None:
            return _call("get_order_details", order_number=n)
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query="refund eligibility window one refund per order")
        e = last_obs(obs, "check_refund_eligibility", n)
        if e is None:
            return _call("check_refund_eligibility", order_number=n)
        v = e["data"]
        lead = self._lead(n, assumed)
        verdict = (f"{lead}{v['reason']} _(policy {v['policy_id']})_")
        if not v["eligible"]:
            alt = ("" if v["refund_verdict"] != "outside_window" else
                   " I can still look at warranty or exchange options if the item has a problem.")
            return _respond(verdict + alt, f"Refund not eligible: {v['refund_verdict']}", state)
        if not request:
            ask = ("This one is over my $100 limit, so a human would approve it - I can send it to them."
                   if v["requires_human_approval"] else "I can start it for you.")
            return _respond(f"{verdict} The refund would be **{_money(v['order_value'])}**. {ask} "
                            f"Just say so if you'd like to go ahead - I won't do anything until you confirm.",
                            "Explained refund eligibility", state)
        return Decision("tool", calls=[ToolCallReq("initiate_refund", {
            "order_number": n, "reason": "Customer requested a refund via chat"})],
            summary="Eligible and customer explicitly asked - propose refund (needs confirmation)")

    async def _refund_question(self, state): return await self._refund_flow(state, request=False)
    async def _refund_request(self, state): return await self._refund_flow(state, request=True)

    async def _refund_status(self, state: dict) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        r = last_obs(obs, "get_order_refunds", n)
        if r is None:
            return _call("get_order_refunds", order_number=n)
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query="refund status timing declined not received")
        recs = r["data"]
        lead = self._lead(n, assumed)
        if not recs:
            return _respond(f"{lead}there's no refund on record yet. Want me to check whether it's eligible?",
                            "No refund records", state)
        lines = [f"- {x['amount']['display']} via {x['method'].replace('_', ' ')} - **{x['status']}**"
                 + (f" ({x['reason']})" if x.get("reason") else "") for x in recs]
        hits = kb_hits(obs)
        why = f"\n\n{_summary_line(hits[0])} _(policy {hits[0]['policy_id']})_" if hits else ""
        return _respond(f"{lead}here are the refund records:\n" + "\n".join(lines) + why,
                        "Reported refund records + policy", state)

    # ----- returns
    async def _return_flow(self, state: dict, request: bool) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        if last_obs(obs, "get_order_details", n) is None:
            return _call("get_order_details", order_number=n)
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query="return policy window condition how to start a return")
        e = last_obs(obs, "check_return_eligibility", n)
        if e is None:
            return _call("check_return_eligibility", order_number=n)
        v = e["data"]
        lead = self._lead(n, assumed)
        returnable = [i for i in v["items"] if i["is_returnable"]]
        blocked = [i["title"] for i in v["items"] if not i["is_returnable"]]
        text = f"{lead}{v['reason']} _(policy {v['policy_id']})_"
        if not v["eligible"] or not returnable:
            return _respond(text, "Return not eligible", state)
        fees = [i for i in returnable if i["restocking_fee_pct"]]
        if blocked:
            text += f" Not returnable: {', '.join(blocked)}."
        if fees:
            text += " Restocking fees apply to: " + ", ".join(
                f"{i['title']} ({i['restocking_fee_pct']}%)" for i in fees) + "."
        if not request:
            return _respond(text + " If you'd like to start a return, just say so - nothing happens "
                                   "until you confirm.", "Explained return eligibility", state)
        return Decision("tool", calls=[ToolCallReq("create_return", {
            "order_number": n, "reason": return_reason_from(state["message"]),
            "order_item_ids": [i["order_item_id"] for i in returnable]})],
            summary="Eligible items found and customer asked - propose return (needs confirmation)")

    async def _return_question(self, state): return await self._return_flow(state, request=False)
    async def _return_request(self, state): return await self._return_flow(state, request=True)

    # ----- cancellation
    async def _cancel_flow(self, state: dict, request: bool) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        e = last_obs(obs, "check_cancellation_eligibility", n)
        if e is None:
            return _call("check_cancellation_eligibility", order_number=n)
        v = e["data"]
        text = f"{self._lead(n, assumed)}{v['reason']} _(policy {v['policy_id']})_"
        if not v["eligible"]:
            return _respond(text, "Cancellation not eligible", state)
        if not request:
            return _respond(text + " Say the word if you'd like me to cancel it - I'll ask you to confirm first.",
                            "Explained cancellation eligibility", state)
        return Decision("tool", calls=[ToolCallReq("cancel_order", {
            "order_number": n, "reason": "Customer requested cancellation via chat"})],
            summary="Cancellable and customer asked - propose cancellation (needs confirmation)")

    async def _cancel_question(self, state): return await self._cancel_flow(state, request=False)
    async def _cancel_request(self, state): return await self._cancel_flow(state, request=True)

    # ----- damaged / wrong item -> ticket
    async def _damaged_or_wrong_item(self, state: dict) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        if last_obs(obs, "get_order_details", n) is None:
            return _call("get_order_details", order_number=n)
        if last_obs(obs, "get_order_items", n) is None:
            return _call("get_order_items", order_number=n)
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query="damaged wrong item received claim evidence photos")
        t = last_obs(obs, "create_support_ticket")
        if t is None:
            items = last_obs(obs, "get_order_items", n)["data"]
            titles = ", ".join(i["title"] for i in items[:5])
            return Decision("tool", calls=[ToolCallReq("create_support_ticket", {
                "subject": f"Damaged/wrong item reported on {n}",
                "category": "damaged", "order_number": n, "priority": "normal",
                "message": f"Customer reports a problem with items on {n} ({titles}). "
                           f"Customer message: {state['message'][:600]}"})],
                summary="Open a support ticket for the reported item problem")
        if not t["ok"]:
            return Decision("respond", summary="Ticket creation failed", escalate_reason="tool_failure")
        hits = kb_hits(obs)
        pol = f" _(policy {hits[0]['policy_id']})_" if hits else ""
        return _respond(f"I'm sorry about that. {self._lead(n, assumed)}I've opened ticket "
                        f"**{t['data']['ticket_number']}**. Photos of the item/packaging help us resolve it "
                        f"quickly - you can choose a replacement or a refund{pol}. A team member will follow up.",
                        "Created ticket for item problem", state)

    async def _payment_question(self, state: dict) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        p = last_obs(state["observations"], "get_order_payment", n)
        if p is None:
            return _call("get_order_payment", order_number=n)
        rows = p["data"]
        if not rows:
            return _respond(f"{self._lead(n, assumed)}I don't see any payment records yet.", "No payments", state)
        lines = [f"- {r['txn_type']}: {r['amount']['display']} - **{r['status']}**"
                 + (f" ({r['method_brand']} ending {r['method_last4']})" if r.get("method_last4") else "")
                 for r in rows]
        return _respond(f"{self._lead(n, assumed)}payment activity:\n" + "\n".join(lines),
                        "Reported payment status", state)

    async def _store_credit(self, state: dict) -> Decision:
        c = last_obs(state["observations"], "get_store_credit_balance")
        if c is None:
            return _call("get_store_credit_balance")
        return _respond(f"Your store-credit balance is **{_money(c['data']['balance'])}**.",
                        "Reported store credit", state)

    async def _product_question(self, state: dict) -> Decision:
        p = last_obs(state["observations"], "search_products")
        if p is None:
            words = [w for w in state["message"].replace("?", " ").split() if len(w) > 3
                     and w.lower() not in {"what", "have", "carry", "sell", "does", "your", "products",
                                           "looking", "available", "stock", "with", "that", "this"}]
            return _call("search_products", q=" ".join(words[:4]) or None, limit=5)
        items = p["data"]["items"]
        if not items:
            return _respond("I couldn't find a matching product. Try different keywords?", "No products found")
        lines = [f"- {i['title']} - {i['price']['display']}" + ("" if i["stock_quantity"] else " (out of stock)")
                 for i in items]
        return _respond("Here's what I found:\n" + "\n".join(lines), "Listed products")

    async def _ticket_status(self, state: dict) -> Decision:
        num = extract_ticket_number(state["message"])
        if not num:
            return _respond("What's the ticket number? It looks like TKT-1234.", "Asked for ticket number",
                            clarification=True)
        t = last_obs(state["observations"], "get_support_ticket")
        if t is None:
            return _call("get_support_ticket", ticket_number=num)
        if not t["ok"]:
            return _respond("I couldn't find that ticket on your account.", "Ticket not found")
        d = t["data"]
        return _respond(f"Ticket **{d['ticket_number']}** is **{d['status']}** (priority {d['priority']}"
                        f"{', escalated to a specialist' if d['escalated'] else ''}).", "Reported ticket status")

    async def _delay_compensation(self, state: dict) -> Decision:
        n, more, assumed = self._need_order(state)
        if more:
            return more
        obs = state["observations"]
        e = last_obs(obs, "get_delivery_estimate", n)
        if e is None:
            return _call("get_delivery_estimate", order_number=n)
        if last_obs(obs, "search_knowledge_base") is None:
            return _call("search_knowledge_base", query="delay compensation tiers goodwill credit")
        late = e["data"].get("days_late") or 0
        lead = self._lead(n, assumed)
        if late < 3:
            return _respond(f"{lead}{e['data']['message']} Goodwill credits start at 3+ days late "
                            f"_(policy DELAY-001)_, so none applies yet.", "No credit due", state)
        return Decision("tool", calls=[ToolCallReq("grant_delay_credit", {"order_number": n})],
                        summary=f"{late} days late - propose goodwill credit (needs confirmation)")

