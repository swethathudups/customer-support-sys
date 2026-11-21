r"""LangGraph orchestration of one customer turn.

    START -> screen --(abuse/legal/safety)----------------------------> escalate -> END
                  \--(pending action + "yes")--> execute_confirmed ---> END
                  \--(pending action + "no")---> cancel_pending -----> END
                  \--> understand --(needs human)--> escalate
                            \--> plan <--> act        (tool loop, bounded by AGENT_MAX_STEPS)
                                   \--> respond -> END   (or escalate on tool failure)

The graph owns every safety-relevant decision; the Planner (rules or Claude) only
chooses tools / words. Writes never run inside the loop: `act` turns a write
proposal into a stored pending action + a confirmation question, and only an
explicit "yes" on a later turn reaches `execute_confirmed`.
"""
from __future__ import annotations

import asyncio
import operator
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.guardrails import clean_reply, parse_consent, screen_message
from app.agent.intents import HUMAN_INTENTS, classify_intent
from app.agent.planner.base import Decision, Planner, last_obs
from app.agent.planner.llm import LLMPlanner
from app.agent.recorder import RunRecorder
from app.agent.redact import mask_text, scrub
from app.agent.tools.executor import ToolExecutor, ToolResult
from app.agent.tools.registry import TOOLS
from app.core.config import settings


class AgentState(TypedDict, total=False):
    # inputs
    customer_id: int
    session_id: str
    run_id: str
    message: str
    history: list[dict]
    session_ctx: dict
    # working
    intent: str
    consent: str | None
    escalate_reason: str | None
    observations: Annotated[list[dict], operator.add]
    llm_turns: Annotated[list[dict], operator.add]
    steps_taken: int
    decision: Any
    # outputs
    reply: str
    status: str
    escalated: bool
    ticket_number: str | None
    pending_action: dict | None
    pending_changed: bool
    clarification: bool
    citations: list[str]
    policy_ids: list[str]
    last_order_number: str | None


@dataclass
class AgentDeps:
    planner: Planner
    executor: ToolExecutor
    recorder: RunRecorder


# --------------------------------------------------------------------------
_ESCALATION = {
    # reason: (category, priority, customer-facing message)
    "abusive_language": ("other", "high",
        "I'm sorry this has been frustrating - I want to get it resolved properly. I've passed your "
        "conversation to a specialist who will follow up with you. Ticket: **{t}**."),
    "legal_threat": ("other", "high",
        "I understand how serious this is. I've escalated your case to a specialist team who will "
        "review it and contact you directly. Ticket: **{t}**."),
    "safety_report": ("other", "urgent",
        "Your safety comes first: please stop using the product and keep it away from people and "
        "flammable items. I've flagged this as urgent for our safety team, who will contact you. "
        "Ticket: **{t}**."),
    "chargeback_dispute": ("payments", "high",
        "Since a payment dispute is already open, refunds and payment changes on that charge are handled "
        "by our disputes team so nothing is duplicated. I've sent them your details. Ticket: **{t}**."),
    "privacy_request": ("account", "normal",
        "Data-access and deletion requests are handled by our privacy team (I can't delete data in chat). "
        "I've opened a request for them and they'll verify your identity and follow up. Ticket: **{t}**."),
    "suspected_fraud": ("account", "urgent",
        "I'm sorry - that sounds worrying. I've escalated this as a suspected-fraud case and our security "
        "team will contact you; in the meantime consider changing your password. Ticket: **{t}**."),
    "tool_failure": ("other", "normal",
        "I'm sorry - I couldn't reliably look that up right now, and I don't want to guess. I've asked a "
        "team member to verify and follow up. Ticket: **{t}**."),
    "model_refusal": ("other", "normal",
        "I'm sorry - I can't help with that one in chat. I've asked a team member to follow up. "
        "Ticket: **{t}**."),
    "unclear_request": ("other", "normal",
        "I'm still not sure I understand, so I'd rather get you to a person than guess. I've opened a "
        "ticket for a team member to follow up: **{t}**."),
}

_NO_TICKET_FALLBACK = ("I'm sorry - I couldn't complete that, and I couldn't open a ticket automatically "
                       "either. Please try again in a moment or contact support directly.")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _scrubbed(res: ToolResult) -> dict:
    return scrub(res.for_llm())


def _obs(res: ToolResult) -> dict:
    return {"tool": res.tool, "args": res.args, "ok": res.ok, "data": res.data,
            "error_code": res.error_code, "message": res.message,
            "order_number": res.order_number, "blocked": res.blocked}


def _describe_pending(tool: str, args: dict, observations: list[dict]) -> str:
    n = args.get("order_number")
    if tool == "initiate_refund":
        e = last_obs(observations, "check_refund_eligibility", n)
        amt = e["data"]["order_value"]["display"] if e and e["ok"] else "the order value"
        human = (" (this is over my $100 limit, so it will be sent to a human for approval)"
                 if e and e["ok"] and e["data"].get("requires_human_approval") else "")
        return f"refund **{amt}** for order **{n}** to the {args.get('method', 'original_payment').replace('_', ' ')}{human}"
    if tool == "create_return":
        e = last_obs(observations, "check_return_eligibility", n)
        ids = set(args.get("order_item_ids", []))
        titles = [i["title"] for i in (e["data"]["items"] if e and e["ok"] else []) if i["order_item_id"] in ids]
        return f"start a return for order **{n}**" + (f" ({', '.join(titles)})" if titles else "")
    if tool == "cancel_order":
        return f"cancel order **{n}**"
    if tool == "grant_delay_credit":
        return f"apply the late-delivery goodwill credit to order **{n}**"
    if tool == "create_claim":
        return f"open a {args.get('type', '').replace('_', ' ')} claim for order **{n}**"
    return f"run {tool} for order {n}"


def _latest_order_number(state: AgentState) -> str | None:
    for o in reversed(state.get("observations", [])):
        if o.get("order_number") and o["ok"]:
            return o["order_number"]
    return state["session_ctx"].get("last_order_number")


# --------------------------------------------------------------------------
def build_graph(deps: AgentDeps):
    rec, ex, planner = deps.recorder, deps.executor, deps.planner

    async def screen(state: AgentState) -> dict:
        s = screen_message(state["message"])
        out: dict = {"consent": None, "escalate_reason": s.escalate_reason}
        notes = []
        if s.escalate_reason:
            notes.append(f"escalation trigger: {s.escalate_reason} ({s.policy_id})")
        if s.injection_suspected:
            notes.append("possible prompt-injection text detected - treated as data, rules unchanged")
        rec.step("guardrail", "input_screen", "; ".join(notes) or "No safety triggers",
                 status="blocked" if s.escalate_reason else "ok",
                 data={"escalate": s.escalate_reason, "injection_suspected": s.injection_suspected})
        pending = state["session_ctx"].get("pending_action")
        if pending:
            age = _now() - datetime.fromisoformat(pending["created_at"])
            if age > timedelta(minutes=settings.agent_pending_ttl_minutes):
                rec.step("confirmation", "pending_expired",
                         f"Pending {pending['tool']} expired after {settings.agent_pending_ttl_minutes} min")
                pending = None
            else:
                out["consent"] = parse_consent(state["message"])
                if out["consent"] is None and not s.escalate_reason:
                    rec.step("confirmation", "pending_discarded",
                             f"Customer moved on; pending {pending['tool']} discarded (no consent given)")
                    pending = None
            if pending is None:
                out.update(pending_action=None, pending_changed=True)
        out["session_ctx"] = {**state["session_ctx"], "pending_action": pending}
        return out

    def route_screen(state: AgentState) -> str:
        if state.get("escalate_reason"):
            return "escalate"
        pending = state["session_ctx"].get("pending_action")
        if pending and state.get("consent") == "yes":
            return "execute_confirmed"
        if pending and state.get("consent") == "no":
            return "cancel_pending"
        return "understand"

    async def understand(state: AgentState) -> dict:
        intent = classify_intent(state["message"])
        rec.step("intent", "classify", f"Intent: {intent}", data={"intent": intent})
        out: dict = {"intent": intent}
        if intent in HUMAN_INTENTS:
            out["escalate_reason"] = intent
        return out

    def route_understand(state: AgentState) -> str:
        return "escalate" if state.get("escalate_reason") else "plan"

    async def plan(state: AgentState) -> dict:
        n = state.get("steps_taken", 0)
        if n >= settings.agent_max_steps:
            rec.step("decision", "step_limit", f"Step limit {settings.agent_max_steps} reached",
                     status="blocked")
            return {"decision": Decision("respond", summary="Step limit", escalate_reason="unclear_request"),
                    "steps_taken": n}
        t0 = time.monotonic()
        d = await planner.decide(state)
        rec.step("decision", f"{planner.name}_planner", d.summary,
                 data={"kind": d.kind, "tools": [c.name for c in d.calls],
                       "escalate": d.escalate_reason}, duration_ms=int((time.monotonic() - t0) * 1000))
        return {"decision": d, "steps_taken": n + 1}

    def route_plan(state: AgentState) -> str:
        d: Decision = state["decision"]
        if d.escalate_reason:
            return "escalate"
        return "act" if d.kind == "tool" else "respond"

    async def act(state: AgentState) -> dict:
        d: Decision = state["decision"]
        reads = [c for c in d.calls if not (TOOLS.get(c.name) and TOOLS[c.name].requires_confirmation)]
        writes = [c for c in d.calls if c not in reads]

        results = await asyncio.gather(*[ex.execute(c.name, c.args) for c in reads])
        for c, r in zip(reads, results):
            rec.step("tool_call", c.name, ("OK" if r.ok else f"{'Blocked' if r.blocked else 'Failed'}: {r.error_code}"),
                     status="ok" if r.ok else ("blocked" if r.blocked else "error"),
                     data={"order_number": r.order_number} if r.order_number else {})
        out: dict = {"observations": [_obs(r) for r in results]}

        # Writes needing consent: stage, never execute.
        for c in writes:
            probe = await ex.execute(c.name, c.args, confirmed=False)
            if probe.error_code == "confirmation_required":
                observations = state["observations"] + out["observations"]
                desc = _describe_pending(c.name, probe.args, observations)
                pending = {"id": uuid.uuid4().hex[:12], "tool": c.name, "args": probe.args,
                           "order_number": probe.order_number, "description": desc,
                           "risk": TOOLS[c.name].risk, "created_at": _now().isoformat()}
                rec.step("confirmation", "confirmation_requested",
                         f"Staged {c.name} for {probe.order_number}; waiting for explicit customer consent",
                         data={"pending_id": pending["id"], "tool": c.name, "risk": pending["risk"]})
                out.update(pending_action=pending, pending_changed=True, status="awaiting_confirmation",
                           reply=f"I can {desc}.\n\n**Please reply \"yes\" to confirm, or \"no\" to cancel.** "
                                 f"I won't do anything until you confirm.")
                return out
            out["observations"].append(_obs(probe))   # prerequisite_missing / invalid args -> planner recovers

        if d.llm_blocks is not None:
            out["llm_turns"] = LLMPlanner.tool_result_turn(d, [
                {"call_id": c.id, "payload": _scrubbed(r)} for c, r in zip(reads, results)]
                + [{"call_id": c.id, "payload": _scrubbed(_blocked_note(o))}
                   for c, o in zip(writes, out["observations"][len(reads):])])
        return out

    def route_act(state: AgentState) -> str:
        return "respond" if state.get("reply") else "plan"

    async def respond(state: AgentState) -> dict:
        d: Decision | None = state.get("decision")
        text = state.get("reply") or (d.reply if d else None) or "Sorry, I couldn't put an answer together."
        text = clean_reply(text)
        rec.step("response", "final_reply", f"Reply composed ({len(text)} chars)",
                 data={"policy_ids": d.policy_ids if d else [], "status": state.get("status", "completed")})
        ctx_updates: dict = {}
        if d and d.clarification:
            ctx_updates["clarifications"] = state["session_ctx"].get("clarifications", 0) + 1
        return {"reply": text, "status": state.get("status", "completed"),
                "citations": d.citations if d else [], "policy_ids": d.policy_ids if d else [],
                "last_order_number": _latest_order_number(state),
                "session_ctx": {**state["session_ctx"], **ctx_updates}}

    async def execute_confirmed(state: AgentState) -> dict:
        p = state["session_ctx"]["pending_action"]
        rec.step("confirmation", "consent_received",
                 f"Customer explicitly confirmed {p['tool']} for {p['order_number']}",
                 data={"pending_id": p["id"]})
        spec = TOOLS[p["tool"]]
        base = {"pending_action": None, "pending_changed": True, "last_order_number": p["order_number"],
                "intent": "confirm_action"}
        # Re-validate against fresh state (it may have changed since the question was asked).
        if spec.prerequisite:
            pre = await ex.execute(spec.prerequisite, {"order_number": p["order_number"]})
            rec.step("validation", spec.prerequisite, "Re-checked eligibility before executing",
                     status="ok" if pre.ok else "error")
            if not pre.ok:
                return {**base, "escalate_reason": "tool_failure"}
            if pre.data.get("eligible") is False:
                return {**base, "status": "completed", "reply": clean_reply(
                    f"I'm sorry - things have changed and I can no longer do that: {pre.data.get('reason')}")}
        res = await ex.execute(p["tool"], p["args"], confirmed=True,
                               idempotency_key=f"{state['session_id']}:{p['id']}")
        rec.step("tool_call", p["tool"], "OK" if res.ok else f"Failed: {res.error_code}",
                 status="ok" if res.ok else "error", data={"write": True, "order_number": p["order_number"]})
        if res.ok:
            msg = (res.data or {}).get("message") or "Done."
            return {**base, "status": "completed", "reply": clean_reply(f"Done - {msg}"),
                    "observations": [_obs(res)]}
        if res.error_code in ("upstream_error", "upstream_unreachable", "circuit_open"):
            return {**base, "escalate_reason": "tool_failure"}
        return {**base, "status": "completed", "reply": clean_reply(
            f"I wasn't able to complete that: {res.message}. Nothing was changed on your order.")}

    def route_confirmed(state: AgentState) -> str:
        return "escalate" if state.get("escalate_reason") else END

    async def cancel_pending(state: AgentState) -> dict:
        p = state["session_ctx"]["pending_action"]
        rec.step("confirmation", "consent_declined", f"Customer declined {p['tool']}; nothing executed")
        return {"pending_action": None, "pending_changed": True, "status": "completed",
                "intent": "decline_action",
                "reply": "No problem - I haven't changed anything. Is there anything else I can help with?"}

    async def escalate(state: AgentState) -> dict:
        reason = state.get("escalate_reason") or (state["decision"].escalate_reason if state.get("decision") else None) \
            or "unclear_request"
        category, priority, template = _ESCALATION.get(reason, _ESCALATION["unclear_request"])
        order_number = _latest_order_number(state)
        recent = " | ".join(f"{m['role']}: {mask_text(m['content'])[:160]}" for m in state["history"][-4:])
        res = await ex.execute("create_support_ticket", {
            "subject": f"Escalation: {reason.replace('_', ' ')}", "category": category,
            "order_number": order_number if order_number and order_number.startswith("ORD-") else None,
            "priority": priority,
            "message": f"Automatic escalation by support agent. Reason: {reason}. Intent: {state.get('intent')}. "
                       f"Latest customer message: {mask_text(state['message'])[:500]}. Recent context: {recent}"})
        ticket = res.data["ticket_number"] if res.ok else None
        rec.step("escalation", reason, f"Escalated to human ({priority}); ticket {ticket or 'not created'}",
                 status="ok" if res.ok else "error", data={"reason": reason, "priority": priority})
        reply = template.format(t=ticket) if ticket else _NO_TICKET_FALLBACK
        return {"reply": clean_reply(reply), "status": "escalated", "escalated": True, "ticket_number": ticket,
                "pending_action": None, "pending_changed": True, "last_order_number": order_number}

    g = StateGraph(AgentState)
    for name, fn in [("screen", screen), ("understand", understand), ("plan", plan), ("act", act),
                     ("respond", respond), ("execute_confirmed", execute_confirmed),
                     ("cancel_pending", cancel_pending), ("escalate", escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "screen")
    g.add_conditional_edges("screen", route_screen, ["escalate", "execute_confirmed", "cancel_pending", "understand"])
    g.add_conditional_edges("understand", route_understand, ["escalate", "plan"])
    g.add_conditional_edges("plan", route_plan, ["escalate", "act", "respond"])
    g.add_conditional_edges("act", route_act, ["respond", "plan"])
    g.add_conditional_edges("execute_confirmed", route_confirmed, ["escalate", END])
    for terminal in ("respond", "cancel_pending", "escalate"):
        g.add_edge(terminal, END)
    return g.compile()


def _blocked_note(obs: dict) -> ToolResult:
    return ToolResult(obs["tool"], obs["args"], obs["ok"], data=obs["data"],
                      error_code=obs["error_code"], message=obs["message"])
