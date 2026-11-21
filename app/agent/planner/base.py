"""Planner contract - the pluggable "brain" behind the LangGraph orchestration.

A planner looks at the turn's state (message, history, observations so far) and
returns the next Decision: call tool(s) or give the final reply. The graph owns
everything else (guardrails, confirmation, execution, tracing), so swapping the
planner (rules <-> Claude <-> another framework) cannot weaken safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolCallReq:
    name: str
    args: dict
    id: str = ""          # tool_use id (LLM planner)


@dataclass
class Decision:
    kind: Literal["tool", "respond"]
    calls: list[ToolCallReq] = field(default_factory=list)
    reply: str | None = None
    summary: str = ""                      # one-line, non-sensitive decision note for the trace
    escalate_reason: str | None = None     # ask the graph to hand off to a human
    citations: list[str] = field(default_factory=list)   # KB paths
    policy_ids: list[str] = field(default_factory=list)
    clarification: bool = False            # this reply asks the customer a clarifying question
    llm_blocks: Any = None                 # raw assistant content to replay next loop (LLM planner)


class Planner(Protocol):
    name: str

    async def decide(self, state: dict) -> Decision: ...


def last_obs(observations: list[dict], tool: str, order_number: str | None = None) -> dict | None:
    for o in reversed(observations):
        if o["tool"] == tool and (order_number is None or o.get("order_number") == order_number):
            return o
    return None


def kb_hits(observations: list[dict]) -> list[dict]:
    hits: list[dict] = []
    for o in observations:
        if o["tool"] == "search_knowledge_base" and o["ok"]:
            hits.extend(o["data"])
    return hits
