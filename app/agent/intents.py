"""Deterministic intent + entity extraction.

Used by both planners: the intent is the trace label and drives the offline
planner; in LLM mode it is a cheap, auditable hint (the model still chooses the
tools). Order matters - the first matching rule wins, most specific first.
"""
from __future__ import annotations

import re

ORDER_RE = re.compile(r"\bORD-\d{3,}\b", re.I)
TICKET_RE = re.compile(r"\bTKT-\d+\b", re.I)

# Intents the agent must hand to a human instead of handling in chat.
HUMAN_INTENTS = {"chargeback_dispute", "privacy_request", "suspected_fraud"}

_RULES: list[tuple[str, re.Pattern]] = [(name, re.compile(p, re.I)) for name, p in [
    ("chargeback_dispute", r"\bchargeback|dispute(d)? (the )?(charge|payment)|filed a dispute"),
    ("privacy_request", r"delete (all )?my (data|account)|erase my data|ccpa|gdpr|my personal data"),
    ("suspected_fraud", r"(didn'?t|did not|never) (place|make|order)|someone (else )?(placed|ordered|used)|"
                        r"unauthori[sz]ed|fraud|hacked|account (was )?compromised"),
    ("ticket_status", r"\bTKT-\d+|ticket (status|number|update)"),
    ("damaged_or_wrong_item", r"(wrong|incorrect|different) (item|product|size|color)|"
                              r"(received|got|arrived).{0,30}(damaged|broken|cracked|defective|wrong|missing)|"
                              r"(damaged|broken|cracked|defective|dead on arrival)|missing (item|part|piece)|"
                              r"not what i ordered"),
    ("refund_status", r"(why|reason).{0,40}refund.{0,30}(reject|denied|declin|refus)|"
                      r"refund.{0,30}(reject|denied|declin|refus)|"
                      r"(where|when|status).{0,30}(my )?refund|refund.{0,20}(not (received|here|arrived)|haven'?t)"),
    ("policy_question", r"\bpolic(y|ies)\b|what('s| is) (your|the) .{0,20}(window|limit|fee)|"
                        r"how (long|many days) (do i|to|can i|will it take)"),
    ("cancel_request", r"\b(cancel|cancellation)\b.{0,30}\border\b|\bcancel (it|my)\b|"
                       r"(i want|please|i'd like|i would like|go ahead and) .{0,15}cancel"),
    ("cancel_question", r"can i cancel|able to cancel|is it too late to cancel|cancel(l)?ation (policy|eligib)"),
    ("refund_request", r"\b(i want|i'd like|i would like|please|can you|could you|go ahead and|need)\b.{0,25}"
                       r"\b(refund|money back)\b|\brefund (me|my order|this)\b|process (a|my) refund|"
                       r"issue (a|my) refund|i want my money back"),
    ("refund_question", r"\brefund\b|money back|reimburse"),
    ("return_request", r"\b(i want|i'd like|i would like|please|need|start|initiate|go ahead and)\b.{0,25}"
                       r"\b(return|send (it|this) back|exchange)\b"),
    ("return_question", r"\breturn(s|ed|ing)?\b|send (it|this) back|exchange"),
    ("delay_compensation", r"(late|delayed|delay|overdue).{0,40}(compensat|credit|make up|goodwill)|"
                           r"(compensat|credit).{0,40}(late|delay)"),
    ("order_tracking", r"where.{0,12}ORD-\d+|where.{0,20}(is )?(my )?(order|package|parcel|delivery|shipment)|\btrack(ing)?\b|"
                       r"when (will|does|is).{0,30}(arrive|deliver|come|get here)|delivery (date|estimate)|"
                       r"hasn'?t (arrived|come)|still waiting|not (yet )?(arrived|delivered)"),
    ("order_status", r"(status|update).{0,20}(of )?(my )?(latest |last |recent )?order|"
                     r"order (status|update)|is my order (shipped|ready|processed)|has my order shipped"),
    ("payment_question", r"\b(charged|charge|payment|paid|billing|invoice|receipt|card)\b"),
    ("store_credit", r"store credit|gift card balance|credit balance"),
    ("product_question", r"\b(do you (sell|have|carry)|what products|product (info|details)|in stock|"
                         r"available in|looking for)\b"),
    ("policy_question", r"\b(policy|policies|how (long|many|much|do|does)|what (is|are)|"
                        r"warranty|shipping (cost|time|method)|hours|contact)\b"),
    ("greeting", r"^\W*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b"),
]]


def classify_intent(text: str) -> str:
    for name, pat in _RULES:
        if pat.search(text):
            return name
    return "order_status" if ORDER_RE.search(text) else "unknown"


def extract_order_number(text: str) -> str | None:
    m = ORDER_RE.search(text)
    return m.group().upper() if m else None


def extract_ticket_number(text: str) -> str | None:
    m = TICKET_RE.search(text)
    return m.group().upper() if m else None


def mentions_latest(text: str) -> bool:
    return bool(re.search(r"\b(latest|last|most recent|newest|recent)\b", text, re.I))


def return_reason_from(text: str) -> str:
    t = text.lower()
    for key, reason in (("wrong item", "wrong_item"), ("wrong size", "wrong_size"),
                        ("defective", "defective"), ("broken", "damaged"), ("damaged", "damaged"),
                        ("missing", "missing_item"), ("not as described", "not_as_described")):
        if key in t:
            return reason
    return "change_of_mind"
