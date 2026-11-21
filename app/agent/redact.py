"""Redaction for anything the agent persists or exposes (traces, logs, LLM input).

Traces are structured audit metadata, never raw customer records: PII-bearing
keys are masked, free text is scrubbed of card-like numbers / emails, and
payloads are size-capped.
"""
from __future__ import annotations

import json
import re
from typing import Any

_SENSITIVE_KEYS = {
    "email", "phone", "password", "password_hash", "token", "api_key", "authorization",
    "line1", "line2", "postal_code", "recipient_name", "bill_to", "signed_by",
    "processor_ref", "card_number", "cvv", "first_name", "last_name",
}
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
MAX_JSON_CHARS = 1800


def mask_text(text: str) -> str:
    """Scrub card-like numbers and emails from free text."""
    text = _CARD_RE.sub(lambda m: "[card number removed]" if _luhn(m.group()) else m.group(), text)
    return _EMAIL_RE.sub("[email removed]", text)


def _luhn(candidate: str) -> bool:
    """Only mask digit runs that are valid card numbers (spares tracking numbers)."""
    digits = [int(c) for c in candidate if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return len(digits) >= 13 and total % 10 == 0


def redact(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "..."
    if isinstance(value, dict):
        return {k: ("[redacted]" if k.lower() in _SENSITIVE_KEYS else redact(v, depth + 1))
                for k, v in value.items()}
    if isinstance(value, list):
        out = [redact(v, depth + 1) for v in value[:20]]
        if len(value) > 20:
            out.append(f"... {len(value) - 20} more")
        return out
    if isinstance(value, str):
        return mask_text(value)[:300]
    return value


def to_json(value: Any, limit: int = MAX_JSON_CHARS) -> str:
    """Redacted, size-capped JSON string (always valid JSON)."""
    text = json.dumps(redact(value), default=str, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return json.dumps({"truncated": True, "preview": text[:limit]})


def scrub(value: Any) -> Any:
    """Drop PII-bearing keys but keep full content (for data handed to the LLM)."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k.lower() not in _SENSITIVE_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value
