"""Deterministic guardrails that sit OUTSIDE the LLM.

Anything security- or money-relevant is decided here with plain code, so a
prompt (or a model mistake) cannot talk its way past it:
  * consent detection for pending write actions
  * escalation triggers (abuse, legal/safety threats) per KB ESC-003/ESC-004
  * prompt-injection flagging (customer text is data, never instructions)
  * output hygiene (no leaked tags, card numbers, system text)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.redact import mask_text

_YES = re.compile(r"^\W*(yes|y|yeah|yep|yup|sure|ok|okay|confirm|confirmed|please do|do it|go ahead|"
                  r"proceed|that'?s (right|correct|fine)|sounds good|absolutely|affirmative)\b", re.I)
_NO = re.compile(r"\b(no|nope|nah|cancel|don'?t|do not|stop|never ?mind|not now|wait|hold on|abort)\b", re.I)

_ABUSE = re.compile(r"\b(idiot|stupid|moron|scam(mer)?s?|f+u+c+k\w*|shit\w*|bullshit|asshole|bastard|"
                    r"useless piece|screw you)\b", re.I)
_LEGAL = re.compile(r"\b(lawyer|attorney|sue\b|suing|lawsuit|legal action|small claims|"
                    r"consumer protection|attorney general|ftc complaint|report you to)\b", re.I)
_SAFETY = re.compile(r"\b(caught fire|on fire|burn(ed|t)|shock(ed)?|injur(y|ed)|hospital|poison|"
                     r"allergic reaction|choking|recall|unsafe|hazard)\b", re.I)
_INJECTION = re.compile(r"(ignore (all |any |the )?(previous|prior|above) (instructions|rules)|"
                        r"disregard (your|the) (rules|instructions|policy)|system prompt|developer mode|"
                        r"you are now|act as (an? )?(admin|manager|supervisor)|"
                        r"override (the )?(limit|policy|rules)|approve .{0,20}(regardless|anyway))", re.I)


@dataclass(frozen=True)
class Screening:
    escalate_reason: str | None      # 'abusive_language' | 'legal_threat' | 'safety_report'
    policy_id: str | None
    injection_suspected: bool


def screen_message(text: str) -> Screening:
    if _SAFETY.search(text):
        return Screening("safety_report", "ESC-004", bool(_INJECTION.search(text)))
    if _LEGAL.search(text):
        return Screening("legal_threat", "ESC-004", bool(_INJECTION.search(text)))
    if _ABUSE.search(text):
        return Screening("abusive_language", "ESC-003", bool(_INJECTION.search(text)))
    return Screening(None, None, bool(_INJECTION.search(text)))


def parse_consent(text: str) -> str | None:
    """'yes' | 'no' | None (unclear / a new question). Deliberately conservative."""
    t = text.strip()
    if not t or len(t.split()) > 14 or "?" in t:
        return None
    if _NO.search(t):
        return "no"
    if _YES.match(t):
        return "yes"
    return None


_TAGS = re.compile(r"</?(system|thinking|tool_call|tool_use|assistant|instructions?)[^>]*>", re.I)


def clean_reply(text: str, limit: int = 3000) -> str:
    text = _TAGS.sub("", text)
    text = mask_text(text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."
