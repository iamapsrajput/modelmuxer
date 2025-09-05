from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.models import ChatCompletionRequest, ChatMessage
from app.telemetry import metrics as m
from app.telemetry.tracing import start_span


@dataclass
class PolicyResult:
    sanitized_prompt: str
    blocked: bool
    reasons: List[str]


# Compile simple PII regexes (no raw PII logs!)
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Additional patterns
IPV4_RE = re.compile(r"\b(?:(?:2[0-5]{2}|1?\d?\d)\.){3}(?:2[0-5]{2}|1?\d?\d)\b")
IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
JWT_RE = re.compile(r"\b[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b")
ADDR_RE = re.compile(r"\b\d+\s+\w+(?:\s+\w+)*\b")
NATIONAL_ID_RE = re.compile(r"\b\w{6,14}\b")


# Jailbreak patterns cache
_PATTERNS_CACHE: List[str] = []
_PATTERNS_TS: float = 0.0
_PATTERNS_TTL_SEC: int = 15


def _get_settings():
    # Obtain current settings instance (supports tests that reload module)
    import app.settings as settings_module

    return settings_module.settings


def _apply_pii_redaction(text: str) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {
        "email": 0,
        "phone": 0,
        "creditcard": 0,
        "ssn": 0,
        "ipv4": 0,
        "ipv6": 0,
        "iban": 0,
        "jwt": 0,
        "address": 0,
        "nationalid": 0,
    }

    def _redact(pattern: re.Pattern[str], label: str, t: str) -> str:
        def repl(match: re.Match[str]) -> str:
            counts[label] += 1
            return f"<{label}-redacted>"

        return pattern.sub(repl, t)

    redacted = text
    redacted = _redact(EMAIL_RE, "email", redacted)
    redacted = _redact(PHONE_RE, "phone", redacted)
    redacted = _redact(CC_RE, "creditcard", redacted)
    redacted = _redact(SSN_RE, "ssn", redacted)
    redacted = _redact(IPV4_RE, "ipv4", redacted)
    redacted = _redact(IPV6_RE, "ipv6", redacted)
    redacted = _redact(IBAN_RE, "iban", redacted)
    redacted = _redact(JWT_RE, "jwt", redacted)
    redacted = _redact(ADDR_RE, "address", redacted)
    redacted = _redact(NATIONAL_ID_RE, "nationalid", redacted)

    # Extra regex from settings
    s = _get_settings()
    extra = getattr(s.policy, "extra_pii_regex", []) or []
    for idx, pattern_str in enumerate(extra[:10]):  # guard cardinality
        label = f"extra{idx}"
        if label not in counts:
            counts[label] = 0
        try:
            pattern = re.compile(pattern_str)
        except re.error:
            continue
        redacted = _redact(pattern, label, redacted)

    # Optional NER hook
    if getattr(s.features, "enable_pii_ner", False):
        for span_text, label in ner_detect(text):
            if span_text in redacted:
                counts[label] = counts.get(label, 0) + 1
                redacted = redacted.replace(span_text, f"<{label}-redacted>")

    return redacted, counts


def ner_detect(text: str) -> List[Tuple[str, str]]:
    # Stub: return empty list unless a backend is wired
    return []


def _load_jailbreak_patterns(path: str) -> List[str]:
    global _PATTERNS_CACHE, _PATTERNS_TS
    now = time.time()
    if _PATTERNS_CACHE and (now - _PATTERNS_TS) < _PATTERNS_TTL_SEC:
        return _PATTERNS_CACHE
    p = Path(path)
    if not p.exists():
        _PATTERNS_CACHE = []
        _PATTERNS_TS = now
        return _PATTERNS_CACHE
    try:
        _PATTERNS_CACHE = [
            line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except Exception:
        _PATTERNS_CACHE = []
    _PATTERNS_TS = now
    return _PATTERNS_CACHE


def _detect_jailbreak(text: str, patterns: List[str]) -> bool:
    tlower = text.lower()
    for pat in patterns:
        if pat.lower() in tlower:
            return True
    return False


def _enforce_allow_deny(
    tenant_id: str, model: str | None, region: str | None
) -> Tuple[bool, List[str]]:
    s = _get_settings()
    reasons: List[str] = []
    blocked = False

    # Model allow/deny
    allow_map = s.policy.model_allow or {}
    deny_map = s.policy.model_deny or {}
    if model:
        allowed = allow_map.get(tenant_id)
        denied = deny_map.get(tenant_id)
        if allowed is not None and model not in allowed:
            blocked = True
            reasons.append("model_denied")
        if denied is not None and model in denied:
            blocked = True
            if "model_denied" not in reasons:
                reasons.append("model_denied")

    # Region allow/deny
    r_allow = s.policy.region_allow or {}
    r_deny = s.policy.region_deny or {}
    if region:
        r_allowed = r_allow.get(tenant_id)
        r_denied = r_deny.get(tenant_id)
        if r_allowed is not None and region not in r_allowed:
            blocked = True
            reasons.append("region_denied")
        if r_denied is not None and region in r_denied:
            blocked = True
            if "region_denied" not in reasons:
                reasons.append("region_denied")

    return blocked, reasons


def _messages_to_text(messages: List[ChatMessage]) -> str:
    return "\n".join([m.content for m in messages if m.content])


def enforce_policies(request: ChatCompletionRequest, tenant_id: str) -> PolicyResult:
    """Apply policy enforcement to a chat request.

    - PII redaction when enabled
    - Jailbreak detection when enabled
    - Per-tenant allow/deny for model and region
    """
    s = _get_settings()
    text = _messages_to_text(request.messages)
    reasons: List[str] = []
    blocked = False
    total_redactions = 0
    pii_types: List[str] = []

    patterns = _load_jailbreak_patterns(s.policy.jailbreak_patterns_path)

    with start_span(
        "policy.enforce",
        attributes={"tenant_id": tenant_id},
    ) as span:
        # PII redaction
        if s.features.redact_pii:
            sanitized, counts = _apply_pii_redaction(text)
            text = sanitized
            for kind, cnt in counts.items():
                if cnt > 0 and kind in {
                    "email",
                    "phone",
                    "creditcard",
                    "ssn",
                    "ipv4",
                    "ipv6",
                    "iban",
                    "jwt",
                    "address",
                    "nationalid",
                }:
                    pii_types.append(kind)
                    total_redactions += cnt
                    m.POLICY_REDACTIONS.labels(pii_type=kind).inc(cnt)
        # Jailbreak detection
        if s.policy.enable_jailbreak_detection and _detect_jailbreak(text, patterns):
            blocked = True
            reasons.append("jailbreak_detected")
            m.POLICY_VIOLATIONS.labels(type="jailbreak").inc()

        # Allow/deny (model/region)
        region = getattr(request, "region", None)
        md_blocked, md_reasons = _enforce_allow_deny(tenant_id, request.model, region)
        if md_blocked:
            blocked = True
            for r in md_reasons:
                tag = r.replace("_", "")
                m.POLICY_VIOLATIONS.labels(type=tag).inc()
            reasons.extend(md_reasons)

        # Set span attributes (no raw PII)
        if span is not None:
            try:
                span.set_attribute("tenant_id", tenant_id)
                span.set_attribute("blocked", blocked)
                span.set_attribute("reasons", ",".join(reasons) if reasons else "")
                span.set_attribute("num_redactions", total_redactions)
                span.set_attribute("pii_types", ",".join(sorted(set(pii_types))))
            except Exception:
                pass

    return PolicyResult(sanitized_prompt=text, blocked=blocked, reasons=reasons)
