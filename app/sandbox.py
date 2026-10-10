"""Protection layer for agent tool use, OpenShell-aware.

Two modes, same policy model as NVIDIA OpenShell sandboxes:

1. `openshell-supervised` — this process runs under `openshell-sandbox`
   supervision (or inside an OpenShell-managed sandbox): network policy,
   shields, and secret rewriting at the boundary are inherited from the
   supervisor. Local guardrails below still apply as defense in depth.
2. `local-guardrails` — standalone: the guardrails in this module are the
   enforcement point.

Guardrails (always on):
- SSRF-safe fetching: http(s) only, no credentials in URL, hostname resolved
  up-front and every resolved IP must be globally routable (blocks localhost,
  RFC-1918, link-local incl. 169.254.169.254, multicast, reserved).
  NOTE: resolve-then-fetch has a residual DNS-rebinding TOCTOU window; under
  OpenShell supervision the L7 proxy closes it with per-connection policy.
- Response caps (default 2 MB) and timeouts so a malicious endpoint cannot
  exhaust memory or hang an agent run.
- Secret redaction: known secret patterns are scrubbed from anything an agent
  stores or returns (excerpts, audit, reports).
- Audit ring: every agent tool use (fetch / invoke) is recorded
  (actor, action, target, verdict) for the console and API.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import threading
from collections import deque
from typing import Any
from urllib.parse import urlparse

FETCH_TIMEOUT = 8
FETCH_MAX_BYTES = 2_000_000
AUDIT_MAXLEN = 200

SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"AKIA[0-9A-Z]{16}", "AKIA[REDACTED]"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "XOX[REDACTED]"),
    (r"gh[pousr]_[A-Za-z0-9]{20,}", "GH[REDACTED]"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[^-]*-----END [A-Z ]*PRIVATE KEY-----", "[PRIVATE KEY REDACTED]"),
    (r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^'\"\s,}]{8,}", r"\1=[REDACTED]"),
    (r"(?i)bearer\s+[A-Za-z0-9\-._~+/=]{12,}", "Bearer [REDACTED]"),
]

_audit: deque = deque(maxlen=AUDIT_MAXLEN)
_audit_lock = threading.Lock()


def redact_secrets(text: str) -> str:
    out = text or ""
    for pattern, repl in SECRET_PATTERNS:
        out = re.sub(pattern, repl, out)
    return out


def audit(actor: str, action: str, target: str, verdict: str, detail: str = "") -> dict[str, Any]:
    from datetime import datetime, timezone
    entry = {"at": datetime.now(timezone.utc).isoformat(), "actor": actor,
             "action": action, "target": redact_secrets(target or "")[:300],
             "verdict": verdict, "detail": redact_secrets(detail or "")[:300]}
    with _audit_lock:
        _audit.append(entry)
    return entry


def recent_audit(limit: int = 50) -> list[dict[str, Any]]:
    with _audit_lock:
        items = list(_audit)
    return items[-max(1, min(limit, AUDIT_MAXLEN)):][::-1]


def _path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def openshell_state() -> dict[str, Any]:
    """Detect OpenShell supervision from env + mounted files. Never raises."""
    sandbox = os.environ.get("OPENSHELL_SANDBOX", "")
    sandbox_id = os.environ.get("OPENSHELL_SANDBOX_ID", "")
    token_file = os.environ.get("OPENSHELL_SANDBOX_TOKEN_FILE", "/etc/openshell/auth/sandbox.jwt")
    supervised = bool(sandbox or sandbox_id)
    return {
        "supervised": supervised,
        "sandbox": sandbox or None,
        "sandbox_id": sandbox_id or None,
        "token_present": _path_exists(token_file),
        "tls_client_present": _path_exists(os.environ.get("OPENSHELL_TLS_CERT", "/etc/openshell/tls/client/tls.crt")),
        "endpoint": os.environ.get("OPENSHELL_ENDPOINT"),
    }


def protection_status() -> dict[str, Any]:
    """Protection posture for the console + API. Never raises."""
    oss = openshell_state()
    return {
        "mode": "openshell-supervised" if oss["supervised"] else "local-guardrails",
        "openshell": oss,
        "guardrails": {
            "ssrf_private_ip_block": True,
            "secret_redaction": True,
            "max_response_bytes": FETCH_MAX_BYTES,
            "fetch_timeout_s": FETCH_TIMEOUT,
            "audit_entries": len(_audit),
        },
        "notes": (
            "Inherited OpenShell network policy, shields, and boundary secret "
            "rewriting, plus in-app guardrails as defense in depth."
            if oss["supervised"] else
            "Standalone mode: in-app SSRF blocking, secret redaction, caps and "
            "audit are the enforcement point. For full OpenShell protection, run "
            "this stack supervised — see README 'Agent protection'."
        ),
    }


def _ips_are_global(host: str) -> tuple[bool, str]:
    """Resolve host and require every address to be globally routable."""
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    ips = sorted({info[4][0] for info in infos})
    if not ips:
        return False, "no addresses resolved"
    for ip in ips:
        try:
            if not ipaddress.ip_address(ip).is_global:
                return False, f"non-routable address {ip}"
        except ValueError:
            return False, f"unparsable address {ip}"
    return True, f"{len(ips)} routable address(es)"


def check_url(url: str) -> tuple[bool, str]:
    """Policy check for an outbound fetch. Returns (allowed, reason)."""
    if not url or not url.strip():
        return False, "empty URL"
    url = url.strip()
    try:
        parts = urlparse(url)
    except Exception as exc:
        return False, f"unparsable URL: {exc}"
    if parts.scheme not in ("http", "https"):
        return False, f"scheme '{parts.scheme or '?'}' not allowed (http/https only)"
    if not parts.hostname:
        return False, "no hostname"
    if "@" in (parts.netloc or ""):
        return False, "credentials in URL are not allowed"
    ok, reason = _ips_are_global(parts.hostname)
    if not ok:
        return False, f"blocked host: {reason}"
    return True, reason


def guarded_fetch(url: str, actor: str = "agent",
                  timeout: int = FETCH_TIMEOUT,
                  max_bytes: int = FETCH_MAX_BYTES) -> dict[str, Any]:
    """SSRF-safe GET with caps, secret redaction, and audit. Never raises."""
    allowed, reason = check_url(url)
    if not allowed:
        audit(actor, "fetch", url, "blocked", reason)
        return {"url": url, "title": "", "excerpt": "", "status": f"blocked: {reason}"}
    try:
        import requests

        with requests.get(url.strip(), timeout=(timeout, timeout), stream=True,
                          headers={"User-Agent": "PostAGI-SecurityAgent/1.0"}) as resp:
            ctype = resp.headers.get("content-type", "")
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    audit(actor, "fetch", url, "truncated", f"exceeded {max_bytes} bytes")
                    break
                chunks.append(chunk)
            body = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        import html as _html
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))).strip()[:200]
        text = re.sub(r"<script.*?</script>", " ", body, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", _html.unescape(text)).strip()
        excerpt = redact_secrets(text[:1500])
        audit(actor, "fetch", url, "allowed", f"HTTP {resp.status_code}, {ctype}")
        return {"url": url, "title": redact_secrets(title),
                "excerpt": excerpt, "status": f"fetched ({resp.status_code}, {ctype})"}
    except Exception as exc:
        audit(actor, "fetch", url, "failed", str(exc)[:200])
        return {"url": url, "title": "", "excerpt": "", "status": f"unreachable: {exc}"}
