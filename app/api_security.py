"""Shared API security helpers: admin authentication + input validation."""
import hmac

from fastapi import Depends, HTTPException, Request

from .config import ADMIN_TOKEN


def require_admin(request: Request) -> None:
    """Guard for destructive/mutating endpoints.

    When AGS_ADMIN_TOKEN is unset (local development), this is a no-op so the
    dashboard keeps working with zero setup. In production, set the env var and
    every mutating endpoint starts requiring the token via:
        X-Admin-Token: <token>     header, or
        Authorization: Bearer <token>
    Comparison is constant-time to avoid timing side channels.
    """
    if not ADMIN_TOKEN:
        return
    supplied = request.headers.get("x-admin-token") or ""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        supplied = supplied or auth[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, ADMIN_TOKEN):
        raise HTTPException(401, "Admin token required for this operation")


# Reusable dependency alias for route signatures.
AdminDep = Depends(require_admin)


def safe_url(url: str | None) -> str | None:
    """Validate a user-supplied URL before it lands in an href attribute.

    Blocks javascript:, data:, vbscript: and other scheme-injection payloads;
    allows http/https/relative URLs. Returns the URL unchanged when safe,
    otherwise None (callers should omit the link).
    """
    if not url:
        return None
    candidate = url.strip()
    # Reject anything containing control chars that could break out of attributes.
    if any(ord(c) < 32 or ord(c) == 127 for c in candidate):
        return None
    lowered = candidate.lower().replace("\t", "").replace("\n", "")
    if ":" in lowered.split("/")[0]:  # scheme portion before first slash
        scheme = lowered.split(":", 1)[0]
        if scheme not in ("http", "https"):
            return None
    return candidate
