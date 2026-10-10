"""Append-only, tamper-evident ledger for agent tool-use logging.

Each entry hash-chains to the previous (genesis `prev_hash` is 64 zeros), so
any edit, deletion, or reordering breaks verification. The ledger lives in its
own SQLite file (`data/ledger.db`, WAL mode) — independent of the app database
so a compromise of app data cannot silently rewrite history without breaking
the chain.

Entry hash = sha256 over canonical JSON of:
    {seq, at, actor, action, target, verdict, detail, prev_hash}

All functions never raise: logging must never break an agent run.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any

GENESIS_PREV = "0" * 64

_lock = threading.Lock()


def _db_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    data = os.path.join(root, "data")
    try:
        os.makedirs(data, exist_ok=True)
    except Exception:
        pass
    return os.path.join(data, "ledger.db")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path(), timeout=10, check_same_thread=False)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass
    con.execute(
        "CREATE TABLE IF NOT EXISTS ledger ("
        "seq INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL, "
        "action TEXT NOT NULL, target TEXT NOT NULL DEFAULT '', "
        "verdict TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT '', "
        "prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_ledger_at ON ledger(at)")
    con.commit()
    return con


def _canonical(seq: int, at: str, actor: str, action: str, target: str,
               verdict: str, detail: str, prev_hash: str) -> str:
    return json.dumps({"seq": seq, "at": at, "actor": actor, "action": action,
                       "target": target, "verdict": verdict, "detail": detail,
                       "prev_hash": prev_hash},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def append(actor: str, action: str, target: str = "",
           verdict: str = "", detail: str = "") -> dict[str, Any]:
    """Append one entry, returning it (with seq + hashes). Never raises."""
    from datetime import datetime, timezone
    try:
        with _lock:
            con = _connect()
            try:
                row = con.execute(
                    "SELECT seq, entry_hash FROM ledger ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                seq = (row[0] + 1) if row else 1
                prev = row[1] if row else GENESIS_PREV
                at = datetime.now(timezone.utc).isoformat()
                h = hashlib.sha256(_canonical(
                    seq, at, actor, action, target, verdict, detail, prev
                ).encode("utf-8")).hexdigest()
                con.execute(
                    "INSERT INTO ledger (seq, at, actor, action, target, verdict, detail, prev_hash, entry_hash)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (seq, at, actor, action, target, verdict, detail, prev, h),
                )
                con.commit()
                return {"seq": seq, "at": at, "actor": actor, "action": action,
                        "target": target, "verdict": verdict, "detail": detail,
                        "prev_hash": prev, "entry_hash": h}
            finally:
                con.close()
    except Exception:
        return {}


def recent(limit: int = 50) -> list[dict[str, Any]]:
    """Newest entries first. Never raises."""
    try:
        con = _connect()
        try:
            rows = con.execute(
                "SELECT seq, at, actor, action, target, verdict, detail, prev_hash, entry_hash"
                " FROM ledger ORDER BY seq DESC LIMIT ?", (max(1, min(limit, 200)),),
            ).fetchall()
            return [{"seq": r[0], "at": r[1], "actor": r[2], "action": r[3],
                     "target": r[4], "verdict": r[5], "detail": r[6],
                     "prev_hash": r[7], "entry_hash": r[8]} for r in rows]
        finally:
            con.close()
    except Exception:
        return []


def verify() -> dict[str, Any]:
    """Recompute the full chain. Returns status + first break, if any."""
    try:
        con = _connect()
        try:
            rows = con.execute(
                "SELECT seq, at, actor, action, target, verdict, detail, prev_hash, entry_hash"
                " FROM ledger ORDER BY seq ASC"
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:
        return {"ok": False, "count": 0, "error": str(exc)[:200]}
    expected_prev = GENESIS_PREV
    for r in rows:
        seq, at, actor, action, target, verdict, detail, prev, h = r
        if prev != expected_prev:
            return {"ok": False, "count": len(rows), "broken_at_seq": seq,
                    "reason": "prev_hash link broken (deletion/reorder?)"}
        if hashlib.sha256(_canonical(
                seq, at, actor, action, target, verdict, detail, prev
        ).encode("utf-8")).hexdigest() != h:
            return {"ok": False, "count": len(rows), "broken_at_seq": seq,
                    "reason": "entry hash mismatch (edit?)"}
        expected_prev = h
    head = rows[-1][8] if rows else GENESIS_PREV
    return {"ok": True, "count": len(rows), "head": head}
