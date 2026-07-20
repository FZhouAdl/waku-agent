"""Stdio MCP server exposing Waku's flagship tools to the `cortex` CLI.

Register once:
    cortex mcp add waku-tools python -m waku.tools.mcp_server

Then `cortex` routes its own `tool_use` blocks here natively — no prompt-stuffed
tool schemas, no shape negotiation. The server reuses the same callables the Waku
loop uses internally; this is transport only.

Implementation: stdio JSON-RPC 2.0 (the MCP transport). Reads newline-delimited
JSON from stdin, writes responses to stdout, logs to stderr. Minimal dependency —
stdlib only.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone


# ── tool schemas — mirror the ToolRegistry definitions for the three
#    flagship tools the agent needs for scheduling / memory / messaging.

_TOOLS = [
    {
        "name": "create_event",
        "description": (
            "Create a calendar event. Use this for any scheduling request — "
            "meetings, calls, reminders, deadlines. The event is saved in the "
            "user's calendar and synced to Apple Calendar if configured."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the event"},
                "start": {"type": "string", "description": "Start time in ISO 8601 (e.g. 2025-07-20T14:00:00)"},
                "end": {"type": "string", "description": "End time in ISO 8601. Defaults to start + 1h if omitted"},
                "attendees": {"type": "string", "description": "Comma-separated list of attendee names"},
                "notes": {"type": "string", "description": "Optional notes or description for the event"},
            },
            "required": ["title", "start"],
        },
    },
    {
        "name": "save_note",
        "description": (
            "Save a note to the user's memory. Use this to remember facts, "
            "preferences, decisions, or any information the user wants kept. "
            "Notes are stored permanently and can be retrieved later."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Short subject line for the note"},
                "content": {"type": "string", "description": "The body of the note — what to remember"},
            },
            "required": ["subject", "content"],
        },
    },
    {
        "name": "send_message",
        "description": (
            "Draft a message to someone. The message is written to the user's "
            "outbox for review — nothing is actually sent. Use this when the user "
            "asks to send a note, message, or email to a specific person."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Who the message is for (name or email)"},
                "body": {"type": "string", "description": "The message content"},
            },
            "required": ["to", "body"],
        },
    },
]

_TOOL_MAP = {t["name"]: t for t in _TOOLS}


# ── tool implementations — thin wrappers around the real tool callables.
#    These import lazily so the MCP server doesn't require waku.app boot.

def _create_event(title, start, end=None, attendees=None, notes=None):
    from waku.tools.calendar import _ensure_paths, _write_db, _write_ics, sync_to_apple_calendar
    import os

    title = title.strip()
    start_str = start.strip()
    if not title or not start_str:
        return "Error: title and start are required"

    end_str = (end or "").strip()
    if not end_str:
        from dateutil.parser import parse as dt_parse
        try:
            dt = dt_parse(start_str)
            end_str = (dt + __import__("datetime").timedelta(hours=1)).isoformat()
        except Exception:
            return "Error: could not parse start time; please provide an explicit end time"

    home = _ensure_paths()

    # idempotence guard
    import sqlite3
    db = sqlite3.connect(str(home / "state.db"))
    cur = db.execute(
        "SELECT 1 FROM calendar_events WHERE title=? AND start=? LIMIT 1",
        (title, start_str),
    )
    if cur.fetchone():
        db.close()
        return f"Event '{title}' at {start_str} already exists (not duplicated)"

    _write_db(db, title, start_str, end_str, attendees or "", notes or "")
    _write_ics(home, title, start_str, end_str, attendees or "", notes or "")
    db.close()

    if os.getenv("WAKU_APPLE_CALENDAR"):
        sync_to_apple_calendar(title, start_str, end_str, attendees or "", notes or "")

    return f"Created event '{title}' from {start_str} to {end_str}"


def _save_note(subject, content):
    from waku.tools.notes import _ensure_paths
    import sqlite3

    home = _ensure_paths()
    db = sqlite3.connect(str(home / "state.db"))
    db.execute(
        "INSERT INTO facts (subject, content, source) VALUES (?, ?, ?)",
        (subject.strip().lower(), content.strip(), "user"),
    )
    db.commit()
    db.close()
    return f"Saved note '{subject}'"


def _send_message(to, body):
    from waku.tools.messages import _ensure_paths
    import re

    home = _ensure_paths()
    outbox = home / "outbox"
    outbox.mkdir(exist_ok=True)
    safe = re.sub(r"[^a-z0-9_.@-]", "_", to.strip().lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = outbox / f"{ts}-{safe}.txt"
    path.write_text(f"To: {to.strip()}\n\n{body.strip()}\n")
    return f"Drafted message to '{to}' at {path.name} (nothing sent — review in .waku/outbox/)"


_HANDLERS = {
    "create_event": _create_event,
    "save_note": _save_note,
    "send_message": _send_message,
}


# ── JSON-RPC server ──────────────────────────────────────────────────

def _rpc(method: str, params: dict | None, id_: str | int | None):
    """Dispatch one JSON-RPC request. Returns a response dict or None (notification)."""
    if method == "initialize":
        return _ok(id_, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "waku-tools", "version": "1.0.0"},
        })
    if method == "notifications/initialized":
        return None  # no response for notifications
    if method == "tools/list":
        return _ok(id_, {"tools": _TOOLS})
    if method == "tools/call":
        name = params.get("name", "")
        tool = _TOOL_MAP.get(name)
        if not tool:
            return _err(id_, -32602, f"Unknown tool: {name}")
        args = params.get("arguments", {}) or {}
        handler = _HANDLERS.get(name)
        if not handler:
            return _err(id_, -32603, f"No handler for tool: {name}")
        try:
            result = handler(**args)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _ok(id_, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })
        return _ok(id_, {
            "content": [{"type": "text", "text": str(result)}],
        })
    return _err(id_, -32601, f"Method not found: {method}")


def _ok(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def main() -> None:
    """Read JSON-RPC lines from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method", "")
        params = req.get("params")
        id_ = req.get("id")
        resp = _rpc(method, params, id_)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
