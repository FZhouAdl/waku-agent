"""Offline tests for the CocoCliClient (Snowflake Cortex CLI wrapper).

All tests are offline — they use captured NDJSON fixtures from real
`cortex exec --format json` runs. No subprocess is ever spawned; the
fixtures are the source of truth for the output schema observed in CLI
v1.1.41 (system/init, assistant, user, result events; content blocks of
text, thinking, tool_use, tool_result, ask_user_question).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from waku.loop.models import (
    CocoCliClient,
    _emit_block,
    _final_message_event,
    _render_prompt,
)

FIXTURES = Path(__file__).with_name("fixtures")


# ── helpers ───────────────────────────────────────────────────────────

def _load_ndjson(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / name).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def _ndjson_lines(fixture_name: str) -> list[str]:
    """Return the raw JSON-string lines for a fixture (one per NDJSON record)."""
    records = _load_ndjson(fixture_name)
    return [json.dumps(r) for r in records]


def _stream_for_lines(lines: list[str]):
    """Create a fake stdout iterable that yields the given lines."""
    class _FakeStdout:
        def __iter__(self):
            return iter(lines)
    return _FakeStdout()


def _fake_popen(stdout_lines: list[str], returncode: int = 0):
    """Create a fake subprocess.Popen substitute."""
    class _FakePopen:
        def __init__(self, *a, **kw):
            self.stdin = MagicMock()
            self.stdout = _stream_for_lines(stdout_lines)
            self.stderr = MagicMock()
            self.returncode = returncode

        def wait(self):
            pass

    return _FakePopen


def _make_client():
    """Build a test CocoCliClient with a no-op notify."""
    return CocoCliClient(
        bin_path="cortex", model="cortex", max_turns=3, allowed=None,
    )


# ── fixture loading ──────────────────────────────────────────────────

@pytest.fixture
def chat_fixture():
    return _load_ndjson("coco_chat.ndjson")


@pytest.fixture
def thinking_fixture():
    return _load_ndjson("coco_thinking.ndjson")


@pytest.fixture
def denial_fixture():
    return _load_ndjson("coco_denial.ndjson")


@pytest.fixture
def error_fixture():
    return _load_ndjson("coco_error.ndjson")


@pytest.fixture
def question_fixture():
    return _load_ndjson("coco_question.ndjson")


@pytest.fixture
def test_client():
    return _make_client()


# ── NDJSON parsing (buffered _create path) ───────────────────────────

def test_coco_parses_ndjson_stream(test_client, chat_fixture):
    notify = MagicMock()
    test_client._notify = notify
    lines = _ndjson_lines("coco_chat.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        events = list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "hi"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    assert len(events) == 1
    msg = events[0].get_final_message()
    assert msg.stop_reason == "end_turn"
    texts = [b.text for b in msg.content if b.type == "text"]
    assert "Hello!" in "".join(texts)
    assert msg.usage.input_tokens == 1000
    assert msg.usage.output_tokens == 50


def test_coco_handles_tool_use_block(test_client, thinking_fixture):
    """Cortex handles its own tools internally — Waku should only get text back."""
    lines = _ndjson_lines("coco_thinking.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        events = list(test_client._stream(
            model="cortex",
            messages=[{"role": "user", "content": "schedule standup"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    msg = events[0].get_final_message()
    texts = [b for b in msg.content if b.type == "text"]
    tools = [b for b in msg.content if b.type == "tool_use"]
    assert len(texts) >= 1
    assert "I'll create that event" in texts[0].text
    assert len(tools) == 0  # cortex tools aren't forwarded to the loop
    assert msg.stop_reason == "end_turn"


# ── stream forwarding guarantee ───────────────────────────────────────

def test_coco_stream_forwards_every_block_kind(test_client, thinking_fixture):
    notify = MagicMock()
    test_client._notify = notify
    lines = _ndjson_lines("coco_thinking.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "schedule"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    kinds_called = {c[0][0] for c in notify.call_args_list}
    assert "thinking" in kinds_called
    assert "text" in kinds_called
    assert "tool" in kinds_called
    assert "session" in kinds_called
    thinking_call = [c for c in notify.call_args_list if c[0][0] == "thinking"]
    assert len(thinking_call) >= 1
    assert "scheduling" in thinking_call[0][0][1]["delta"].lower()


def test_coco_stream_forwards_question(test_client, question_fixture):
    notify = MagicMock()
    test_client._notify = notify
    lines = _ndjson_lines("coco_question.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        list(test_client._stream(
            model="cortex",
            messages=[{"role": "user", "content": "schedule meeting"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    question_calls = [c for c in notify.call_args_list if c[0][0] == "question"]
    assert len(question_calls) >= 1
    assert question_calls[0][0][1]["questions"] == [
        "What time would you like the meeting?",
        "Who should attend?",
    ]


def test_coco_stream_preserves_unknown_blocks(test_client):
    notify = MagicMock()
    test_client._notify = notify
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s-unk"}),
        json.dumps({"type": "assistant", "message": {
            "id": "m", "type": "message", "role": "assistant",
            "content": [{"type": "web_fetch", "url": "https://example.com", "result": "ok"}],
        }}),
        json.dumps({"type": "result", "subtype": "success", "is_error": False,
                     "usage": {"input_tokens": 1, "output_tokens": 1},
                     "permission_denials": []}),
    ]
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "fetch"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    cortex_calls = [c for c in notify.call_args_list if c[0][0] == "cortex_block"]
    assert len(cortex_calls) == 1
    assert cortex_calls[0][0][1]["type"] == "web_fetch"
    assert "example.com" in str(cortex_calls[0][0][1]["raw"])


# ── permission denial honesty ─────────────────────────────────────────

def test_coco_surfaces_tool_denial_per_tool(test_client, denial_fixture):
    notify = MagicMock()
    test_client._notify = notify
    lines = _ndjson_lines("coco_denial.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "delete file"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    tool_result_calls = [c for c in notify.call_args_list if c[0][0] == "tool_result"]
    denial_calls = [c for c in notify.call_args_list if c[0][0] == "denial"]
    assert len(tool_result_calls) >= 1
    tr_call = tool_result_calls[0][0][1]
    assert tr_call["denied"] is True
    assert tr_call["name"] == "bash"
    assert len(denial_calls) >= 1
    d_call = denial_calls[0][0][1]
    assert d_call["name"] == "bash"
    assert "headless mode" in d_call["reason"]


def test_coco_surfaces_permission_denials_rollup(test_client, denial_fixture):
    notify = MagicMock()
    test_client._notify = notify
    lines = _ndjson_lines("coco_denial.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "delete"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    denials_calls = [c for c in notify.call_args_list if c[0][0] == "denials"]
    assert len(denials_calls) == 1
    assert len(denials_calls[0][0][1]["denials"]) >= 1
    assert denials_calls[0][0][1]["denials"][0]["tool_name"] == "bash"


def test_coco_synthetic_user_event_forwarded(test_client, thinking_fixture):
    notify = MagicMock()
    test_client._notify = notify
    lines = _ndjson_lines("coco_thinking.ndjson")
    FakePopen = _fake_popen(lines)
    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "x"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    tool_result_calls = [c for c in notify.call_args_list if c[0][0] == "tool_result"]
    assert len(tool_result_calls) == 1
    assert tool_result_calls[0][0][1]["name"] == "create_event"
    assert "Created" in tool_result_calls[0][0][1]["output"]


# ── retry on error ────────────────────────────────────────────────────

def test_coco_retries_on_error_event(test_client, error_fixture):
    notify = MagicMock()
    test_client._notify = notify
    error_lines = [json.dumps(e) for e in error_fixture]
    chat_lines = _ndjson_lines("coco_chat.ndjson")
    call_count = [0]

    def _build_lines():
        call_count[0] += 1
        return chat_lines if call_count[0] > 1 else error_lines

    class _FakePopen:
        def __init__(self, *a, **kw):
            self.stdin = MagicMock()
            self.stdout = _stream_for_lines(_build_lines())
            self.stderr = MagicMock()
            self.returncode = 0
        def wait(self):
            pass

    original = subprocess.Popen
    subprocess.Popen = _FakePopen
    try:
        events = list(test_client._stream(
            model="cortex", messages=[{"role": "user", "content": "x"}],
            max_tokens=2048))
    finally:
        subprocess.Popen = original
    assert call_count[0] >= 2
    error_calls = [c for c in notify.call_args_list if c[0][0] == "error"]
    assert len(error_calls) >= 1
    assert error_calls[0][0][1]["errors"] == ["Max tool call iterations reached"]
    assert len(events) == 1
    msg = events[0].get_final_message()
    texts = [b.text for b in msg.content if b.type == "text"]
    assert "Hello!" in "".join(texts)


# ── command building ──────────────────────────────────────────────────

def test_coco_client_builds_command(test_client):
    argv = test_client._build_argv(model="cortex", max_tokens=2048, tools=None)
    assert argv[0] == "cortex"
    assert "exec" in argv
    assert "--format" in argv
    assert "json" in argv
    assert "--bypass" in argv
    assert "--max-turns" in argv
    assert str(test_client._max_turns) in argv


def test_coco_client_builds_command_with_allowed():
    client = CocoCliClient(
        bin_path="/usr/local/bin/cortex", model="cortex-agent",
        max_turns=5, allowed=["Read", "Edit", "Bash(git *)"],
    )
    argv = client._build_argv(model="cortex-agent", max_tokens=2048, tools=None)
    assert argv[0] == "/usr/local/bin/cortex"
    assert "--bypass" in argv
    assert "--allowed" in argv
    allowed_idx = argv.index("--allowed")
    allowed_val = argv[allowed_idx + 1]
    assert "Read" in allowed_val
    assert "Edit" in allowed_val
    assert "Bash(git *)" in allowed_val


# ── _emit_block unit tests ────────────────────────────────────────────

def test_emit_block_text():
    notify = MagicMock()
    _emit_block({"type": "text", "text": "Hello"}, {}, notify)
    notify.assert_called_once_with("text", {"delta": "Hello"})


def test_emit_block_thinking():
    notify = MagicMock()
    _emit_block({"type": "thinking", "thinking": "hmm..."}, {}, notify)
    notify.assert_called_once_with("thinking", {"delta": "hmm..."})


def test_emit_block_tool_use():
    notify = MagicMock()
    _emit_block({"type": "tool_use", "id": "t1", "name": "read",
                 "input": {"path": "/x"}}, {}, notify)
    notify.assert_called_once_with(
        "tool", {"id": "t1", "name": "read", "input": {"path": "/x"}},
    )


def test_emit_block_tool_result_ok():
    notify = MagicMock()
    _emit_block({"type": "tool_result", "tool_use_id": "t1", "content": "done"},
                {"t1": "read"}, notify)
    assert len(notify.call_args_list) == 1
    call = notify.call_args_list[0]
    assert call[0][0] == "tool_result"
    assert call[0][1]["denied"] is False
    assert call[0][1]["name"] == "read"


def test_emit_block_tool_result_denied():
    notify = MagicMock()
    _emit_block(
        {"type": "tool_result", "tool_use_id": "t2",
         "content": "Tool denied: headless mode requires --allowed-tools"},
        {"t2": "bash"}, notify,
    )
    tr_call = [c for c in notify.call_args_list if c[0][0] == "tool_result"]
    assert len(tr_call) == 1
    assert tr_call[0][0][1]["denied"] is True
    assert tr_call[0][0][1]["name"] == "bash"
    denial_call = [c for c in notify.call_args_list if c[0][0] == "denial"]
    assert len(denial_call) == 1
    assert denial_call[0][0][1]["name"] == "bash"
    assert "headless" in denial_call[0][0][1]["reason"]


def test_emit_block_permanently_blocked():
    notify = MagicMock()
    _emit_block(
        {"type": "tool_result", "tool_use_id": "t3",
         "content": "This tool is PERMANENTLY blocked for this entire session"},
        {"t3": "write"}, notify,
    )
    denial_call = [c for c in notify.call_args_list if c[0][0] == "denial"]
    assert len(denial_call) == 1


def test_emit_block_unknown():
    notify = MagicMock()
    _emit_block({"type": "custom_future", "data": 42}, {}, notify)
    expected = {"type": "custom_future", "raw": {"type": "custom_future", "data": 42}}
    notify.assert_called_once_with("cortex_block", expected)


# ── _final_message_event ──────────────────────────────────────────────

def test_final_message_event_text():
    msg = _final_message_event(
        [{"type": "text", "text": "Hello world"}],
        {"input_tokens": 10, "output_tokens": 2},
    ).message
    assert msg.stop_reason == "end_turn"
    assert len(msg.content) == 1
    assert msg.content[0].text == "Hello world"
    assert msg.usage.input_tokens == 10


def test_final_message_event_drops_tools():
    """Cortex handles tool execution internally — tool_use blocks must NOT
    reach the loop's ToolRegistry (which doesn't know cortex's tools)."""
    msg = _final_message_event(
        [{"type": "tool_use", "id": "tu1", "name": "bash",
          "input": {"command": "ls"}},
         {"type": "text", "text": "Done"}],
        {"input_tokens": 5, "output_tokens": 3},
    ).message
    assert msg.stop_reason == "end_turn"
    assert len(msg.content) == 1
    assert msg.content[0].text == "Done"


# ── _render_prompt ────────────────────────────────────────────────────

def test_render_prompt_simple():
    out = _render_prompt("You are helpful.", [{"role": "user", "content": "Hi"}])
    assert "[SYSTEM]" in out
    assert "You are helpful." in out
    assert "[USER]" in out
    assert "Hi" in out


def test_render_prompt_with_tools():
    out = _render_prompt(
        "System prompt",
        [{"role": "user", "content": "create event"}],
        tools=[{"name": "create_event", "description": "Create a calendar event",
                "input_schema": {"type": "object"}}],
    )
    assert "[TOOLS]" in out
    assert "create_event" in out
    assert "Create a calendar event" in out


def test_render_prompt_with_content_blocks():
    out = _render_prompt("System prompt", [
        {"role": "assistant", "content": [
            {"type": "text", "text": "Hello"},
        ]},
    ])
    assert "[ASSISTANT]" in out
    assert "Hello" in out
    assert "[USER]" not in out
