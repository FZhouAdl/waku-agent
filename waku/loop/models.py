"""Model access — nine providers, one loop, zero framework.

The loop speaks one dialect: Anthropic's Messages shape (system/messages/tools
in, content blocks out). Providers plug in two ways:

  anthropic wire format (native)     → Anthropic, Kimi/Moonshot, GLM/Z.ai, MiniMax
  openai wire format (thin adapter)  → OpenAI, Google Gemini, DeepSeek, OpenRouter
  custom (subprocess wrapper)        → Snowflake Cortex (wraps the `cortex` CLI)

Pick with WAKU_PROVIDER=anthropic|openai|gemini|deepseek|minimax|kimi|glm|openrouter|snowflake
and set that provider's API key in .env. Override the model ids with WAKU_MODEL /
WAKU_SMALL_MODEL if the defaults below age out — they're just strings. This
matters most for openrouter: it's a single key in front of hundreds of models,
so WAKU_MODEL=<vendor>/<model> (e.g. "google/gemini-3.5-flash") picks whichever
one you want — and its defaults below are $0 ":free" ids, so it works with no
spend at all (rate-limited). The dashboard Settings tab lists the live catalog.

Snowflake Cortex is a CLI-wrapper provider: it shells out to `cortex exec
--format json` instead of making HTTP calls. Auth is handled by the CLI's own
browser OAuth + OS keyring — Waku never sees the token. See coco.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace

from waku.config import Settings


@dataclass(frozen=True)
class Provider:
    kind: str        # 'anthropic' or 'openai' — the wire format
    key_env: str     # which env var holds the key
    base_url: str | None
    model: str       # default main model (the loop)
    small_model: str  # default cheap model (retrieval gate + consolidation)
    # Where to LIST this provider's models (the Settings picker). openai-wire
    # providers get {base_url}/models automatically; set this for providers
    # whose chat endpoint and catalog endpoint differ (e.g. kimi talks the
    # anthropic wire but lists models on its OpenAI-compatible API). The
    # defaults above are just starting points — any listed model is one click.
    catalog_url: str | None = None
    # The two models the chat switcher pins by default for this provider: a
    # flagship (top quality) and a fast one (cheap/low-latency). Distinct from
    # model/small_model — e.g. anthropic's loop default is sonnet-5, but the
    # flagship you'd showcase is opus-4.8. Blank falls back to model/small_model.
    flagship: str = ""
    fast: str = ""

    def default_pair(self) -> list[str]:
        """[flagship, fast], deduped — the switcher's default picks."""
        pair = [self.flagship or self.model, self.fast or self.small_model]
        return list(dict.fromkeys(m for m in pair if m))


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider("anthropic", "ANTHROPIC_API_KEY", None,
                          "claude-sonnet-5", "claude-haiku-4-5-20251001",
                          catalog_url="https://api.anthropic.com/v1/models",
                          flagship="claude-opus-4-8", fast="claude-sonnet-5"),
    # The gpt-5.6 REASONING models (luna/sol/terra) can't use function tools on
    # /v1/chat/completions (they need /v1/responses), so every Waku turn 400s on
    # them. The non-reasoning "chat" line DOES call tools fine; gpt-5.3-chat-latest
    # is the newest concrete one (preferred over the gpt-5-chat-latest alias so a
    # benchmark is reproducible). gpt-4.1-mini is a cheap tool-capable gate.
    # base_url is None (SDK default) so point the picker at OpenAI's catalog.
    "openai":    Provider("openai", "OPENAI_API_KEY", None,
                          "gpt-5.3-chat-latest", "gpt-4.1-mini",
                          catalog_url="https://api.openai.com/v1/models"),
    # one key, every lab's models, and a $0 tier: the default models below are
    # free ids (":free" suffix). Rate-limited (~50 req/day without credits).
    "openrouter": Provider("openai", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                           "nvidia/nemotron-3-super-120b-a12b:free",
                           "google/gemma-4-26b-a4b-it:free"),
    "gemini":    Provider("openai", "GEMINI_API_KEY",
                          "https://generativelanguage.googleapis.com/v1beta/openai/",
                          "gemini-3.5-flash", "gemini-3.1-flash-lite",
                          # Google's Pro tier isn't "gemini-3.5-pro" (that id
                          # 404s); the current Pro is gemini-3.1-pro-preview.
                          flagship="gemini-3.1-pro-preview", fast="gemini-3.5-flash"),
    "deepseek":  Provider("openai", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
                          "deepseek-v4-pro", "deepseek-v4-pro"),
    "minimax":   Provider("anthropic", "MINIMAX_API_KEY", "https://api.minimaxi.com/anthropic",
                          "MiniMax-M3", "MiniMax-M2"),
    # K3 is the flagship default; the gate/summarizer stays on cheap K2.6
    # (the live catalog has no plain "kimi-k2.7" — only -code variants; we
    # checked). Override with WAKU_SMALL_MODEL=kimi-k3 if your key is K3-only.
    "kimi":      Provider("anthropic", "MOONSHOT_API_KEY", "https://api.moonshot.ai/anthropic",
                          "kimi-k3", "kimi-k2.6",
                          catalog_url="https://api.moonshot.ai/v1/models",
                          flagship="kimi-k3", fast="kimi-k2.7-code-highspeed"),
    "glm":       Provider("anthropic", "ZHIPU_API_KEY", "https://api.z.ai/api/anthropic",
                          "glm-5.2", "glm-5-turbo"),
    # xAI Grok on its OpenAI-compatible endpoint. The model ids below are
    # starting points — add XAI_API_KEY and the picker lists the live catalog
    # (the authoritative source); pin whatever the current flagship/fast are.
    "xai":       Provider("openai", "XAI_API_KEY", "https://api.x.ai/v1",
                          "grok-4", "grok-4-fast",
                          catalog_url="https://api.x.ai/v1/models"),
    # Wraps the installed `cortex` code CLI (subprocess + NDJSON parse). kind
    # = "custom" because there's no HTTP endpoint Waku talks directly — the
    # CLI carries its own Snowflake auth (browser OAuth, OS keyring) and
    # emits Claude-Messages-shaped JSON events. No base_url; no new deps.
    "snowflake": Provider("custom", "SNOWFLAKE_USER",
                          base_url=None,
                          model="cortex", small_model="cortex",
                          ),
}


def get_client(settings: Settings):
    """Build the client for settings.provider and fill in default model ids.
    Returns anything with .messages.create(...) in the Anthropic shape."""
    provider = PROVIDERS.get(settings.provider)
    if provider is None:
        raise SystemExit(f"Unknown WAKU_PROVIDER '{settings.provider}'. "
                         f"Pick one of: {', '.join(PROVIDERS)}")

    # .strip() so a trailing newline/space from a copy-paste doesn't corrupt the
    # auth header (headers are latin-1; a stray non-ASCII char errors cryptically).
    api_key = (settings.api_key or os.getenv(provider.key_env, "")).strip()
    if not api_key:
        raise SystemExit(
            f"No API key for provider '{settings.provider}'. "
            f"Set {provider.key_env} in .env (see .env.example)."
        )
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError:
        raise SystemExit(
            f"{provider.key_env} contains a non-ASCII character (e.g. a smart quote "
            f"or arrow from a bad paste). Re-paste the key with no spaces or line breaks."
        )

    settings.model = settings.model or provider.model
    settings.small_model = settings.small_model or provider.small_model
    base_url = settings.base_url or provider.base_url

    # a hung network call must never freeze a turn silently
    timeout = float(os.getenv("WAKU_LLM_TIMEOUT", "120"))

    if provider.kind == "anthropic":
        import anthropic

        kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)
    if provider.kind == "custom":
        return _build_coco_client(settings, provider)
    return OpenAICompatClient(api_key=api_key, base_url=base_url, timeout=timeout)


def _build_coco_client(settings: Settings, provider: Provider):
    """Resolve the `cortex` binary and return a CocoCliClient that shells out to
    `cortex exec --format json` for every turn. Auth is handled by the CLI."""
    bin_path = shutil.which(settings.base_url or os.getenv("CORTEX_BIN", "") or "cortex")
    if not bin_path:
        raise SystemExit(
            f"No Cortex CLI found and {provider.key_env} is not set. "
            f"Install the `cortex` CLI (https://docs.snowflake.com/cortex) or set "
            f"{provider.key_env} in .env (see .env.example)."
        )
    max_turns = int(os.getenv("WAKU_MAX_ITERATIONS", "10"))
    model = settings.model or provider.model
    allowed_raw = os.getenv("CORTEX_ALLOWED", "")
    allowed = [t.strip() for t in allowed_raw.split(",") if t.strip()] or None
    return CocoCliClient(bin_path=bin_path, model=model, max_turns=max_turns, allowed=allowed)


_DENIED_MARKERS = (
    "Tool denied:",
    "headless mode requires",
    "PERMANENTLY blocked",
)


class CocoCliClient:
    """Wraps the installed `cortex` code CLI. Every turn shells out to
    `cortex exec --format json`, which emits an NDJSON stream of typed records
    (`system/init`, `assistant`, `user`, `result`). The adapter parses these
    into Anthropic-shaped Message / stream events that the loop consumes."""

    def __init__(self, *, bin_path: str, model: str, max_turns: int,
                 allowed: list[str] | None, notify=None):
        self._bin = bin_path
        self._model = model
        self._max_turns = max_turns
        self._allowed = allowed
        self._notify = notify or (lambda k, e: None)
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    # ── public API (mirrors anthropic.Anthropic.messages) ──────────────

    def _create(self, *, model, messages, max_tokens, system=None, tools=None):
        """Non-streaming fallback: drain _stream() to completion, return final Message."""
        last = None
        for ev in self._stream(
            model=model, messages=messages, max_tokens=max_tokens,
            system=system, tools=tools,
        ):
            last = ev
        if last is None:
            raise RuntimeError("cortex exec produced no output")
        return last.message

    def _stream(self, *, model, messages, max_tokens, system=None, tools=None):
        """Yields Anthropic-shaped MessageStreamEvent objects, one per NDJSON
        record. Calls notify(kind, payload) for intermediate events so gateways
        can surface thinking, tool calls, denials, and unknown blocks live."""
        argv = [self._bin, "exec", "--format", "json", "--bypass",
                "--max-turns", str(self._max_turns)]
        if self._allowed:
            argv += ["--allowed", ",".join(self._allowed)]
        prompt = _render_prompt(system, messages, tools)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        proc.stdin.write(prompt)
        proc.stdin.close()

        accumulated: list[dict] = []
        tool_name_by_id: dict[str, str] = {}
        for line in proc.stdout:
            evt = json.loads(line)
            kind = evt.get("type")
            if kind == "system":
                self._notify("session", {"session_id": evt.get("session_id", "")})
            elif kind == "assistant":
                for block in evt["message"]["content"]:
                    if block.get("type") == "tool_use":
                        tool_name_by_id[block["id"]] = block.get("name", "")
                    _emit_block(block, tool_name_by_id, self._notify)
                    accumulated.append(block)
            elif kind == "user":
                for block in evt["message"]["content"]:
                    _emit_block(block, tool_name_by_id, self._notify)
            elif kind == "result":
                if evt.get("is_error"):
                    self._notify("error", {"errors": evt.get("errors", []),
                                            "usage": evt.get("usage", {})})
                    yield from self._stream(
                        model=model, messages=messages, max_tokens=max_tokens,
                        system=system, tools=tools,
                    )
                    return
                denials = evt.get("permission_denials") or []
                if denials:
                    self._notify("denials", {"denials": denials})
                yield _final_message_event(accumulated, evt.get("usage", {}))
                return
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"cortex exec failed (exit {proc.returncode}): {stderr.strip()}")
        raise RuntimeError("cortex exec produced no result event")

    # ── helpers ────────────────────────────────────────────────────────

    def _build_argv(self, model, max_tokens, tools):
        """Exposed for tests — builds the argument list without spawning."""
        argv = [self._bin, "exec", "--format", "json", "--bypass",
                "--max-turns", str(self._max_turns)]
        if self._allowed:
            argv += ["--allowed", ",".join(self._allowed)]
        return argv


def _render_prompt(system, messages, tools=None) -> str:
    """Build a plain-text prompt from the loop's system + messages list for
    `cortex exec`. The CLI reads from stdin, so this is just a text blob."""
    parts = []
    if system:
        parts.append(f"[SYSTEM]\n{system}")
    if tools:
        schemas = "\n".join(
            f"  {t.get('name')}: {t.get('description', '')} "
            f"({json.dumps(t.get('input_schema', {}))})"
            for t in tools
        )
        parts.append(f"[TOOLS]\n{schemas}")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role.upper()}]\n{content}")
        elif isinstance(content, list):
            texts = [b.get("text", "") or "" for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            parts.append(f"[{role.upper()}]\n" + "\n".join(texts))
    return "\n\n".join(parts)


def _emit_block(block: dict, tool_name_by_id: dict[str, str], notify):
    """Forward one Cortex content block as a notify() call so the gateway
    can render it. Handles text, thinking, tool_use, tool_result (including
    denials), ask_user_question, and forwards unknown types as raw events."""
    t = block.get("type")
    if t == "thinking":
        notify("thinking", {"delta": block.get("thinking", "")})
    elif t == "text":
        notify("text", {"delta": block.get("text", "")})
    elif t == "tool_use":
        notify("tool", {"id": block.get("id", ""),
                         "name": block.get("name", ""),
                         "input": block.get("input", {})})
    elif t == "tool_result":
        output = block.get("content", "")
        name = tool_name_by_id.get(block.get("tool_use_id", ""), "<unknown>")
        denied = isinstance(output, str) and any(m in output for m in _DENIED_MARKERS)
        notify("tool_result", {"id": block.get("tool_use_id", ""),
                                 "name": name,
                                 "output": output,
                                 "denied": denied})
        if denied:
            notify("denial", {"id": block.get("tool_use_id", ""),
                               "name": name,
                               "reason": output})
    elif t == "ask_user_question":
        notify("question", {"questions": block.get("questions", []),
                             "raw": block})
    elif t:
        notify("cortex_block", {"type": t, "raw": block})


def _final_message_event(accumulated: list[dict], usage: dict):
    """Build the final Anthropic Message from accumulated assistant blocks.
    Cortex `exec` with `--bypass` handles its own tool execution internally
    (the NDJSON stream carries tool_use → tool_result pairs). Waku should
    never try to dispatch cortex's internal tools through its own
    ToolRegistry. So we return only text blocks and always set
    stop_reason="end_turn" to let the loop finish the turn cleanly."""
    texts = [b.get("text", "") for b in accumulated
             if b.get("type") == "text"]
    blocks = [SimpleNamespace(type="text", text=t) for t in texts if t]
    return _make_stream_event(blocks, usage)


def _make_stream_event(blocks: list, usage: dict):
    """Shared helper: wrap blocks + usage into an Anthropic-shaped
    MessageStreamEvent the loop can consume."""
    usage_obj = SimpleNamespace(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )
    message = SimpleNamespace(
        stop_reason="end_turn",
        content=blocks,
        usage=usage_obj,
    )

    class _Event:
        def __init__(self, msg):
            self.message = msg
        def get_final_message(self):
            return self.message

    return _Event(message)


class OpenAICompatClient:
    """Speaks the Anthropic Messages shape the loop expects, backed by an
    OpenAI-style chat.completions API. ~60 lines is the entire difference
    between the two wire formats — worth reading once.
    """

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 120.0):
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def _to_openai(self, *, model, messages, max_tokens, system=None, tools=None) -> dict:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                oai_messages.append({"role": message["role"], "content": content})
            elif message["role"] == "assistant":
                # anthropic content blocks → assistant text + tool_calls
                text = "".join(b.text for b in content if getattr(b, "type", "") == "text")
                calls = []
                for b in content:
                    if getattr(b, "type", "") != "tool_use":
                        continue
                    call = {"id": b.id, "type": "function",
                            "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                    extra = getattr(b, "extra", None)   # Gemini thought_signature
                    if extra:
                        call["extra_content"] = extra
                    calls.append(call)
                entry: dict = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                oai_messages.append(entry)
            else:
                # anthropic tool_result blocks → one 'tool' message each
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })

        kwargs: dict = {"model": model, "messages": oai_messages,
                        "max_completion_tokens": max_tokens}
        if tools:
            kwargs["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}}
                for t in tools
            ]
        return kwargs

    def _call(self, kwargs: dict, **extra):
        """Run chat.completions.create with the max_tokens key-name fallback
        (older OpenAI-compatible endpoints only know max_tokens, not the newer
        max_completion_tokens). Only retry when the error is ABOUT that param —
        retrying on any error masked the real failure (e.g. a gpt-5.x call would
        fail for some other reason, then the max_tokens retry buried it under a
        confusing 'use max_completion_tokens' message)."""
        try:
            return self._client.chat.completions.create(**kwargs, **extra)
        except Exception as exc:
            m = str(exc).lower()
            if "max_completion_tokens" not in m and "max_tokens" not in m:
                raise
            k = dict(kwargs)
            k["max_tokens"] = k.pop("max_completion_tokens", None)
            return self._client.chat.completions.create(**k, **extra)

    def _create(self, *, model, messages, max_tokens, system=None, tools=None):
        response = self._call(self._to_openai(
            model=model, messages=messages, max_tokens=max_tokens, system=system, tools=tools))
        if not getattr(response, "choices", None):
            # some OpenAI-compatible endpoints (e.g. OpenRouter on a rate
            # limit) return 200 with an error body and no choices: surface
            # that message instead of dying on a TypeError below
            err = getattr(response, "error", None) or "endpoint returned no choices"
            raise RuntimeError(f"{model}: {err}")
        choice = response.choices[0].message
        blocks = []
        if choice.content:
            blocks.append(SimpleNamespace(type="text", text=choice.content))
        for call in choice.tool_calls or []:
            blocks.append(SimpleNamespace(
                type="tool_use", id=call.id, name=call.function.name,
                input=json.loads(call.function.arguments or "{}"),
                # Gemini's thinking models attach a thought_signature here and
                # REQUIRE it echoed back with the tool call next turn, else the
                # follow-up 400s ("missing a thought_signature"). Carry it so
                # _to_openai can put it back. None for every other provider.
                extra=getattr(call, "extra_content", None),
            ))
        usage = getattr(response, "usage", None)
        return SimpleNamespace(
            stop_reason="tool_use" if choice.tool_calls else "end_turn",
            usage=SimpleNamespace(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            ),
            content=blocks,
        )

    def _stream(self, *, model, messages, max_tokens, system=None, tools=None):
        """Anthropic-shaped streaming over an OpenAI chat.completions stream —
        same two-format bridge as _create, but yielding text as it arrives.
        Used by the loop when stream=True (e.g. the dashboard's live chat)."""
        kwargs = self._to_openai(
            model=model, messages=messages, max_tokens=max_tokens, system=system, tools=tools)
        return _OpenAIStream(self, kwargs)


class _OpenAIStream:
    """A context manager mirroring anthropic's messages.stream(): iterate
    .text_stream for text deltas, then .get_final_message() for the assembled
    Anthropic-shaped response (text + reassembled tool calls + usage)."""

    def __init__(self, client: OpenAICompatClient, kwargs: dict):
        self._client = client
        self._kwargs = kwargs
        self._text: list[str] = []
        self._tools: dict[int, dict] = {}   # index → {id, name, args}
        self._usage = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        stream = self._client._call(
            self._kwargs, stream=True, stream_options={"include_usage": True})
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self._usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                self._text.append(delta.content)
                yield delta.content
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = self._tools.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

    def get_final_message(self):
        blocks = []
        text = "".join(self._text)
        if text:
            blocks.append(SimpleNamespace(type="text", text=text))
        for slot in self._tools.values():
            blocks.append(SimpleNamespace(
                type="tool_use", id=slot["id"], name=slot["name"],
                input=json.loads(slot["args"] or "{}")))
        usage = self._usage
        return SimpleNamespace(
            stop_reason="tool_use" if self._tools else "end_turn",
            usage=SimpleNamespace(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0)),
            content=blocks,
        )
