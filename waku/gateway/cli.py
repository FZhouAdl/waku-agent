"""CLI gateway — the zero-setup way to talk to your Waku.

The Gateway Interface box: a gateway only moves text in and out; everything
interesting happens in the loop. The Telegram gateway is the same ~60 lines
with polling instead of input().
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from waku.app import Waku

console = Console()


def _observer(kind: str, event: dict) -> None:
    """Show the loop's internals live — the video's 'transparent harness' beat."""
    if kind == "tool":
        console.print(f"  [dim]tool · {event['tool']}({event['args']}) → {event['output'][:80]}[/dim]")
    elif kind == "gate":
        console.print(f"  [dim]gate · {event['decision']} — {event.get('reason','')}[/dim]")
    elif kind == "consolidation":
        console.print(f"  [dim]memory · consolidated {event['new_facts']} fact(s) from recent chats[/dim]")
    elif kind == "thinking":
        preview = event.get("delta", "")[:160]
        console.print(f"  [dim]» thinking · {preview}[/dim]")
    elif kind == "question":
        questions = event.get("raw", {}).get("questions", [])
        qs = "; ".join(questions) if questions else str(event.get("raw", ""))[:160]
        console.print(f"  [yellow]? cortex asks · {qs}[/yellow]")
    elif kind == "tool_result":
        name = event.get("name", "<tool>")
        denied = event.get("denied", False)
        if denied:
            reason = event.get("output", "")[:120]
            console.print(f"  [red]! cortex denied {name}: {reason}[/red]")
        else:
            output = event.get("output", "")[:80]
            console.print(f"  [dim]← {name} ok · {output}[/dim]")
    elif kind == "denial":
        name = event.get("name", "<tool>")
        reason = event.get("reason", "")[:120]
        console.print(f"  [red]! cortex denied {name}: {reason}[/red]")
    elif kind == "denials":
        for d in event.get("denials", []):
            console.print(f"  [red]! cortex denied: {str(d)[:120]}[/red]")
    elif kind == "cortex_block":
        t = event.get("type", "?")
        raw = str(event.get("raw", ""))[:120]
        console.print(f"  [dim][cortex: {t}] {raw}[/dim]")
    elif kind == "session":
        pass   # silently stash session_id for future cortex resume
    elif kind == "error":
        errors = event.get("errors", ["unknown error"])
        console.print(f"  [red]! cortex error · {errors[0] if errors else 'unknown'}[/red]")


def main() -> None:
    waku = Waku()
    waku.session.session_id = "terminal"   # its own conversation thread in the inbox
    console.print(Panel.fit(
        "[bold]Waku[/bold] — local, yours, transparent.\n"
        f"home: {waku.settings.home.resolve()}   model: {waku.settings.model}\n"
        "Ctrl-D or /quit to exit.",
        border_style="cyan",
    ))
    while True:
        try:
            user_message = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_message:
            continue
        if user_message in ("/quit", "/exit"):
            break
        result = waku.respond(user_message, observer=_observer, source="cli")
        console.print(f"[bold green]waku ›[/bold green] {result.reply}\n")
    console.print("[dim]bye — your memory stays in state.db[/dim]")


if __name__ == "__main__":
    main()
