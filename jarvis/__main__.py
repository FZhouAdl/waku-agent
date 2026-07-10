"""Entrypoints:

  python -m jarvis                       chat in the terminal (default)
  python -m jarvis voice                 talk to it (needs the [voice] extra)
  python -m jarvis telegram              phone → laptop (needs TELEGRAM_BOT_TOKEN)
  python -m jarvis skill install <url>   install a community skill
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        from jarvis.gateway.cli import main as cli_main

        cli_main()
    elif args[0] == "voice":
        from jarvis.gateway.voice import main as voice_main

        voice_main()
    elif args[0] == "telegram":
        from jarvis.gateway.telegram import main as tg_main

        tg_main()
    elif args[0] == "skill" and len(args) >= 3 and args[1] == "install":
        from jarvis.memory.procedural.installer import install

        install(args[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
