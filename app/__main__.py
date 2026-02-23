from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    if len(args) <= 1:
        from app.gui import main as gui_main

        gui_main()
        return 0
    from app.cli import main as cli_main

    return int(cli_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
