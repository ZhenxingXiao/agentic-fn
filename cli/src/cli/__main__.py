"""Allow ``python -m cli`` / ``uv run python -m cli`` entry."""

from cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
