# cli

Command-line playground for the local [`agentic-fn`](../agentic-fn) framework.

Use it to try sync/async pipelines, loops, parallel merge, effect retries, soft
errors, record/replay, and picologging-backed log effects - without building a
full agent app.

## Prerequisites

- Python `>= 3.12`
- [uv](https://github.com/astral-sh/uv)

## Setup

From the repository root:

```bash
cd cli
uv sync
```

If sync fails because a mirror (for example Tsinghua tuna) returns HTTP 403,
point uv at the official index for this shell session:

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.org/simple"
uv sync
```

This creates `.venv/` and installs:

- this `cli` package (editable)
- `agentic-fn` from `../agentic-fn` (editable path dependency)
- `picologging` (backend for the `log-effect` demo)

## Start / run

List demos:

```bash
uv run cli list
```

Run a single demo:

```bash
uv run cli run seq
uv run cli run loop
uv run cli run parallel
uv run cli run retry
uv run cli run soft-error
uv run cli run replay
uv run cli run async-seq
uv run cli run log-effect
```

Run all demos:

```bash
uv run cli run --all
```

Equivalent module form:

```bash
uv run python -m cli list
uv run python -m cli run seq
```

## Demo map

| Name | What it shows |
|------|----------------|
| `seq` | Sequential capabilities (`value` 3 → 8) |
| `loop` | Bounded loop until a state predicate |
| `parallel` | Fan-out + `MergeByKeys` |
| `retry` | Idempotent `@effect` retries on `TransientError` |
| `soft-error` | `ErrorAction.CONTINUE` then keep going |
| `replay` | `RECORD` live effect, then `REPLAY` offline |
| `async-seq` | `AsyncPipeline` with an awaitable capability |
| `log-effect` | Log side effects via `PicologgingLogSink` + `make_log_effect` / `ctx.log` |

Sources live under `src/cli/demos/` and include English comments explaining the
framework contracts.

## Project layout

```text
cli/
├── pyproject.toml          # uv project; depends on ../agentic-fn + picologging
├── README.md
└── src/cli/
    ├── __main__.py         # python -m cli
    ├── app.py              # argparse entry (console script: cli)
    ├── printing.py         # shared terminal output helpers
    ├── sinks/              # third-party LogSink adapters (picologging, ...)
    └── demos/              # one module per demo
```

## Notes

- Prefer `uv run ...` so the project virtualenv is used automatically.
- After changing `agentic-fn`, re-run demos immediately - the editable install
  picks up library edits without reinstalling.
- Logging stays pluggable: agentic-fn emits to `LogSink`; picologging is only
  one concrete backend used by the CLI demo (`src/cli/sinks/picologging_sink.py`).
