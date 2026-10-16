# agentic-fn

Core library for the agentic-fn monorepo: explicit `State`, capability/effect
functions, and production-ready sync/async pipelines.

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

Optional model adapter extras:

```bash
uv sync --extra openai
uv sync --extra models-all
```

### Model IO (Router-first)

All model traffic goes through `ModelRuntime` + `DefaultModelRouter`.
Pipeline steps use logical names (`chat.default`); vendors are registered
clients (`openai`, `dashscope_compat`, `laya`, …).

```python
from agentic_fn import Context, Message, ModelRequest, demo_runtime

runtime = demo_runtime()
ctx = Context(models=runtime)
resp = runtime.complete(ModelRequest.chat("chat.default", Message.user("hi")), ctx)
```

See the repository root README and `docs/pipeline-design.md` for architecture.
