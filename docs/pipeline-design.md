# Pipeline 设计（生产级）

> 状态：草案 v0.1 · 对应仓库主张：函数优先、副作用隔离、可追溯、可进企业生产  
> 范围：核心编排层；不含具体 LLM / 向量库适配器实现

## 1. 目标与非目标

### 目标

- 用**显式状态 + 可组合步骤**表达 Agent 与知识库流水线（串行 / 并行 / 分支 / 循环）
- **Sync / Async** 两类 Pipeline，规则清晰，避免混用踩坑
- 默认具备：超时、重试、取消、预算、步骤级可观测、录制与回放钩子
- API 面小，类型友好（`py.typed`），便于企业二次封装

### 非目标（v1）

- 不内置分布式工作流引擎（Airflow / Temporal 级调度）
- 不强制 DAG 可视化编辑器
- 不绑定具体 LLM / 存储 SDK

---

## 2. 分层总览

```text
┌─────────────────────────────────────────────────────────┐
│  Loop Engineer / 业务 Pipeline 配方（ReAct、Parse…）      │
├─────────────────────────────────────────────────────────┤
│  Pipeline（SyncPipeline / AsyncPipeline）                 │
│    组合子：seq | parallel | branch | loop | map           │
├─────────────────────────────────────────────────────────┤
│  Step = Capability | Effect（统一 StepSpec + Policy）     │
├─────────────────────────────────────────────────────────┤
│  Runtime：Context | EffectRuntime | Harness | Tracer      │
└─────────────────────────────────────────────────────────┘
```

**原则：** Pipeline 只编排；可靠性与权限由 Runtime / Harness 提供；业务函数保持「小而可测」。

---

## 3. 核心类型

### 3.1 State

状态是步骤间唯一数据通道，必须可序列化（至少支持 JSON / msgpack 友好子集）。

```python
@dataclass(frozen=True, slots=True)
class State:
    """业务载荷 + 框架元数据分离。"""
    data: Mapping[str, Any]          # 业务字段（建议 TypedDict / 领域 State 子类）
    meta: StateMeta                  # run_id、step 游标、版本、标签等

@dataclass(frozen=True, slots=True)
class StateMeta:
    schema_version: str              # 状态契约版本，便于演进与回放
    run_id: str
    parent_run_id: str | None
    labels: Mapping[str, str]
    # 可选：checksum / 业务 trace_id
```

**约定：**

| 规则 | 说明 |
|------|------|
| 不可变优先 | 步骤返回**新 State**（`dataclasses.replace` / 深拷贝策略由实现定义） |
| 禁止塞密钥 | API Key、密码不得进入 `data`；经 EffectRuntime / 密钥提供者注入 |
| 大对象外置 | 文件字节、大向量放入 ObjectRef，State 只存引用与摘要 |
| 版本字段 | `schema_version` 变更需提供迁移或拒绝回放 |

企业可继承 / 包装 `State`，但进入 Pipeline 的边界类型保持统一。

### 3.2 Context（运行上下文）

`Context` **不是**业务状态，是一次执行的控制面：

```python
class Context(Protocol):
    run_id: str
    cancelled: bool                          # 协作式取消
    deadline: float | None                   # 单调时钟截止时间
    budget: Budget                           # token / 调用次数 / 费用上限
    clock: Clock
    rng: Random                              # 可注入，便于回放
    effects: EffectRuntime
    tracer: Tracer
    hooks: HookBus                           # callback / 内部事件
    harness: Harness | None

    def check(self) -> None:
        """取消 / 超时 / 预算耗尽时抛出框架异常。"""

    def query(self) -> PipelineSnapshot:
        """任意时刻只读快照（线程/任务安全）。"""
```

`State` 在步骤间传递；`Context` 贯穿整次 `run`，步骤通过参数或闭包获得（推荐显式传入，避免隐式全局）。

### 3.3 两类函数

```python
# 同步
CapabilityFn = Callable[[State, Context], State]
EffectFn     = Callable[[State, Context], State]   # 仅允许经 effects 做 IO

# 异步
AsyncCapabilityFn = Callable[[State, Context], Awaitable[State]]
AsyncEffectFn     = Callable[[State, Context], Awaitable[State]]
```

| | Capability | Effect |
|--|------------|--------|
| 允许 | 纯计算、本地变换、确定性决策 | LLM、HTTP、文件、DB、日志外发 |
| 禁止 | 网络 / 磁盘 / 进程外调用 | 绕过 `EffectRuntime` 的裸 IO（规范约束 + 可选审计） |
| 测试 | 无 mock 即可单测 | 用 FakeEffectRuntime / 录制回放 |
| 重试 | 默认可安全重试 | 必须声明幂等性（见 Policy） |

实现上可用 `@capability` / `@effect(idempotent=True)` 装饰器打标，供 Pipeline 校验与 trace 分类。

### 3.4 StepSpec（步骤描述）

步骤 = 函数 + 策略 + 元数据，而不是「裸函数扔进 list」。

```python
@dataclass(frozen=True, slots=True)
class StepSpec:
    name: str                              # 稳定名，用于 trace / 权限 / 回放定位
    kind: Literal["capability", "effect"]
    fn: Any                                # Capability/Effect sync 或 async
    policy: StepPolicy
    tags: tuple[str, ...] = ()
```

```python
@dataclass(frozen=True, slots=True)
class StepPolicy:
    timeout: float | None = None           # 秒；None 继承 Pipeline 默认
    retry: RetryPolicy | None = None
    on_error: ErrorAction = ErrorAction.FAIL
    circuit_breaker: CircuitBreakerKey | None = None
    idempotent: bool = False               # Effect 重试前置条件
    critical: bool = True                  # False 时失败可记入 State 并继续（需配合 on_error）
```

```python
class ErrorAction(Enum):
    FAIL = "fail"                 # 失败终止 Pipeline（默认）
    SKIP = "skip"                 # 跳过本步，State 不变（或打标）
    FALLBACK = "fallback"         # 使用 fallback 步骤 / 默认 State 补丁
    CONTINUE = "continue"         # 将错误写入 State.data["_errors"] 并继续
```

---

## 4. 组合子（Pipeline 图原语）

v1 只提供少量原语；复杂 Agent 模式由原语拼出。

| 原语 | 语义 | Sync | Async |
|------|------|:----:|:-----:|
| `seq(a, b, …)` | 严格顺序 | ✓ | ✓ |
| `parallel(a, b, …, merge=…)` | 扇出后汇合 | ✓* | ✓ |
| `branch(pred, then, else_)` | 条件分支 | ✓ | ✓ |
| `loop(body, until, max_iter)` | 循环（ReAct 核心） | ✓ | ✓ |
| `map(fn, over=key, concurrency=n)` | 对集合映射 | ✓* | ✓ |

\* Sync 的 `parallel` / `map`：内部用有界线程池执行**已声明的 Effect**，或仅允许纯 Capability 的 CPU 并行；默认配置需显式开启，避免隐式占满线程。

### 4.1 parallel 合并策略

```python
class MergeStrategy(Protocol):
    def merge(self, base: State, parts: Sequence[State]) -> State: ...
```

内置：

- `MergeByKeys(mapping)`：各分支写入互不冲突的 key
- `Reduce(fn)`：用户自定义归约
- 冲突 key 默认 **报错**（生产安全），可选 `overwrite` / `prefer_left`

设计图中 Memory「特征 & 关系 & 实体抽取」= `parallel` + `MergeByKeys`。

### 4.2 loop

```python
loop(
    body=seq(thought, action, evolution),
    until=lambda s, ctx: s.data.get("final") or ctx.budget.exhausted,
    max_iter=8,                    # 硬上限，防死循环
    on_max_iter=ErrorAction.FAIL,  # 或输出当前 State 并告警
)
```

对应设计图 ReAct：`In → 终止判断 → Thought → Action → Evolution → 回环`。

`max_iter` + Harness 预算 **双保险**，缺一不可。

### 4.3 Pipeline 本体

```python
class SyncPipeline:
    def __init__(self, root: Node, *, defaults: PipelineDefaults): ...
    def run(self, state: State, *, ctx: Context | None = None) -> PipelineResult: ...
    def query(self) -> PipelineSnapshot: ...          # 运行中可调（同 Context.query）

class AsyncPipeline:
    def __init__(self, root: Node, *, defaults: PipelineDefaults): ...
    async def run(self, state: State, *, ctx: Context | None = None) -> PipelineResult: ...
    def query(self) -> PipelineSnapshot: ...
```

`Node` = `StepSpec | Seq | Parallel | Branch | Loop | Map` 的代数数据类型（可用 `@dataclass` 联合或小型访问者执行器）。

**Sync / Async 混用规则（硬约束）：**

1. `SyncPipeline` 的步骤函数必须是同步的；禁止在 Capability/Effect 内 `asyncio.run`。
2. `AsyncPipeline` 可调度 sync Capability（直接调用）与 sync Effect（经 `asyncio.to_thread` 或专用 executor，且受并发上限约束）。
3. 不得在 Sync 步骤中创建未托管的后台任务泄漏出 `run()`。
4. 一次业务调用选定一种 Pipeline 类型；需要桥接时用明确的边界 API（例如 `await async_pipe.run(...)` 的结果再喂给 sync 后处理），而不是嵌套两套事件循环。

---

## 5. 执行模型与韧性

### 5.1 单步执行伪流程

```text
ctx.check()
emit(STEP_START)
acquire circuit / semaphore（若配置）
with timeout(policy.timeout or defaults.timeout):
    for attempt in retry:
        ctx.check()
        if kind == effect:
            out = effects.run(fn, state, ctx, record=...)
        else:
            out = fn(state, ctx)
        validate_state(out)          # schema / 不可变性抽查
        emit(STEP_SUCCESS)
        return out
on failure:
    apply ErrorAction → FAIL | SKIP | FALLBACK | CONTINUE
    emit(STEP_ERROR)
```

### 5.2 RetryPolicy

```python
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff: Backoff = ExponentialBackoff(0.2, 2.0, max_delay=5.0)
    retry_on: tuple[type[BaseException], ...] = (TransientError,)
    # Effect 仅当 idempotent=True 或 attempt==0 才允许自动重试
```

分类异常（框架定义）：

| 异常 | 含义 | 默认可重试 |
|------|------|:----------:|
| `TransientError` | 网络抖、429、超时 | ✓ |
| `PermanentError` | 校验失败、权限拒绝 | ✗ |
| `CancelledError` | 用户/上游取消 | ✗ |
| `BudgetExceeded` | 配额用尽 | ✗ |
| `TimeoutError` | 步骤或全局 deadline | 按策略 |

### 5.3 超时与取消

- **步骤超时** 与 **Pipeline deadline** 取更严者。
- 取消为**协作式**：循环与 EffectRuntime 在边界调用 `ctx.check()`；对线程中的阻塞 IO，依赖超时包装与可中断客户端（文档要求适配器支持）。
- `run()` 结束时 Context 标记 closed，拒绝晚到的回调写入。

### 5.4 熔断（可选但建议企业默认开）

按 `CircuitBreakerKey`（如 `effect:llm:provider-a`）统计连续失败；打开后短时间快速失败，避免雪崩。状态机：Closed → Open → HalfOpen。

### 5.5 幂等与 Exactly-once（务实标准）

分布式 exactly-once 不做承诺。生产约定：

- **至少一次** 执行 + **幂等 Effect**（或带 idempotency key）
- 录制回放时：已成功 Effect 可短路返回录制结果（`ReplayMode.STRICT`）

---

## 6. 可观测、Query、Callback

### 6.1 事件模型

```python
class StepEvent(Enum):
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    STEP_START = "step.start"
    STEP_END = "step.end"
    STEP_RETRY = "step.retry"
    STEP_ERROR = "step.error"
    BRANCH_TAKEN = "branch.taken"
    LOOP_ITER = "loop.iter"
    PARALLEL_JOIN = "parallel.join"
    BUDGET = "budget"
    CANCELLED = "cancelled"
```

每条事件携带：`run_id`、`step_name`、`attempt`、`duration_ms`、`error_type`（可选）、`state_digest`（哈希，默认不落全文）。

### 6.2 Callback

```python
OnEvent = Callable[[Event], None]                 # sync pipeline
AsyncOnEvent = Callable[[Event], Awaitable[None]] # async pipeline
```

约束：callback **不得抛出业务异常打断主路径**（框架捕获并记 `callback_error`）；不得做长时间阻塞（Async 中可 `create_task` 但需有界队列，防反压失控）。

### 6.3 Query

`pipeline.query()` / `ctx.query()` 返回：

```python
@dataclass(frozen=True)
class PipelineSnapshot:
    run_id: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    current_path: tuple[str, ...]     # 如 ("react_loop", "action")
    iteration: int | None
    state_digest: str
    started_at: float
    budget_remaining: BudgetView
    last_error: str | None
```

实现：对当前 State 的引用用锁 / `asyncio.Lock` 保护只读快照拷贝。

### 6.4 OpenTelemetry

`Tracer` Protocol 默认 NoOp；企业注入 OTel：每 Step 一个 span，属性含 `step.name`、`step.kind`、`run.id`。框架不强制依赖 `opentelemetry` 包。

---

## 7. 录制与回放（Harness 基础）

```python
class Recorder(Protocol):
    def record_effect(self, key: EffectKey, input_digest: str, output: State) -> None: ...

class Replay(Protocol):
    def lookup(self, key: EffectKey, input_digest: str) -> State | None: ...
```

| 模式 | 行为 |
|------|------|
| `LIVE` | 正常执行并可选录制 |
| `REPLAY` | Effect 只读录制；缺失则失败 |
| `RECORD` | 强制录制（评测造数） |

Capability 不录结果（应由输入 State 确定性推出）；若发现 Capability 非确定性，回放测试会暴露问题——这是刻意设计。

---

## 8. PipelineDefaults 与构建 API

```python
@dataclass(frozen=True)
class PipelineDefaults:
    timeout: float | None = 30.0
    retry: RetryPolicy | None = None
    max_parallel: int = 8
    executor: Executor | None = None      # Sync parallel 用
    on_event: OnEvent | None = None
    validate_state: bool = True
    fail_fast: bool = True
```

推荐的流畅构建（示意）：

```python
pipe = (
    SyncPipeline.builder(name="response_filter", defaults=...)
    .capability("confidence", confidence_fn, timeout=1.0)
    .capability("security", security_fn)
    .capability("sensitive", sensitive_fn)
    .build()
)

react = (
    SyncPipeline.builder(name="react")
    .loop(
        name="react_loop",
        body=lambda b: b
            .effect("thought", thought_llm, idempotent=False, timeout=60)
            .effect("action", action_tools, idempotent=True)
            .capability("evolution", evolution_fn),
        until=is_done,
        max_iter=8,
    )
    .build()
)

memory = (
    AsyncPipeline.builder(name="memory_save")
    .parallel(
        name="extract",
        branches={
            "feature": effect_fn(feature_extract),
            "relation": effect_fn(relation_extract),
            "entity": effect_fn(entity_extract),
        },
        merge=MergeByKeys(...),
    )
    .capability("filter", filter_fn)
    .capability("summarize", summarize_fn)
    .capability("check", check_fn)
    .effect("save", save_fn, idempotent=True, retry=RetryPolicy(max_attempts=5))
    .build()
)
```

---

## 9. 结果与错误面

```python
@dataclass(frozen=True)
class PipelineResult:
    state: State
    status: Literal["succeeded", "failed", "cancelled"]
    trace: tuple[StepRecord, ...]
    error: FrameworkError | None
    metrics: RunMetrics
```

对外只暴露少量异常类型（见 §5.2），业务异常应包装为 `PermanentError` / `TransientError`，避免裸 `Exception` 导致错误的重试。

---

## 10. 包结构建议（核心库）

```text
agentic_fn/
  __init__.py
  state.py          # State, StateMeta
  context.py        # Context, Budget, Clock
  fn.py             # @capability @effect, 类型别名
  step.py           # StepSpec, StepPolicy, RetryPolicy, ErrorAction
  nodes.py          # Seq, Parallel, Branch, Loop, Map
  pipeline/
    sync.py
    async_.py
    builder.py
    defaults.py
  runtime/
    effect.py       # EffectRuntime, Recorder, Replay
    executor.py     # 有界并发
  observe/
    events.py
    tracer.py
    snapshot.py
  errors.py
  harness/          # 后续：预算强制、权限、评测
```

依赖：v1 **零第三方强依赖**；OTel / Pydantic 等为可选 extra。

---

## 11. 生产检查清单（DoD）

- [ ] 任意步骤可超时；全局 deadline 生效  
- [ ] Effect 非幂等默认不自动重试  
- [ ] `max_iter` + budget 双限制 Loop  
- [ ] `query()` 与事件回调在并发下安全  
- [ ] 取消后不再执行新步骤；结果 `cancelled`  
- [ ] State 校验失败 = Permanent，不重试  
- [ ] parallel 合并 key 冲突可配置且默认失败  
- [ ] 录制回放可通过「假 Effect」回归测试  
- [ ] 类型通过 pyright / mypy strict（目标）  
- [ ] 无隐式全局可变单例承载业务 State  

---

## 12. 分阶段落地

| 阶段 | 交付 |
|------|------|
| **P0** | `State` / `Context` / `StepSpec` / `seq` / SyncPipeline.run / 事件回调 / 基础超时 |
| **P1** | AsyncPipeline、`parallel`+merge、Retry、Budget、query 快照 |
| **P2** | `loop` / `branch`、EffectRuntime 录制回放、熔断 |
| **P3** | Harness 权限与评测夹具、OTel 适配、Builder 糖 |

---

## 13. 关键设计决策（请确认）

1. **State 默认 `frozen` dataclass + `data` Mapping**，领域方可再包一层 TypedDict——是否接受？  
2. **Sync parallel 默认关闭**，需显式 `max_parallel`——是否接受？  
3. **v1 零强依赖**，重试/熔断自研小实现——是否接受（相对依赖 tenacity 等）？  
4. **错误写入 State 的 CONTINUE 模式**是否开放给业务，还是仅 Harness 评测使用？  

确认后按 P0 → P1 在 `agentic-fn` 包内落代码骨架。
