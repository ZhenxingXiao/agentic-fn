# agentic-fn

面向函数的轻量 Agent / 知识库编排框架。把能力拆成可组合的函数，由 Pipeline 统一编排；状态可查询、可回调、可回放——为可追溯与可调试而设计，面向企业生产环境。

> 目标：用更小的核心表面完成 Harness、Loop Engineer 等能力，替代 LangChain 等重型同类框架中「编排层」的角色。

## 设计主张

| 原则 | 含义 |
|------|------|
| **函数优先** | 一切能力是函数；函数之间传递显式状态，而非隐式全局对象 |
| **副作用隔离** | 能力函数（纯逻辑）与副作用函数（LLM / 文件 / 网络 / 日志）分离 |
| **Pipeline 编排** | 同步与异步两种 Pipeline；异步可对接 `asyncio` / 线程池 |
| **可观测默认** | 任意步骤可 `query` 状态，或通过 callback 推送；支持逐步回放 |
| **轻量可替换** | 核心只做编排与契约；模型、向量库、存储以适配器接入 |

## 仓库结构

```text
.
├── agentic-fn/          # 核心库（uv 管理、可独立发布）
├── docs/                # 设计图与架构文档
├── examples/            # 示例与 Demo（规划中）
├── LICENSE
└── README.md
```

| 路径 | 说明 |
|------|------|
| [`agentic-fn/`](./agentic-fn/) | 核心 Python 包 `agentic_fn`，由 [uv](https://github.com/astral-sh/uv) 创建与构建 |
| [`docs/`](./docs/) | 框架设计图（如 `frameworkd.mmd` / `.svg`） |
| `examples/` | 后续放置 Agent、知识库等端到端 Demo |

## 核心概念

### 1. 两类函数

```text
CapabilityFn  : State -> State          # 能力：过滤、决策、变换、重排……
EffectFn      : State -> State          # 副作用：LLM、HTTP、文件 IO、日志、持久化……
                （实现上经 Effect 运行时执行，便于注入、重试、录制）
```

- **能力函数**：可单元测试、可确定性重放；尽量无副作用。
- **副作用函数**：统一走 Effect 边界（超时、重试、配额、审计、录制回放）。

### 2. 两类 Pipeline

| 类型 | 适用 | 典型场景（见设计图） |
|------|------|----------------------|
| **Sync Pipeline** | 低延迟、强顺序 | 应答过滤、检索重排、ReAct 单轮控制流 |
| **Async Pipeline** | IO 密集、可并行 | 文档解析、记忆落库、批量抽取 |

Pipeline 支持：串行、并行扇出/汇合、分支、循环（Loop）、提前终止。状态在步骤间显式传递；运行中可通过 **query** 或 **callback** 观察进度与中间态。

### 3. 状态与可追溯

- 状态为显式、可序列化结构（版本化字段，避免隐式可变共享）。
- 每一步产生可关联的执行记录（step id、输入/输出摘要、耗时、错误）。
- 副作用结果可录制，支持离线 **replay**——生产排障与评测的基础。

### 4. Harness 与 Loop Engineer

框架将提供（而非仅文档描述）两类一等能力：

- **Harness**：权限与沙箱边界、工具白名单、预算（token / 时间 / 调用次数）、评测与回归夹具。
- **Loop Engineer**：把 ReAct、Plan-Execute、Reflection 等控制环固化为可配置的 Pipeline 模式（最大轮次、终止条件、进化/评估步骤）。

## 设计图中的 Pipeline 示例

设计图见 [`docs/frameworkd.mmd`](./docs/frameworkd.mmd)：

| Pipeline | 同步/异步 | 步骤概要 |
|----------|-----------|----------|
| Response generation | Sync | Confidence → Security → Sensitive Word |
| ReAct Agent | Sync | In → 终止判断 → Thought → Action → Evolution → 回环 |
| Memory saving | Async | 特征/关系/实体抽取 → Filter → Summarize → Check → Save |
| Retrieve from DB | Sync | Query Vector / Summary → Rerank → Top-k → Causal Search |
| Parse files | Async | OCR → Parse → Transform → Summarize |

同一套函数与 Pipeline 抽象，覆盖 **Agent 运行** 与 **知识库构建**。

## 与同类框架的定位

| | agentic-fn | 典型重型 Agent 框架 |
|--|------------|---------------------|
| 核心 | 函数 + 状态 + Pipeline | 类层次 / 图 DSL / 大量抽象 |
| 扩展 | Effect 适配器 | 深度绑定的集成生态 |
| 调试 | 步骤级状态、录制回放 | 依赖外部 tracing 或黑盒链 |
| 体积 | 刻意保持小核心 | 依赖与概念面较重 |

**不替代**：具体 LLM SDK、向量库、对象存储。  
**要替代**：编排层的复杂度与不可控副作用边界。

## 快速开始（规划）

核心库使用 uv：

```bash
cd agentic-fn
uv sync
```

安装为可编辑依赖（在 monorepo 或示例中）：

```bash
uv add --editable ./agentic-fn
```

最小心智模型（示意，API 以正式发布为准）：

```python
# Sync: State 经能力函数与副作用函数流转
pipeline = SyncPipeline([confidence, security, sensitive_filter])
result = pipeline.run(state, on_step=callback)

# Async: IO 密集阶段
pipeline = AsyncPipeline([ocr, parse, transform, summarize])
result = await pipeline.run(state)
```

完整 Demo 将放在 `examples/`。

## 生产就绪路线（目标能力）

面向企业落地，核心库将逐步具备：

1. **契约与类型**：严格类型标注（`py.typed`）、Pipeline 步骤签名校验  
2. **可靠性**：步骤级超时 / 重试 / 熔断、取消传播、幂等副作用约定  
3. **安全**：Harness 权限、密钥不入状态明文、审计日志钩子  
4. **可观测**：结构化日志、OpenTelemetry span、关联 ID  
5. **可运维**：配置与密钥外置、优雅降级、资源配额  
6. **可验证**：确定性回放、黄金用例回归、与 examples 联动的集成测试  

## 开发与贡献

- 语言：Python（核心库 `requires-python` 以 `agentic-fn/pyproject.toml` 为准）  
- 包管理 / 构建：uv  
- 许可：[MIT](./LICENSE)  

欢迎围绕「更薄的编排、更清晰的副作用边界、更好的回放与调试」提 Issue 与 PR。

## 路线图（摘要）

- [ ] 核心：`State` / `CapabilityFn` / `EffectFn` / Sync·Async `Pipeline`
- [ ] 观测：query、callback、step trace、录制与 replay
- [ ] Harness：预算、权限、评测夹具
- [ ] Loop Engineer：ReAct 等内置环模式
- [ ] 适配器：LLM、存储、检索（按需、可插拔）
- [ ] `examples/`：Agent Demo、知识库构建 Demo

---

**agentic-fn** — Functions compose. Pipelines orchestrate. State tells the story.
```
