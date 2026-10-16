"""ModelRuntime: router-enforced model invocation control plane."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.model import Content
from agentic_fn.content.protocols import ContentStore
from agentic_fn.content.refs import ContentRef, MemoryContentStore, externalize
from agentic_fn.core.context import Context
from agentic_fn.core.errors import CircuitOpenError, FrameworkError, TimeoutError
from agentic_fn.models.cost import DEFAULT_COST_MODEL
from agentic_fn.models.errors import (
    ModelCapabilityError,
    ModelExhaustedError,
    ModelPermanentError,
    ModelReplayError,
    ModelTransientError,
)
from agentic_fn.models.policy import DEFAULT_MODEL_CALL_POLICY, ModelCallPolicy
from agentic_fn.models.protocols import CostModel, ModelRouter, OutputFilter, SecretProvider
from agentic_fn.models.registry import ClientRegistry
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import (
    DecideResponse,
    ModelResponse,
    RouteAttemptTrace,
    RouteTrace,
    Usage,
)
from agentic_fn.models.router.health import ProviderHealthView
from agentic_fn.models.safety import NoOpOutputFilter
from agentic_fn.observe.events import Event
from agentic_fn.runtime.circuit import CircuitBreakerRegistry
from agentic_fn.type.enums import EventKind, ReplayMode


@dataclass(slots=True)
class ModelRuntime:
    """
    Production entrypoint for all model IO.

    Pipeline code must call ``complete`` / ``decide`` / ``stream`` here so that
    routing, budget, circuit breaking, safety, and replay stay centralized.
    """

    router: ModelRouter
    clients: ClientRegistry
    policy: ModelCallPolicy = field(default_factory=lambda: DEFAULT_MODEL_CALL_POLICY)
    secrets: SecretProvider | None = None
    health: ProviderHealthView | None = None
    breakers: CircuitBreakerRegistry = field(default_factory=CircuitBreakerRegistry)
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COST_MODEL)
    output_filter: OutputFilter = field(default_factory=NoOpOutputFilter)
    raw_store: ContentStore = field(default_factory=MemoryContentStore)
    recorder_key_prefix: str = "model"

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Execute a routed unary completion."""

        ctx.check()
        self._allow_model(ctx, request.model, request.modality.value)
        digest = request.digest(include_route=self.policy.digest_includes_route)
        replayed = self._replay_lookup(ctx, digest, kind="complete")
        if replayed is not None:
            return ModelResponse.from_dict(replayed)

        decision = self.router.resolve(request, ctx)
        self._emit(ctx, "model.route", request.model, route=decision)
        errors: list[BaseException] = []
        attempt_traces: list[RouteAttemptTrace] = []

        for attempt in decision.attempts:
            ctx.check()
            client = self.clients.get(attempt.provider)
            bound = request.bind_route(
                provider=attempt.provider,
                physical_model=attempt.physical_model,
            )
            # Merge route-level default params when request left fields unset.
            bound = self._merge_defaults(bound, decision.entry.defaults)
            breaker = self.breakers.get(attempt.breaker_key)
            health = self._health()
            try:
                breaker.before_call()
            except CircuitOpenError as exc:
                health.set_open(attempt.breaker_key, True)
                attempt_traces.append(
                    RouteAttemptTrace(
                        provider=attempt.provider,
                        physical_model=attempt.physical_model,
                        reason=attempt.reason,
                        error_type=type(exc).__name__,
                    )
                )
                errors.append(exc)
                if not self._should_failover(exc, decision):
                    break
                continue

            started = ctx.clock.monotonic()
            try:
                with_timeout = bound
                if with_timeout.timeout_s is None:
                    with_timeout = replace(with_timeout, timeout_s=self.policy.timeout_s)
                self._emit(
                    ctx,
                    "model.start",
                    request.model,
                    provider=attempt.provider,
                    physical_model=attempt.physical_model,
                )
                raw_resp = client.complete(with_timeout, ctx)
                latency_ms = (ctx.clock.monotonic() - started) * 1000.0
                resp = self._normalize_complete(
                    raw_resp,
                    request=bound,
                    attempt_provider=attempt.provider,
                    attempt_model=attempt.physical_model,
                    decision=decision,
                    attempt_traces=attempt_traces,
                    latency_ms=latency_ms,
                    digest=digest,
                )
                if self.policy.apply_output_filter:
                    self.output_filter.check_response(resp, bound)
                self._account(ctx, resp.usage)
                breaker.on_success()
                health.set_open(attempt.breaker_key, False)
                health.record_success(attempt.breaker_key, latency_ms)
                self._maybe_record(ctx, digest, resp.to_dict(), kind="complete")
                self._emit(ctx, "model.success", request.model, latency_ms=latency_ms)
                return resp
            except Exception as exc:  # noqa: BLE001 - classified below
                latency_ms = (ctx.clock.monotonic() - started) * 1000.0
                breaker.on_failure()
                health.record_failure(attempt.breaker_key)
                if isinstance(exc, CircuitOpenError) or self._is_open_after_failure(breaker):
                    health.set_open(attempt.breaker_key, True)
                attempt_traces.append(
                    RouteAttemptTrace(
                        provider=attempt.provider,
                        physical_model=attempt.physical_model,
                        reason=attempt.reason,
                        error_type=type(exc).__name__,
                        latency_ms=latency_ms,
                    )
                )
                errors.append(exc)
                self._emit(
                    ctx,
                    "model.failover",
                    request.model,
                    error_type=type(exc).__name__,
                    provider=attempt.provider,
                )
                if not self._should_failover(exc, decision):
                    if isinstance(exc, FrameworkError):
                        raise
                    raise ModelPermanentError(str(exc), cause=exc) from exc
                continue

        raise ModelExhaustedError(
            f"all route attempts failed for {request.model!r}: "
            + ", ".join(type(e).__name__ for e in errors)
        )

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Async complete with the same control plane (async client methods)."""

        # Shared resolve + failover loop, calling acomplete on clients.
        ctx.check()
        self._allow_model(ctx, request.model, request.modality.value)
        digest = request.digest(include_route=self.policy.digest_includes_route)
        replayed = self._replay_lookup(ctx, digest, kind="complete")
        if replayed is not None:
            return ModelResponse.from_dict(replayed)

        decision = self.router.resolve(request, ctx)
        errors: list[BaseException] = []
        attempt_traces: list[RouteAttemptTrace] = []
        for attempt in decision.attempts:
            ctx.check()
            client = self.clients.get(attempt.provider)
            bound = self._merge_defaults(
                request.bind_route(
                    provider=attempt.provider,
                    physical_model=attempt.physical_model,
                ),
                decision.entry.defaults,
            )
            if bound.timeout_s is None:
                bound = replace(bound, timeout_s=self.policy.timeout_s)
            breaker = self.breakers.get(attempt.breaker_key)
            health = self._health()
            try:
                breaker.before_call()
            except CircuitOpenError as exc:
                errors.append(exc)
                if not self._should_failover(exc, decision):
                    break
                continue
            started = ctx.clock.monotonic()
            try:
                raw_resp = await client.acomplete(bound, ctx)
                latency_ms = (ctx.clock.monotonic() - started) * 1000.0
                resp = self._normalize_complete(
                    raw_resp,
                    request=bound,
                    attempt_provider=attempt.provider,
                    attempt_model=attempt.physical_model,
                    decision=decision,
                    attempt_traces=attempt_traces,
                    latency_ms=latency_ms,
                    digest=digest,
                )
                if self.policy.apply_output_filter:
                    self.output_filter.check_response(resp, bound)
                self._account(ctx, resp.usage)
                breaker.on_success()
                health.record_success(attempt.breaker_key, latency_ms)
                self._maybe_record(ctx, digest, resp.to_dict(), kind="complete")
                return resp
            except Exception as exc:  # noqa: BLE001
                breaker.on_failure()
                health.record_failure(attempt.breaker_key)
                errors.append(exc)
                if not self._should_failover(exc, decision):
                    if isinstance(exc, FrameworkError):
                        raise
                    raise ModelPermanentError(str(exc), cause=exc) from exc
                continue
        raise ModelExhaustedError(
            f"all async route attempts failed for {request.model!r}"
        )

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Execute a routed decide call."""

        ctx.check()
        self._allow_model(ctx, request.model, "decide")
        digest = request.digest(include_route=self.policy.digest_includes_route)
        replayed = self._replay_lookup(ctx, digest, kind="decide")
        if replayed is not None:
            return DecideResponse.from_dict(replayed)

        model_req = request.as_model_request()
        decision = self.router.resolve(model_req, ctx)
        errors: list[BaseException] = []
        for attempt in decision.attempts:
            ctx.check()
            client = self.clients.get(attempt.provider)
            bound = request.bind_route(
                provider=attempt.provider,
                physical_model=attempt.physical_model,
            )
            breaker = self.breakers.get(attempt.breaker_key)
            health = self._health()
            try:
                breaker.before_call()
            except CircuitOpenError as exc:
                errors.append(exc)
                if not self._should_failover(exc, decision):
                    break
                continue
            started = ctx.clock.monotonic()
            try:
                raw = client.decide(bound, ctx)
                latency_ms = (ctx.clock.monotonic() - started) * 1000.0
                resp = self._normalize_decide(
                    raw,
                    request=bound,
                    attempt_provider=attempt.provider,
                    attempt_model=attempt.physical_model,
                    decision=decision,
                    latency_ms=latency_ms,
                    digest=digest,
                )
                if self.policy.apply_output_filter:
                    self.output_filter.check_decide(resp, bound)
                self._account(ctx, resp.usage)
                breaker.on_success()
                health.record_success(attempt.breaker_key, latency_ms)
                self._maybe_record(ctx, digest, resp.to_dict(), kind="decide")
                return resp
            except ModelCapabilityError as exc:
                errors.append(exc)
                breaker.on_failure()
                health.record_failure(attempt.breaker_key)
                continue
            except Exception as exc:  # noqa: BLE001
                breaker.on_failure()
                health.record_failure(attempt.breaker_key)
                errors.append(exc)
                if not self._should_failover(exc, decision):
                    if isinstance(exc, FrameworkError):
                        raise
                    raise ModelPermanentError(str(exc), cause=exc) from exc
                continue
        raise ModelExhaustedError(
            f"all decide attempts failed for {request.model!r}: "
            + ", ".join(type(e).__name__ for e in errors)
        )

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        """
        Stream from the primary eligible candidate.

        Failover mid-stream is not attempted (partial output already emitted).
        """

        ctx.check()
        request = replace(request, stream=True)
        decision = self.router.resolve(request, ctx)
        if not decision.attempts:
            raise ModelExhaustedError(f"no stream candidates for {request.model!r}")
        attempt = decision.attempts[0]
        client = self.clients.get(attempt.provider)
        bound = self._merge_defaults(
            request.bind_route(
                provider=attempt.provider,
                physical_model=attempt.physical_model,
            ),
            decision.entry.defaults,
        )
        breaker = self.breakers.get(attempt.breaker_key)
        breaker.before_call()
        started = ctx.clock.monotonic()
        try:
            final: ModelResponse | None = None
            for partial in client.stream(bound, ctx):
                final = partial
                yield partial
            latency_ms = (ctx.clock.monotonic() - started) * 1000.0
            breaker.on_success()
            self._health().record_success(attempt.breaker_key, latency_ms)
            if final is not None:
                self._account(ctx, final.usage)
        except Exception:
            breaker.on_failure()
            self._health().record_failure(attempt.breaker_key)
            raise

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _health(self) -> ProviderHealthView:
        """Resolve health view (router-owned or local)."""

        if self.health is not None:
            return self.health
        router_health = getattr(self.router, "health", None)
        if isinstance(router_health, ProviderHealthView):
            return router_health
        # Attach a local view so accounting still works.
        self.health = ProviderHealthView()
        return self.health

    def _allow_model(self, ctx: Context, logical: str, modality: str) -> None:
        """Invoke optional harness allow_model hook."""

        harness = ctx.harness
        if harness is None:
            return
        allow = getattr(harness, "allow_model", None)
        if allow is None:
            return
        if not allow(logical, modality):
            from agentic_fn.core.errors import HarnessDeniedError

            raise HarnessDeniedError(
                f"model not allowed: {logical!r} modality={modality!r}"
            )

    def _should_failover(self, exc: BaseException, decision: Any) -> bool:
        """Return whether the exception is failover-eligible."""

        if isinstance(exc, self.policy.failover_on):
            return True
        failover = getattr(decision, "failover", None)
        if failover is None:
            return False
        name = type(exc).__name__.lower()
        tokens = set(failover.on)
        if "transient" in tokens and isinstance(exc, ModelTransientError):
            return True
        if "timeout" in tokens and isinstance(exc, TimeoutError):
            return True
        if "circuit_open" in tokens and isinstance(exc, CircuitOpenError):
            return True
        return any(token in name for token in tokens)

    def _is_open_after_failure(self, breaker: Any) -> bool:
        """Best-effort check if breaker opened."""

        state = getattr(breaker, "state", None)
        return str(getattr(state, "value", state)) == "open"

    def _merge_defaults(self, request: ModelRequest, defaults: Any) -> ModelRequest:
        """Fill unset sampling params from route defaults."""

        from agentic_fn.models.request import ModelParams

        params = request.params
        merged = ModelParams(
            temperature=params.temperature
            if params.temperature is not None
            else defaults.temperature,
            top_p=params.top_p if params.top_p is not None else defaults.top_p,
            max_tokens=params.max_tokens
            if params.max_tokens is not None
            else defaults.max_tokens,
            stop=params.stop or defaults.stop,
            seed=params.seed if params.seed is not None else defaults.seed,
            presence_penalty=params.presence_penalty
            if params.presence_penalty is not None
            else defaults.presence_penalty,
            frequency_penalty=params.frequency_penalty
            if params.frequency_penalty is not None
            else defaults.frequency_penalty,
            extra={**dict(defaults.extra), **dict(params.extra)},
        )
        return replace(request, params=merged)

    def _normalize_complete(
        self,
        resp: ModelResponse,
        *,
        request: ModelRequest,
        attempt_provider: str,
        attempt_model: str,
        decision: Any,
        attempt_traces: list[RouteAttemptTrace],
        latency_ms: float,
        digest: str,
    ) -> ModelResponse:
        """Fill route/usage metadata and optionally externalize raw."""

        usage = resp.usage
        if usage.cost is None and not usage.unknown:
            estimated = self.cost_model.estimate(
                provider=attempt_provider,
                physical_model=attempt_model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            if estimated is not None:
                usage = replace(usage, cost=estimated)
        attempt_traces.append(
            RouteAttemptTrace(
                provider=attempt_provider,
                physical_model=attempt_model,
                reason="success",
                latency_ms=latency_ms,
            )
        )
        route = RouteTrace(
            logical_model=request.model,
            strategy=str(decision.strategy),
            attempts=tuple(attempt_traces),
            selected_provider=attempt_provider,
            selected_physical_model=attempt_model,
            filtered=tuple(getattr(decision, "filtered", ()) or ()),
        )
        raw_ref = resp.raw_ref
        if self.policy.externalize_raw and raw_ref is None and resp.labels.get("raw_json"):
            raw_ref = self._store_raw(str(resp.labels.get("raw_json")))
        return replace(
            resp,
            logical_model=request.model,
            provider=attempt_provider,
            physical_model=attempt_model,
            usage=usage,
            route=route,
            latency_ms=latency_ms,
            request_digest=digest,
            raw_ref=raw_ref,
        )

    def _normalize_decide(
        self,
        resp: DecideResponse,
        *,
        request: DecideRequest,
        attempt_provider: str,
        attempt_model: str,
        decision: Any,
        latency_ms: float,
        digest: str,
    ) -> DecideResponse:
        """Normalize decide response metadata."""

        usage = resp.usage
        if usage.cost is None and not usage.unknown:
            estimated = self.cost_model.estimate(
                provider=attempt_provider,
                physical_model=attempt_model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            if estimated is not None:
                usage = replace(usage, cost=estimated)
        route = RouteTrace(
            logical_model=request.model,
            strategy=str(decision.strategy),
            selected_provider=attempt_provider,
            selected_physical_model=attempt_model,
            filtered=tuple(getattr(decision, "filtered", ()) or ()),
        )
        return replace(
            resp,
            logical_model=request.model,
            provider=attempt_provider,
            physical_model=attempt_model,
            usage=usage,
            route=route,
            latency_ms=latency_ms,
            request_digest=digest,
        )

    def _store_raw(self, raw_json: str) -> ContentRef | Content:
        """Externalize raw provider JSON."""

        content = Content.of(ContentKind.JSON, raw_json, validate=False)
        return externalize(content, self.raw_store, force=True)

    def _account(self, ctx: Context, usage: Usage) -> None:
        """Consume budget dimensions from usage."""

        calls = 1 if self.policy.consume_budget_calls else 0
        tokens = 0
        if self.policy.consume_budget_tokens and usage.total_tokens:
            tokens = int(usage.total_tokens)
        cost = 0.0
        if self.policy.consume_budget_cost and usage.cost:
            cost = float(usage.cost)
        if calls or tokens or cost:
            ctx.budget.consume(calls=calls, tokens=tokens, cost=cost)
            if ctx.harness is not None:
                view = ctx.budget.view()
                ctx.harness.on_budget(
                    {
                        "remaining_calls": view.remaining_calls,
                        "remaining_tokens": view.remaining_tokens,
                        "remaining_cost": view.remaining_cost,
                    }
                )

    def _replay_lookup(self, ctx: Context, digest: str, *, kind: str) -> dict[str, Any] | None:
        """Return recorded payload when EffectRuntime is in REPLAY."""

        effects = ctx.effects
        if effects.mode is not ReplayMode.REPLAY:
            return None
        if effects.replay is None:
            raise ModelReplayError("REPLAY mode requires a replay store")
        key = f"{self.recorder_key_prefix}:{kind}"
        state = effects.replay.lookup(key, digest)
        if state is None:
            raise ModelReplayError(f"missing replay for {key} digest={digest[:16]}")
        payload = state.get("payload")
        if not isinstance(payload, dict):
            raise ModelReplayError("replay payload missing")
        return payload

    def _maybe_record(
        self,
        ctx: Context,
        digest: str,
        payload: dict[str, Any],
        *,
        kind: str,
    ) -> None:
        """Record response when RECORD/LIVE+recorder configured."""

        if not self.policy.record_responses:
            return
        effects = ctx.effects
        if effects.mode is ReplayMode.REPLAY:
            return
        if effects.recorder is None and effects.mode is not ReplayMode.RECORD:
            return
        if effects.recorder is None:
            return
        from agentic_fn.core.state import State

        key = f"{self.recorder_key_prefix}:{kind}"
        effects.recorder.record_effect(
            key,
            digest,
            State.empty().update(payload=payload),
        )

    def _emit(self, ctx: Context, kind: str, logical: str, **fields: Any) -> None:
        """Best-effort event emission into hooks and optional log sink."""

        mapping = {
            "model.route": EventKind.MODEL_ROUTE,
            "model.start": EventKind.MODEL_START,
            "model.success": EventKind.MODEL_SUCCESS,
            "model.failover": EventKind.MODEL_FAILOVER,
            "model.error": EventKind.MODEL_ERROR,
        }
        event_kind = mapping.get(kind, EventKind.STEP_END)
        try:
            ctx.hooks.emit(
                Event(
                    kind=event_kind,
                    run_id=ctx.run_id,
                    step_name=logical,
                    error_type=fields.get("error_type"),
                    attributes={
                        k: v
                        for k, v in fields.items()
                        if k != "error_type" and isinstance(v, (str, int, float, bool))
                    },
                )
            )
        except Exception:  # noqa: BLE001 - never break model path on hooks
            pass
        try:
            ctx.log.bind(component="model_runtime", event=kind).info(
                kind,
                logical_model=logical,
                **{
                    k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                    for k, v in fields.items()
                },
            )
        except Exception:  # noqa: BLE001
            pass
