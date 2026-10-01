"""Effect runtime with live execution, recording, and replay support."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from agentic_fn.core.errors import ReplayMissError
from agentic_fn.core.state import State
from agentic_fn.type.enums import ReplayMode
from agentic_fn.type.protocols import Recorder, ReplayStore

if TYPE_CHECKING:
    from agentic_fn.core.context import Context
    from agentic_fn.type.aliases import AsyncEffectFn, EffectFn


@dataclass(slots=True)
class MemoryRecorder:
    """In-memory recorder useful for tests and local evaluation harnesses."""

    storage: MutableMapping[tuple[str, str], State] = field(default_factory=dict)

    def record_effect(self, key: str, input_digest: str, output: State) -> None:
        """Store an effect outcome."""

        self.storage[(key, input_digest)] = output

    def lookup(self, key: str, input_digest: str) -> State | None:
        """Look up a previously recorded outcome."""

        return self.storage.get((key, input_digest))


@dataclass(slots=True)
class EffectRuntime:
    """
    Boundary through which all effectful steps should execute.

    Provides call accounting hooks, optional recording, and strict replay.
    """

    mode: ReplayMode = ReplayMode.LIVE
    recorder: Recorder | None = None
    replay: ReplayStore | None = None

    def run(
        self,
        fn: EffectFn,
        state: State,
        ctx: Context,
        *,
        effect_key: str,
    ) -> State:
        """
        Execute a synchronous effect according to the configured replay mode.

        Parameters
        ----------
        fn:
            Effect callable.
        state:
            Current pipeline state.
        ctx:
            Active run context.
        effect_key:
            Stable identity used for recording/replay (usually step name).
        """

        digest = state.digest()
        if self.mode is ReplayMode.REPLAY:
            return self._lookup_or_raise(effect_key, digest)

        output = fn(state, ctx)
        self._maybe_record(effect_key, digest, output)
        return output

    async def run_async(
        self,
        fn: AsyncEffectFn | EffectFn,
        state: State,
        ctx: Context,
        *,
        effect_key: str,
        is_async: bool,
    ) -> State:
        """
        Execute an effect that may be sync or async under the current mode.

        Sync callables are invoked directly; callers that need thread offload
        should wrap the callable before passing it in.
        """

        digest = state.digest()
        if self.mode is ReplayMode.REPLAY:
            return self._lookup_or_raise(effect_key, digest)

        if is_async:
            output = await cast(Any, fn)(state, ctx)
        else:
            output = cast(Any, fn)(state, ctx)

        self._maybe_record(effect_key, digest, output)
        return output

    def _maybe_record(self, effect_key: str, digest: str, output: State) -> None:
        """Record an effect outcome when a recorder is configured."""

        if self.recorder is None:
            return
        if self.mode in {ReplayMode.RECORD, ReplayMode.LIVE}:
            self.recorder.record_effect(effect_key, digest, output)

    def _lookup_or_raise(self, effect_key: str, digest: str) -> State:
        """Resolve a replay hit or raise ``ReplayMissError``."""

        if self.replay is None:
            raise ReplayMissError(f"replay store missing for effect {effect_key!r}")
        found = self.replay.lookup(effect_key, digest)
        if found is None:
            raise ReplayMissError(
                f"no recording for effect={effect_key!r} digest={digest[:12]}"
            )
        return found
