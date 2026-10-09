"""Demo: sequential capability steps on a SyncPipeline."""

from __future__ import annotations

from agentic_fn import Context, State, SyncPipeline, capability

from cli.printing import banner, note, print_result


def run() -> None:
    """
    Show the simplest pipeline shape: capability -> capability.

    Capabilities are pure(ish) State transformers. They should avoid I/O;
    side effects belong in ``@effect`` steps (see the retry / replay demos).
    """

    banner("demo: seq - sequential capabilities")
    note("Start with value=3, then add_one -> double  =>  expected value=8")

    # ---------------------------------------------------------------------------
    # Step functions
    # ---------------------------------------------------------------------------
    # Each capability receives the current State plus a Context (control plane:
    # cancel / budget / clock / effects...). Business data lives only in State.
    @capability
    def add_one(state: State, ctx: Context) -> State:
        """Increment ``value`` by one and return a *new* State."""

        _ = ctx  # unused here; keep the signature consistent with the contract
        return state.update(value=int(state.get("value", 0)) + 1)

    @capability
    def double(state: State, ctx: Context) -> State:
        """Double ``value``."""

        _ = ctx
        return state.update(value=int(state.get("value", 0)) * 2)

    # ---------------------------------------------------------------------------
    # Pipeline assembly (fluent builder)
    # ---------------------------------------------------------------------------
    pipe = (
        SyncPipeline.builder("demo_seq")
        .capability("add_one", add_one)
        .capability("double", double)
        .build()
    )

    # State.empty() allocates run metadata; .update(...) sets business fields.
    result = pipe.run(State.empty().update(value=3))
    print_result(result)
