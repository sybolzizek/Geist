"""FractalLoop — adapter layer for the fractal self-invocation flow.

Reads a fractal's ``own`` → applies ``context_updates`` to the store →
returns blueprints for the next fractal generation.  Each blueprint
tells the adapter what context to fetch from the store (refs) and what
message to pass to the next fractal (prompt).

The loop does NOT hold or call the engine.  Driving the fractal loop
is the caller's responsibility::

    loop = FractalLoop(invariant)
    ctx = FractalContext(invariant, growth=root_growth)

    while True:
        ctx = await engine.execute(ctx, ...)   # caller drives
        blueprints = loop.process(ctx)          # loop reads + writes store
        if not blueprints:
            break
        for bp in blueprints:
            next_ctx = loop.build(bp)           # loop builds next context
            # engine.execute(next_ctx, ...)     # caller decides order
"""

from __future__ import annotations

from typing import Any

from geist.context import FractalContext
from geist.store import SharedContext


class FractalLoop:
    """Adapter between fractal outputs and the next fractal's inputs.

    ``invariant``
        The immutable self-anchor (shared by all fractals).  The loop
        reads ``_contract`` from it to know the blueprint / context_updates
        field names.

    ``store``
        Optional ``SharedContext``.  A fresh one is created if omitted.
    """

    def __init__(
        self,
        invariant: dict[str, Any],
        store: SharedContext | None = None,
    ) -> None:
        self._store = store if store is not None else SharedContext()
        self._invariant = invariant
        self._contract: dict[str, Any] = invariant.get("_contract", {})

    # -- public accessors --------------------------------------------------

    @property
    def shared(self) -> SharedContext:
        """The shared context store for this fractal flow."""
        return self._store

    # -- contract helpers --------------------------------------------------

    def _key_blueprints(self) -> str:
        return self._contract.get("blueprints_key", "blueprints")

    def _key_updates(self) -> str:
        return self._contract.get("context_updates_key", "context_updates")

    def _field_prompt(self) -> str:
        return self._contract.get("blueprint", {}).get("prompt_field", "prompt")

    def _field_refs(self) -> str:
        return self._contract.get("blueprint", {}).get("refs_field", "refs")

    # -- extract blueprints + updates from fractal output -----------------

    def _extract(self, own: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
        """Extract blueprints and context_updates from a fractal's ``own``.

        Prefers ``tool_results[0]`` (structured LLM output).
        Falls back to direct keys in ``own``.
        """
        bp_key = self._key_blueprints()
        up_key = self._key_updates()

        tool_results = own.get("tool_results", [])
        if tool_results and isinstance(tool_results[0], dict):
            payload = tool_results[0]
            if bp_key in payload or up_key in payload:
                return payload.get(bp_key, []), payload.get(up_key, {})

        # fallback: direct own keys (for test / direct injection adapters)
        return own.get(bp_key, []), own.get(up_key, {})

    # -- process fractal output -------------------------------------------

    def process(self, ctx: FractalContext) -> list[dict]:
        """Read a completed fractal's ``own``, apply side-effects, return blueprints.

        Side-effects:
            ``context_updates`` from the fractal's output are written to
            the shared store, making them available for subsequent fractals
            via ``refs``.

        Returns:
            A list of blueprints.  Each blueprint is a dict with at least
            ``prompt`` (the message for the next fractal) and ``refs``
            (keys to resolve from the shared store).  An empty list means
            this fractal chose not to continue — no more generations.
        """
        blueprints, updates = self._extract(ctx.own)
        if updates:
            self._store.apply(updates)
        return blueprints

    # -- build next fractal context from blueprint ------------------------

    def build(self, blueprint: dict[str, Any]) -> FractalContext:
        """Construct a ``FractalContext`` from a *blueprint* dict.

        The blueprint carries:
          * the *prompt* — a direct message from the previous fractal
            (injected into ``growth["prompt"]``).
          * the *refs* — keys to resolve from the shared store.
            Resolved key-value pairs are merged directly into ``growth``.

        The full blueprint is also injected as ``growth["_blueprint"]``
        so the next fractal can see what it was asked to do.
        """
        prompt_key = self._field_prompt()
        refs_key = self._field_refs()

        growth: dict[str, Any] = {}

        # direct message from previous fractal
        prompt = blueprint.get(prompt_key)
        if prompt:
            growth["prompt"] = prompt

        # auto-fetched context from shared store
        resolved = self._store.resolve(blueprint.get(refs_key, []))
        growth.update(resolved)

        # let the fractal see the blueprint that produced it
        growth["_blueprint"] = dict(blueprint)

        return FractalContext(
            invariant=self._invariant,
            growth=growth,
        )
