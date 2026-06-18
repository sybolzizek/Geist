# fractal_lab extraction

This document tracks the first extraction passes from `fractal_lab` into the
open `geist` base.

## Extracted now

- `geist.core.fractal.protocol`
  Model-facing native fractal packet protocol.
- `geist.core.fractal.runtime`
  API-call scheduler with explicit packets, tool observations, frontier
  expansion, trace projection, loop guards, verification guards, and final
  response synthesis.
- `geist.core.agent.tool_spec`
  Generic tool metadata used by the scheduler and model-facing manifests.
- `geist.core.agent.tool_scheduler`
  Dependency-aware local tool batching.
- `geist.core.agent.decision_parser`
  Runtime fenced-JSON parser for model output.
- `geist.local.artifact_store`
  File-backed store for large text materials addressed by refs.
- `geist.local.trace_store`
  Append-only trace object layer for runtime motion history.
- `geist.local.generated_tools`
  JSON-manifest backed registry for generated `local.*` Python tools.
- `geist.local.workspace`
  Workspace-root scoped file operations, bounded command execution, and
  read-only git instruments.
- `geist.local.dispatcher`
  Default dispatcher for `read`, `write`, `edit`, `ls`, `bash`,
  `git.*`, `artifact.*`, `trace.*`, `tool.scaffold`, and generated
  `local.*` tools.

## Not extracted yet

- Higher-level coding tool groups such as AST search, project maps, process
  management, sandboxing, previews, HTTP, secrets, skills, knowledge, and MCP.
- Workspace state and session layout.
- CLI UI and SSE/event rendering.
- Provider adapters.
- Scope memory and per-turn trace integration.
- Skills, knowledge, MCP, and their on-demand loading surfaces.

## Extraction constraints

- Keep the kernel portable: no product-domain imports, no VinEnd coupling, no
  `nous_engine` dependencies.
- Keep the runtime protocol explicit and addressable. Large materials should
  move as references or observations, not as hidden conversation history.
- Preserve freedom of form in continuation material. The runtime can carry
  structured data, natural language, refs, or any useful mixture; it should not
  impose metaphysical categories.
- Prefer hard factual guards over workflow prescriptions. Verification guards
  should report concrete pending evidence, not force a fixed ReAct ceremony.
