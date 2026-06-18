# Geist Technical Overview

This document describes the current engineering shape of Geist after the
`fractal_lab` extraction. It is intentionally technical and separate from the
project README.

## Status

Geist is currently a `fractal_lab`-derived agent runtime base. The repository no
longer treats the older root-level selfcall experiment as the package entry. The
active package lives under `src/geist`.

Current package layers:

```text
src/geist/
  agent.py
  cli.py
  context.py
  provider.py
  session.py
  trust.py
  core/
    agent/
      decision_parser.py
      tool_scheduler.py
      tool_spec.py
    fractal/
      protocol.py
      runtime.py
  local/
    artifact_store.py
    dispatcher.py
    generated_tools.py
    tool_specs.py
    trace_store.py
    workspace.py
```

## Core Runtime

`geist.core.fractal.runtime` contains the runtime-native fractal API-call
scheduler extracted from `fractal_lab`.

Important exported types:

- `FractalRuntime`
- `FractalCall`
- `FractalCompleted`
- `FractalLimits`
- `FractalRun`

The runtime schedules one or more model API calls from explicit packets. A call
may:

- complete with `final_response`
- request local tools through `tool_calls`
- emit later API calls through `fractals`
- pass selected handoff material through `continuation_context`

The runtime does not clone hidden conversation history and does not define
subagent identities. Continuation material and observations are explicit inputs.

## Model-Facing Protocol

`geist.core.fractal.protocol.NATIVE_FRACTAL_PROTOCOL` defines the model-facing
runtime JSON contract.

The model returns exactly one fenced `runtime` JSON object in native fractal
calls:

```runtime
{
  "final_response": "",
  "continuation_context": "",
  "clear_continuation_context": false,
  "tool_calls": [{"tool": "exact.tool.name", "arguments": {}}],
  "fractals": [{"instruction": "", "continuation_context": ""}]
}
```

This protocol is intentionally small. It describes movement shape, not a fixed
thought process or ontology.

## Agent Tool Substrate

`geist.core.agent` provides generic model/tool helpers:

- `ToolSpec`
  Tool metadata: name, description, argument hints, side-effect profile,
  read/write sets, and serial execution requirements.
- `ToolScheduler`
  Batches independent tool calls and serializes calls that conflict by declared
  read/write sets.
- `DecisionParser`
  Parses fenced `runtime` JSON blocks and normalizes tool calls.

These helpers are domain-neutral and have no `nous_engine` dependency.

## Agent Wiring

`geist.agent.GeistAgent` wires the runtime into a local coding agent:

- loads provider configuration
- opens a local dispatcher
- loads project context
- opens or continues a session
- sends a `FractalCall` through `FractalRuntime`
- stores user/assistant/trace events in the session JSONL
- materializes large runtime projection bodies into `LocalArtifactStore`

`AgentResult` is the high-level return object for one user turn.

## Provider Configuration

`geist.provider` contains an OpenAI-compatible `/chat/completions` adapter.

Configuration sources:

- environment variables:
  - `GEIST_API_KEY`
  - `GEIST_BASE_URL`
  - `GEIST_MODEL`
- fallback OpenAI-style variables:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `OPENAI_MODEL`
- saved config:
  - `~/.geist/agent/auth.json`

The CLI can save provider config:

```powershell
geist login --api-key <key> --base-url <url> --model <model>
```

## Sessions

`geist.session.SessionStore` stores JSONL sessions under:

```text
~/.geist/agent/sessions/<workspace-hash>/<session-id>.jsonl
```

One turn stores at least:

- `user`
- `trace`
- `assistant`

`geist -c` continues the latest session for the current workspace.

## Project Context And Trust

`geist.context` loads context documents:

- global `~/.geist/agent/AGENTS.md`
- `AGENTS.md` files on the path to the current workspace
- project-local `.geist/SYSTEM.md` and `.geist/APPEND_SYSTEM.md` only when
  the workspace is trusted

`geist.trust.TrustStore` stores trusted workspace roots under:

```text
~/.geist/agent/trusted_projects.json
```

CLI:

```powershell
geist trust
```

## Local Substrate

`geist.local` is the default local coding substrate.

### LocalWorkspace

`LocalWorkspace` is scoped to one workspace root. It provides:

- file read/write/edit/list
- bounded command execution without shell chaining
- read-only git instruments:
  - `git.status`
  - `git.diff_summary`
  - `git.diff_read`
  - `git.snapshot`
  - `git.delta`

Paths are resolved under the workspace root and `.git` targets are rejected.

### LocalArtifactStore

`LocalArtifactStore` persists large text materials by ref. This lets runtime
context carry small previews and durable addresses instead of full raw bodies.

### LocalTraceStore

`LocalTraceStore` is an append-only trace object layer. It stores readable trace
objects and can filter by event, run id, call id, branch path, tool, path, text,
or source.

Trace is treated as addressable runtime memory for one agent movement, not as
hidden conversation history.

### LocalToolRegistry

`LocalToolRegistry` stores generated `local.*` Python handlers in a JSON-backed
registry. A generated tool defines:

```python
async def execute(arguments, state, workspace, tool_api=None):
    ...
```

`LocalToolApi` lets a generated local tool call other registered dispatcher
tools while blocking direct recursive loops.

### LocalToolDispatcher

`LocalToolDispatcher` wires the default local tool surface into a dispatcher
compatible with `FractalRuntime`.

Default tool names:

- `read`
- `write`
- `edit`
- `ls`
- `bash`
- `git.status`
- `git.diff_summary`
- `git.diff_read`
- `git.snapshot`
- `git.delta`
- `artifact.read`
- `artifact.list`
- `artifact.search`
- `trace.read`
- `trace.write`
- `tool.list_local`
- `tool.scaffold`
- generated `local.*` tools

## Minimal Runtime Wiring

A minimal runtime can be assembled with:

```python
from geist.core.fractal import FractalCall, FractalRuntime
from geist.local import LocalToolDispatcher

dispatcher = LocalToolDispatcher("path/to/workspace")
runtime = FractalRuntime(llm, dispatcher)

run = await runtime.run(
    FractalCall(
        root_task="inspect this project",
        tools=dispatcher.get_tools(),
        workspace_label="project",
    )
)
```

The `llm` object must provide an async `generate(messages, **kwargs)` method.
If streaming is enabled, it may also provide `stream_generate(messages,
**kwargs)`.

## CLI

Registered console script:

```powershell
geist
```

Primary modes:

```powershell
geist                         # interactive mode
geist -p "inspect this repo"   # one-shot text output
geist --json "inspect repo"    # one-shot JSON output
geist -c                       # continue latest session
geist --session <id>           # use a specific session id
geist --no-session "task"      # do not persist session events
```

Management commands:

```powershell
geist login --api-key <key> --base-url <url> --model <model>
geist doctor
geist trust
geist sessions
geist sessions --json
```

Interactive commands:

- `/tools`
- `/session`
- `/model`
- `/compact`
- `/trust`
- `/exit`

## Distribution

The Python distribution is named `geist-agent`, while the import package and CLI
command remain `geist`.

Supported local entry points:

```powershell
python -m pip install -e ".[dev]"
geist --help
python -m geist --help
```

The repository also includes an npm/pnpm launcher at `npm/bin/geist.js`. It
locates Python 3.10+, adds the bundled `src` directory to `PYTHONPATH`, and runs
`python -m geist.cli`. This keeps npm as a thin distribution shell rather than a
second implementation of the agent.

The npm package exposes both `geist` and `geist-agent` bin names. Provider setup
can be done with either full arguments or an interactive `geist login`, and
`geist doctor` checks whether the local install and provider config are usable.

More detail lives in `docs/install_distribution.md`.

## Repository Layout

The package uses a standard `src` layout:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["geist*"]
```

Tests live under `tests/` and use:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

## Verification

Current verification command:

```powershell
python -m pytest -q
```

Current test coverage checks:

- runtime import and minimal completion
- runtime tool execution through a dispatcher-like object
- high-level `GeistAgent` one-turn execution
- provider auth config loading and CLI login
- project context trust gating
- artifact put/read/search
- trace append/read/filter
- generated local tool registration and execution
- workspace file operations
- bounded command execution
- read-only git status/diff/snapshot/delta
- default local dispatcher wiring

## Extraction Constraints

The active direction is to follow `fractal_lab`, not to preserve the old Geist
experiment as a compatibility target.

Current constraints:

- no `nous_engine` imports in `src/geist`
- no product-domain coupling
- no VinEnd coupling
- no hidden conversation-history cloning
- no scoring mechanism
- no fixed cognitive phase ontology
- continuation material may be natural language, structured data, addresses, or
  any mixture useful to the next call
- large materials should move by durable refs or explicit observations

## Next Extraction Targets

Likely next layers from `fractal_lab`:

- CLI shell and terminal UI
- provider adapters
- AST/code search and project-map tools
- process manager
- sandbox/change-package manager
- preview/server manager
- HTTP and secret-handle tools
- scope memory as automatic read/write substrate
- skills, knowledge, and MCP as optional on-demand layers
