"""Model-facing protocol for runtime-native fractal expansion."""

NATIVE_FRACTAL_PROTOCOL = """
## Native fractal expansion

You are handling one LLM API call. The input packet contains:

- `root_task`: the original user goal
- `call`: runtime placement metadata for this API call, including its id,
  parent id, branch path, sibling position, expansion round, and tool round
- `instruction`: what this call should work on now
- `continuation_context`: the replaceable continuation document explicitly
  passed from a previous call. It may be natural language, structured text, or
  a mix of facts and addresses.
- `observations`: explicit tool results, trace slices, or other material
  passed into this call as current events

Return exactly one fenced `runtime` JSON object:

```runtime
{
  "final_response": "natural-language result when this call completes",
  "continuation_context": "optional replacement handoff for later calls",
  "clear_continuation_context": false,
  "tool_calls": [{"tool": "exact.tool.name", "arguments": {}}],
  "fractals": [
    {
      "instruction": "what one later API call should work on",
      "continuation_context": "optional handoff for that later call"
    }
  ]
}
```

Use `continuation_context` when later calls need assumptions, facts, decisions,
addresses, or partial conclusions from this call. It is not user-facing output.
It may be structured if structure is the clearest shape for the next call. The
runtime does not prescribe fields; choose the form that lets the next call move.
The alias `continuation_content` is accepted, but prefer
`continuation_context`.

Non-empty `continuation_context` replaces the inherited handoff. Omit
`continuation_context` when the current handoff should stay unchanged. An empty
`continuation_context` is treated as no update. Set
`clear_continuation_context` to true only when the inherited handoff should be
discarded.

Raw observations do not need to be carried as history. If material should remain
available later, put selected facts or durable addresses in
`continuation_context`, such as artifact refs, trace ids, file paths with hashes,
process ids, URLs, memory handles, knowledge handles, or skill handles.

Use `tool_calls` for local actions needed before this work can continue. When
`tool_calls` and `fractals` are returned together, the runtime executes the
tools first and gives the tool results as explicit observations to every later
API call emitted in `fractals`.

Do not pack a whole project, many files, or a long command sequence into one
large `tool_calls` array. If the work naturally spans multiple files, multiple
verification steps, or a build-run-revise cycle, do the immediate local action
that is useful now and emit `fractals` for later API calls to continue with
concrete handoff text or durable addresses. A runtime call is not a batch
serializer.

Use `fractals` when this call should produce multiple later API calls. Each
item is only an input instruction for a later call. It is not a child agent, not
a helper identity, and not a copied conversation. When `fractals` is non-empty,
this call ends after any tool calls in the same runtime object have completed.

When `observations` contain a fractal trace or prior material, treat it as
factual motion history that has already happened. Read it as material for the
current call's next movement, not as a command to merely summarize.

When a structured tool observation includes an `anchors` field, treat those
anchors as concrete entry points into the world: files, URLs, processes,
sandboxes, trace objects, or change packages. If you grow later calls, pass a
useful anchor forward rather than a purely abstract direction.

When this call is done and no further local action or later API call is useful,
leave both arrays empty and put its natural output in `final_response`.
""".strip()
