# @argszero/cordis-plugin-llm-tool-call-guard

Runaway tool-call argument guard for the **dsh** harness (deepseek-harness). It wraps
the [`llm/stream`](https://deepseek-ai.github.io/deepseek-harness/) waterfall and cuts a
single tool call that streams **more than a configurable number of arguments bytes**,
terminating the stream with a routed `TOOL_CALL_ARGUMENTS_TOO_LARGE` error finish **so the
oversized call is never executed**.

## Why

A model can start a tool call and stream its JSON arguments until it exhausts the entire
response output-token budget. The assembled tool-call `arguments` grows unboundedly, the
provider returns a terminal `max-tokens` finish, and the user only sees the generic
"Output token limit reached" — the model's reasoning is invisible and the oversized (often
malformed) call is either executed or silently discarded. This is discussion **#6059**.

## What it does

- Counts the accumulated `argumentsDelta` bytes **per tool-call block index** (a model can
  interleave several calls, so the budget applies to each call independently).
- When one call's arguments exceed `maxArgsBytes` (default **24576** = 24 KiB), it stops
  consuming the upstream generator (releasing the provider connection and the still-growing
  arguments stream) and emits a **synthesized terminal `error` finish** with the stable
  failure code `TOOL_CALL_ARGUMENTS_TOO_LARGE`.
- The agent loop branches on the finish reason **before** it filters assistant content for
  tool-call blocks, so an `error` finish routes to `agent/request-error` and **never executes**
  the oversized call. `llm-invariant` explicitly allows an error/aborted finish to carry open
  block indexes, so the finish is protocol-legal.
- Every chunk under the budget passes through **byte-for-byte**; only the breaching call is cut.

## Install / mount

```sh
npm install @argszero/cordis-plugin-llm-tool-call-guard
```

Then mount it in a dsh profile (the bundle patch exposes the plugin id `llm-tool-call-guard`):

```yaml
# cordis.patch.yml (or an overlay)
- insert:
    - id: llm-tool-call-guard
      name: '@argszero/cordis-plugin-llm-tool-call-guard'
```

Tune via config:

```yaml
- set:
    - id: llm-tool-call-guard
      config:
        maxArgsBytes: 8192   # per-call argument budget (byte); 0 disables
        fail: true            # emit error finish (never execute the oversized call)
```

## Config

| Field | Default | Description |
|-------|---------|-------------|
| `maxArgsBytes` | `24576` | Max accumulated `argumentsDelta` bytes per tool-call block index. `0` disables the guard (pure pass-through). |
| `fail` | `true` | On breach, emit a terminal `error` finish (routes to `agent/request-error`, call not executed). `false` = observe-only: cut the source but emit a normal `stop` finish (partial call treated normally). |

## License

MIT
