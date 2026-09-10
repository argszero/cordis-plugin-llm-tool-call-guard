# @argszero/cordis-plugin-llm-tool-call-guard

Runaway tool-call argument guard for the **dsh** harness (deepseek-harness). It wraps
the [`llm/stream`](https://deepseek-ai.github.io/deepseek-harness/) waterfall and cuts a
single tool call that streams **more than a configurable number of arguments bytes** (or
**fragments**), terminating the stream with a routed error finish so the oversized call is
**never executed**.

## Why

A model can start a tool call and stream its JSON arguments until it exhausts the entire
response output-token budget. The assembled tool-call `arguments` grows unboundedly, the
provider returns a terminal `max-tokens` finish, and the user only sees the generic
"Output token limit reached" — the model's reasoning is invisible and the oversized (often
malformed) call is either executed or silently discarded. This is discussion **#6059**.

## What it does

- Counts the accumulated `argumentsDelta` **UTF-8 bytes** per tool-call block index (a model
  can interleave several calls, so the budget applies to each call independently).
- When one call's arguments exceed `maxArgsBytes` (default **24576** = 24 KiB), it stops
  consuming the upstream generator (releasing the provider connection and the still-growing
  arguments stream) and emits a **synthesized terminal `error` finish** with the stable
  failure code `TOOL_CALL_ARGUMENTS_TOO_LARGE`.
- Optionally counts the **number of `tool-call-delta` fragments** per call index
  (`maxArgsFragments`, default `0` = off). Bytes and fragments measure *different* runaway
  shapes: the recorded #6059 stream is 4,074 deltas but only 4,658 bytes — a **fragment-count
  runaway** that a byte budget alone never fires. When an index exceeds `maxArgsFragments` it
  cuts the source and emits the distinct code `TOOL_CALL_ARGUMENTS_TOO_MANY_FRAGMENTS`.
- Optionally counts the **whole-request aggregate** `argumentsDelta` across *every* tool-call
  block in one stream (`maxTotalArgsBytes`, default `0` = off). When the sum exceeds that
  budget it cuts the source and emits the distinct code `TOOL_CALL_ARGUMENTS_TOTAL_TOO_LARGE`
  — a different failure shape from the per-call guard (one call too big vs. too many calls).
- The agent loop branches on the finish reason **before** it filters assistant content for
  tool-call blocks, so an `error` finish routes to `agent/request-error` and **never executes**
  the oversized call. `llm-invariant` explicitly allows an error/aborted finish to carry open
  block indexes, so the finish is protocol-legal.
- **Repairs a tool call that streams an empty identity** (`repair`, default `'repair'`) — the
  `{"id":"","name":""}` shape behind discussion **#6152**. See [Empty tool-call identity](#empty-tool-call-identity-6152) below.
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
        maxArgsBytes: 8192        # per-call argument budget (byte); 0 disables
        maxArgsFragments: 1024    # per-call fragment budget (count); 0 disables
        maxTotalArgsBytes: 32768  # whole-request aggregate budget (byte); 0 disables
        fail: true                 # emit error finish (never execute the oversized call)
```

## Config

| Field | Default | Description |
|-------|---------|-------------|
| `maxArgsBytes` | `24576` | Max accumulated `argumentsDelta` **UTF-8 bytes** per tool-call block index. `0` disables the byte guard. |
| `maxArgsFragments` | `0` | Max number of `tool-call-delta` **fragments** per tool-call block index. `0` disables the fragment guard. Catches the #6059 fragment-count runaway that a byte budget alone misses. |
| `maxTotalArgsBytes` | `0` | Max *cumulative* `argumentsDelta` **UTF-8 bytes** across all tool-call blocks in one stream (whole-request budget). `0` disables the aggregate guard. |
| `fail` | `true` | On breach, emit a terminal `error` finish (routes to `agent/request-error`, call not executed). `false` = observe-only: cut the source but emit a normal `stop` finish (partial call treated normally). |
| `repair` | `'repair'` | What to do with an **empty tool-call identity** (#6152): `'repair'` substitutes a deterministic synthetic call id, `'error'` cuts the stream with `TOOL_CALL_EMPTY_IDENTITY`, `'off'` preserves v0.1.3 exactly. |
| `repairBytes` | `true` | With `repair: 'repair'`, also substitute `{}` for an empty `argumentsDelta` on the repaired call. Only the block's first delta is inspected, so this can only affect a genuinely identity-less block. |

## Empty tool-call identity (#6152)

A model — observed with an OpenAI-completions-style provider — can emit a tool call whose
identity is blank:

```json
{"type": "tool-call", "id": "", "name": "", "arguments": "{}"}
```

The harness persists that as-is, and the **next** load rejects the stored `tool/result`
(`assertMessageEventShape` requires a non-empty `source.callId`). The whole conversation is
then marked corrupt and the session can no longer be started or resumed; recovery means
hand-patching the `.jsonl.zstd` frame container.

The asymmetry is worth stating precisely, because it explains why no existing layer caught
it: the validator that rejects the event **already exists** and is called from the *read*
boundary and the seed path — but `Session.append` never calls it. The OpenAI-completions
adapter emits `''` as a deliberate "id not yet seen" sentinel, and the assembler's
`?? 'call-N'` fallback never fires on it because `''` is not `undefined`.

This plugin sits on the producer-side seam (`llm/stream`), so it can close the gap before the
event is persisted:

- **`repair: 'repair'`** (default) rewrites the empty id to a deterministic
  `repaired-call-<index>` on **both** the `tool-call-delta` chunks and the `block-end` block.
  Repairing only the deltas would be a silent no-op, because the core assembler treats
  `block-end` as authoritative and overwrites the accumulated values.
- An empty **`name`** is deliberately left alone. The harness already turns it into a
  `ToolNotFoundError` / `UNKNOWN_TOOL` error result, which is an honest and *resumable*
  outcome. Blanking the name would make the call vanish; inventing a plausible one would
  fabricate a call the model never requested.
- **`repair: 'error'`** cuts the stream with a terminal `error` finish (code
  `TOOL_CALL_EMPTY_IDENTITY`), so the degenerate call is never executed at all.
- **`repair: 'off'`** preserves v0.1.3 behaviour byte-for-byte.

Every repaired block is reported **once** at `warn` level, naming the field(s) that were
empty. That is deliberate: "nothing was visible" is the heart of the #6152 report.

> **Scope.** This is a mitigation at the producer seam, not the authoritative fix. A plugin
> cannot prevent the append — the write boundary needs its own check. Reported upstream in
> [#6152](https://github.com/deepseek-ai/deepseek-harness/discussions/6152).

## UTF-8 byte counting

The guard counts `argumentsDelta` in **UTF-8 bytes** (via `TextEncoder`), matching the option
name and docs. It does **not** use `.length`, which counts UTF-16 code units and would
under-count non-ASCII arguments (e.g. `'中'` is 3 bytes but 1 code unit).

## Regression fixture

The test suite includes a **boundary-exact recorded stream** from discussion #6059: 4,074
`tool-call-delta` fragments / 4,658 bytes / histogram `{ 1: 3492, 2: 581, 4: 1 }`, supplied by
`@luisnomad`. The fixture asserts the byte-only guard passes it unchanged (the blind spot) while
the fragment guard correctly catches it.

## Peer range

The plugin declares `@deepseek-ai/dsh-llm` as a peer dependency with range:

```
>=0.1.2-rc.1 <0.2.0 || >=0.1.5-alpha.1 <0.2.0
```

Every dsh release published today is a prerelease (`0.1.2-rc.1`, `0.1.5-alpha.1`,
`0.1.5-rc.1`, …), and a semver comparator only admits prereleases that share its own
`major.minor.patch` tuple. Two failure modes follow, and both must be avoided:

```jsonc
// Matches nothing: 0.1.2-rc.1 is LOWER than 0.1.2, and every other
// prerelease has a different tuple.
">=0.1.2"

// Only 0.1.2-rc.1: a 0.1.5-line user gets ERESOLVE.
">=0.1.2-rc.1 <0.2.0"

// What we ship: one comparator per supported tuple line.
">=0.1.2-rc.1 <0.2.0 || >=0.1.5-alpha.1 <0.2.0"
```

The npm `latest` tag for `@deepseek-ai/dsh` is on the `0.1.2` line while `next`/`alpha`
point at `0.1.5`, so both comparators are needed. v0.1.2 shipped the second form and
rejected the `0.1.5` line (`ERESOLVE`); v0.1.3 fixes it.

## License

MIT
