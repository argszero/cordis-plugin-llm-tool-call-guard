/**
 * Runaway tool-call argument guard for the dsh harness.
 *
 * A model can start a tool call and then stream its JSON arguments until it
 * exhausts the entire response output-token budget (or a large fraction of it).
 * The assembled tool-call `arguments` string grows unboundedly, the provider
 * returns a terminal `max-tokens` finish, and the surfaced message is only the
 * generic "Output token limit reached" — the model's reasoning for *why* it
 * needed so many arguments is invisible, and the oversized (often malformed)
 * call is either executed or silently discarded. This is discussion #6059.
 *
 * Rather than let the call run to budget exhaustion, this plugin wraps the
 * `llm/stream` waterfall (the same around-dispatch seam the in-tree
 * `llm-invariant` uses) and counts the accumulated `argumentsDelta` bytes per
 * tool-call block **index**. When a single call's arguments exceed a configurable
 * budget (`maxArgsBytes`, default 24 KiB), the guard:
 *
 *   1. stops consuming the upstream generator (releasing the provider connection
 *      and the still-growing arguments stream), and
 *   2. emits a synthesized terminal `error` finish with a stable
 *      `TOOL_CALL_ARGUMENTS_TOO_LARGE` failure.
 *
 * An optional whole-request budget (`maxTotalArgsBytes`, default 0 = off) bounds
 * the *cumulative* arguments across every tool-call block in one stream. This is
 * a distinct failure shape from the per-call guard: a model can legitimately
 * keep every call under `maxArgsBytes` yet issue many of them until the sum
 * consumes the entire response output budget. When the total exceeds
 * `maxTotalArgsBytes`, the guard cuts the source and emits a terminal
 * `TOOL_CALL_ARGUMENTS_TOTAL_TOO_LARGE` failure instead of letting the stream run
 * to `max-tokens`.
 *
 * Boundary honesty — what this genuinely does (verified against
 * `packages/core/agent-loop/src/agent.ts` on dsh 0.1.5-alpha.1):
 *  - The agent loop branches on the finish reason *before* it filters the
 *    assistant content for tool-call blocks. When `finish.kind === 'error'` or
 *    `'aborted'`, it settles the attempt, dispatches `agent/request-error`
 *    (retry / throw), and **never executes** the assembled tool-call. Emitting
 *    an `error` finish therefore genuinely prevents a runaway / oversized call
 *    from being executed, not merely from being logged.
 *  - `llm-invariant` (packages/llm/llm/src/invariant.ts) explicitly allows an
 *    `error`/`aborted` finish to carry open block indexes, so the synthesized
 *    finish is protocol-legal.
 *  - The plugin never rewrites or drops chunks before the breach — it forwards
 *    every chunk verbatim until the per-call argument budget is exceeded, then
 *    replaces the rest of the stream. A call that stays under budget passes
 *    through byte-for-byte identical to the upstream stream.
 *
 * The "do not execute this oversized call" behavior is provided by the core
 * `error`-finish path; this plugin supplies the *detection* and the *cut*. If the
 * harness later needs to convert the failure into a structured assistant message
 * (rather than a thrown/retried `LlmError`) that is a core change, out of scope
 * here. `maxTokens` is not a substitute for the guard: a small `maxTokens` can
 * still be consumed entirely by one runaway call and gives no *per-call* signal.
 *
 * @module @argszero/cordis-plugin-llm-tool-call-guard
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
// Type-only: loads the @deepseek-ai/dsh-llm declaration merging that adds the
// `llm/stream` event to Cordis' Context.Events, and provides StreamChunk /
// LlmFailure / FinishReason / GenerateOptions types.
import type { FinishReason, GenerateOptions, LlmFailure, StreamChunk } from '@deepseek-ai/dsh-llm'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'llm-tool-call-guard'

/** The LLM service this plugin wraps (`llm/stream`). */
export const inject = ['llm']

/** Plugin configuration. */
export interface Config {
  /**
   * Maximum accumulated `argumentsDelta` bytes allowed for one tool call
   * (per block index). Default 24 KiB (24576). Set to `0` to disable the guard
   * entirely (pass-through). A call whose arguments exceed this budget is cut
   * and the stream terminated with `TOOL_CALL_ARGUMENTS_TOO_LARGE`.
   */
  maxArgsBytes?: number
  /**
   * Maximum *cumulative* `argumentsDelta` bytes across all tool-call blocks in
   * one stream (whole-request budget). Default `0` (off): only the per-call
   * guard applies. When the sum of every call's arguments exceeds this budget,
   * the stream is cut and terminated with `TOOL_CALL_ARGUMENTS_TOTAL_TOO_LARGE`.
   */
  maxTotalArgsBytes?: number
  /**
   * When `true` (default), on breach emit a terminal `error` finish that the
   * agent loop routes to `agent/request-error` (never executes the call).
   * Set `false` to observe-only: cut the source but forward a normal `stop`
   * finish (the assembled partial call is then treated as a normal tool call).
   */
  fail?: boolean
}

/** Resolved config: every field carries its validated default. */
export type ResolvedConfig = Required<Config>

export const Config: z<Config> = z.object({
  maxArgsBytes: z.number().min(0).default(24576),
  maxTotalArgsBytes: z.number().min(0).default(0),
  fail: z.boolean().default(true),
})

/** Default per-call argument budget in bytes. */
export const DEFAULT_MAX_ARGS_BYTES = 24576

/** Default whole-request aggregate budget in bytes (`0` = off). */
export const DEFAULT_MAX_TOTAL_ARGS_BYTES = 0

/** Stable machine-routing code for the per-call breach. */
export const BREACH_CODE = 'TOOL_CALL_ARGUMENTS_TOO_LARGE'

/** Stable machine-routing code for the whole-request aggregate breach. */
export const BREACH_TOTAL_CODE = 'TOOL_CALL_ARGUMENTS_TOTAL_TOO_LARGE'

/**
 * The Failure emitted at the per-call breach. Exported so tests (and the core
 * author's draft) can assert on the exact stable shape.
 */
export function breachFailure(
  index: number,
  bytes: number,
  limit: number,
): LlmFailure {
  return {
    message:
      `tool-call at index ${index} streamed ${bytes} bytes of arguments, ` +
      `exceeding the ${limit}-byte guard`,
    code: BREACH_CODE,
  }
}

/**
 * The Failure emitted at the whole-request aggregate breach. Distinct code so a
 * consumer can tell "one call was too big" from "the sum of many calls was too
 * big" — the fixes differ (raise a per-call cap vs. reduce how many calls the
 * model issues).
 */
export function breachTotalFailure(totalBytes: number, limit: number): LlmFailure {
  return {
    message:
      `tool-call arguments summed to ${totalBytes} bytes across all calls, ` +
      `exceeding the ${limit}-byte total guard`,
    code: BREACH_TOTAL_CODE,
  }
}

/** The terminal finish emitted at the per-call breach. */
export function breachFinish(index: number, bytes: number, limit: number): FinishReason {
  return { kind: 'error', failure: breachFailure(index, bytes, limit) }
}

/** The terminal finish emitted at the whole-request aggregate breach. */
export function breachTotalFinish(totalBytes: number, limit: number): FinishReason {
  return { kind: 'error', failure: breachTotalFailure(totalBytes, limit) }
}

/**
 * The guard. Returns the upstream stream, or a cut stream that emits up to and
 * including the breaching chunk then a terminal `error` finish.
 *
 * Counting is per `index`: `stream[Symbol.asyncIterator]()` may interleave
 * deltas across multiple tool-call blocks (a model can start several calls and
 * stream their arguments concurrently), so the budget applies to *each* call
 * independently, not to the whole stream. The guard never counts text or
 * reasoning deltas — only tool-call arguments — since those belong to the model
 * response the user asked for.
 */
export async function* guardStream(
  source: AsyncIterable<StreamChunk>,
  config: ResolvedConfig,
): AsyncIterable<StreamChunk> {
  if (config.maxArgsBytes <= 0 && config.maxTotalArgsBytes <= 0) {
    // Guard disabled: pure pass-through.
    for await (const chunk of source) yield chunk
    return
  }

  const bytesByIndex = new Map<number, number>()
  let totalBytes = 0
  for await (const chunk of source) {
    if (chunk.type === 'tool-call-delta') {
      // Track this call's running argument byte count. `index` is the block
      // index, `argumentsDelta` is a string (possibly empty, e.g. a blank name
      // or a re-emitted id with `''` args — assistant-stream keeps those on the
      // `arguments === ''` branch too, so count them consistently).
      const prior = bytesByIndex.get(chunk.index) ?? 0
      const next = prior + chunk.argumentsDelta.length

      // Per-call budget: cut when this single call exceeds its cap.
      if (config.maxArgsBytes > 0 && next > config.maxArgsBytes) {
        yield chunk // emit the chunk that pushed us over, so the cut is visible
        yield { type: 'finish', reason: config.fail ? breachFinish(chunk.index, next, config.maxArgsBytes) : { kind: 'stop' } }
        return
      }

      // Whole-request aggregate budget (off when 0): cut when the sum of every
      // call's arguments exceeds the total cap. Checked after the per-call test
      // so a call that is itself oversized reports the per-call code first.
      totalBytes += chunk.argumentsDelta.length
      if (config.maxTotalArgsBytes > 0 && totalBytes > config.maxTotalArgsBytes) {
        yield chunk
        yield { type: 'finish', reason: config.fail ? breachTotalFinish(totalBytes, config.maxTotalArgsBytes) : { kind: 'stop' } }
        return
      }

      bytesByIndex.set(chunk.index, next)
    }
    yield chunk
  }
}

/**
 * Register the guard. {@link guardStream} is a plain async generator, so the
 * registration is a direct delegate with no casts.
 */
export function apply(ctx: Context, config: ResolvedConfig): void {
  ctx.on('llm/stream', (options: GenerateOptions, next: () => AsyncIterable<StreamChunk>) => {
    void options
    return guardStream(next(), config)
  })
}
