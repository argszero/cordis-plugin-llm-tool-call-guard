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
import { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import type { FinishReason, LlmFailure, StreamChunk } from '@deepseek-ai/dsh-llm';
/** Cordis plugin name used by loader diagnostics. */
export declare const name = "llm-tool-call-guard";
/** The LLM service this plugin wraps (`llm/stream`). */
export declare const inject: string[];
/** Plugin configuration. */
export interface Config {
    /**
     * Maximum accumulated `argumentsDelta` bytes allowed for one tool call
     * (per block index). Default 24 KiB (24576). Set to `0` to disable the guard
     * entirely (pass-through). A call whose arguments exceed this budget is cut
     * and the stream terminated with `TOOL_CALL_ARGUMENTS_TOO_LARGE`.
     */
    maxArgsBytes?: number;
    /**
     * When `true` (default), on breach emit a terminal `error` finish that the
     * agent loop routes to `agent/request-error` (never executes the call).
     * Set `false` to observe-only: cut the source but forward a normal `stop`
     * finish (the assembled partial call is then treated as a normal tool call).
     */
    fail?: boolean;
}
/** Resolved config: every field carries its validated default. */
export type ResolvedConfig = Required<Config>;
export declare const Config: z<Config>;
/** Default per-call argument budget in bytes. */
export declare const DEFAULT_MAX_ARGS_BYTES = 24576;
/** Stable machine-routing code for the breach. */
export declare const BREACH_CODE = "TOOL_CALL_ARGUMENTS_TOO_LARGE";
/**
 * The Failure emitted at the breach. Exported so tests (and the core author's
 * draft) can assert on the exact stable shape.
 */
export declare function breachFailure(index: number, bytes: number, limit: number): LlmFailure;
/** The terminal finish emitted at the breach. */
export declare function breachFinish(index: number, bytes: number, limit: number): FinishReason;
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
export declare function guardStream(source: AsyncIterable<StreamChunk>, config: ResolvedConfig): AsyncIterable<StreamChunk>;
/**
 * Register the guard. {@link guardStream} is a plain async generator, so the
 * registration is a direct delegate with no casts.
 */
export declare function apply(ctx: Context, config: ResolvedConfig): void;
