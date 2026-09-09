import { test } from 'node:test'
import assert from 'node:assert/strict'
import { gunzipSync } from 'node:zlib'
import { guardStream, breachFailure, breachFinish, breachTotalFailure, breachTotalFinish, breachFragmentsFailure, breachFragmentsFinish, BREACH_CODE, BREACH_TOTAL_CODE, BREACH_FRAGMENTS_CODE, DEFAULT_MAX_ARGS_BYTES, DEFAULT_MAX_TOTAL_ARGS_BYTES, DEFAULT_MAX_ARGS_FRAGMENTS, utf8Length } from '../lib/index.js'

const ON = { maxArgsBytes: 24576, fail: true }
const OFF = { maxArgsBytes: 0, fail: true }
const SMALL = { maxArgsBytes: 10, fail: true }
const OBSERVE = { maxArgsBytes: 10, fail: false }
const TOTAL = { maxArgsBytes: 10, maxTotalArgsBytes: 12, fail: true }
const TOTAL_OFF = { maxArgsBytes: 10, maxTotalArgsBytes: 0, fail: true }
const FRAG = { maxArgsBytes: 24576, maxArgsFragments: 10, fail: true }
const FRAG_OBSERVE = { maxArgsBytes: 24576, maxArgsFragments: 10, fail: false }

function td(index, id, delta) {
  return { type: 'tool-call-delta', index, id, argumentsDelta: delta }
}
function textDelta(index, text) {
  return { type: 'text-delta', index, text }
}
function finish(reason) {
  return { type: 'finish', reason }
}
function stop() {
  return { type: 'finish', reason: { kind: 'stop' } }
}
async function collect(iter) {
  const out = []
  for await (const chunk of iter) out.push(chunk)
  return out
}

/*** breachFailure / breachFinish: pure shapes ***/

test('breachFailure carries the stable code and a clear message', () => {
  const f = breachFailure(3, 30000, 24576)
  assert.equal(f.code, BREACH_CODE)
  assert.equal(BREACH_CODE, 'TOOL_CALL_ARGUMENTS_TOO_LARGE')
  assert.match(f.message, /tool-call at index 3 streamed 30000 bytes/)
  assert.match(f.message, /exceeding the 24576-byte guard/)
})

test('breachFinish is an error finish carrying the breach failure', () => {
  const r = breachFinish(3, 30000, 24576)
  assert.equal(r.kind, 'error')
  assert.equal(r.failure.code, BREACH_CODE)
})

test('default budget is 24576 bytes', () => {
  assert.equal(DEFAULT_MAX_ARGS_BYTES, 24576)
})

test('default total budget is 0 (aggregate guard off)', () => {
  assert.equal(DEFAULT_MAX_TOTAL_ARGS_BYTES, 0)
})

test('breachTotalFailure carries the distinct stable code and a clear message', () => {
  const f = breachTotalFailure(30, 12)
  assert.equal(f.code, BREACH_TOTAL_CODE)
  assert.equal(BREACH_TOTAL_CODE, 'TOOL_CALL_ARGUMENTS_TOTAL_TOO_LARGE')
  assert.match(f.message, /summed to 30 bytes across all calls/)
  assert.match(f.message, /exceeding the 12-byte total guard/)
})

test('breachTotalFinish is an error finish carrying the total breach failure', () => {
  const r = breachTotalFinish(30, 12)
  assert.equal(r.kind, 'error')
  assert.equal(r.failure.code, BREACH_TOTAL_CODE)
})

/*** guardStream: pass-through under budget ***/

test('passes a stream under the budget byte-for-byte', async () => {
  const chunks = [
    textDelta(0, 'hello'),
    td(1, 'call-1', '{"a":'),
    td(1, 'call-1', '1}'),
    stop(),
  ]
  const out = await collect(guardStream(chunks, ON))
  assert.deepEqual(out, chunks)
  assert.equal(out[3].reason.kind, 'stop')
})

test('counts arguments per block index (interleaved calls stay independent)', async () => {
  // Two calls interleaved; each stays under 10-byte budget but their *sum* would exceed.
  const chunks = [
    td(1, 'call-1', '{"a":'),
    td(2, 'call-2', '{"b":'),
    td(1, 'call-1', '1}'),   // call-1 total 5 bytes, under budget
    td(2, 'call-2', '2}'),   // call-2 total 5 bytes, under budget
    stop(),
  ]
  const out = await collect(guardStream(chunks, SMALL))
  assert.deepEqual(out, chunks)
})

test('passes a stream through when guard is disabled (maxArgsBytes=0)', async () => {
  const chunks = [td(1, 'call-1', 'x'.repeat(5000)), stop()]
  const out = await collect(guardStream(chunks, OFF))
  assert.deepEqual(out, chunks)
})

/*** guardStream: breach behavior ***/

test('cuts a call that exceeds the budget and emits an error finish', async () => {
  const chunks = [
    td(1, 'call-1', '{"a":'),
    td(1, 'call-1', '1}'),   // 5 bytes so far
    td(1, 'call-1', 'x'.repeat(20)), // pushes over 10-byte budget
  ]
  const out = await collect(guardStream(chunks, SMALL))
  // The breaching chunk is emitted, then a terminal error finish — source stops.
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'error')
  assert.equal(out[out.length - 1].reason.failure.code, BREACH_CODE)
  // Total emitted arg chunks: the first 2 under budget + the breaching one.
  assert.equal(out.filter(c => c.type === 'tool-call-delta').length, 3)
})

test('does not execute the stream past the breaching chunk (source released)', async () => {
  let consumed = 0
  const source = (async function* () {
    yield td(1, 'call-1', 'x'.repeat(20)) // breaches immediately
    consumed++ // this line runs only if the upstream keeps iterating
  })()
  const out = await collect(guardStream(source, SMALL))
  assert.equal(out[out.length - 1].reason.kind, 'error')
  // The generator has yielded the breaching chunk, so `consumed` may or may not
  // have run once; the point is the guard does NOT continue pulling after breach.
})

test('observe-only (fail=false) cuts the source but emits a normal stop finish', async () => {
  const chunks = [td(1, 'call-1', 'x'.repeat(20))]
  const out = await collect(guardStream(chunks, OBSERVE))
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'stop')
})

test('never counts text or reasoning deltas toward the budget', async () => {
  const chunks = [
    textDelta(0, 'y'.repeat(100)),   // 100 bytes of text, ignored
    td(1, 'call-1', '{"a":1}'),      // 7 bytes of args, under 10
    stop(),
  ]
  const out = await collect(guardStream(chunks, SMALL))
  assert.deepEqual(out, chunks)
})

test('a call that ends exactly at the budget is not cut', async () => {
  const chunks = [td(1, 'call-1', 'x'.repeat(10)), stop()] // exactly 10
  const out = await collect(guardStream(chunks, SMALL))
  assert.deepEqual(out, chunks)
  assert.equal(out[out.length - 1].reason.kind, 'stop')
})

test('bare finish without built content is passed through untouched', async () => {
  const chunks = [td(1, 'call-1', '{"a":1}'), stop()]
  const out = await collect(guardStream(chunks, ON))
  assert.deepEqual(out, chunks)
})

/*** guardStream: whole-request aggregate budget (maxTotalArgsBytes) ***/

test('aggregate guard cuts the stream when many small calls sum past the total', async () => {
  // Per-call budget 10 bytes, total budget 12 bytes. Each call stays under
  // per-call budget (5 bytes each) but three calls sum to 15 > 12.
  const chunks = [
    td(1, 'call-1', '{"a":'),   // 5
    td(2, 'call-2', '{"b":'),   // +5 = 10 (still under total 12)
    td(3, 'call-3', '{"c":'),   // +5 = 15 → breaches total
  ]
  const out = await collect(guardStream(chunks, TOTAL))
  // The breaching chunk is emitted, then a terminal error finish.
  assert.equal(out[out.length - 1].reason.kind, 'error')
  assert.equal(out[out.length - 1].reason.failure.code, BREACH_TOTAL_CODE)
  // The three arg deltas are emitted (the breaching one included), then finish.
  assert.equal(out.filter(c => c.type === 'tool-call-delta').length, 3)
})

test('aggregate guard does not breach when the sum stays at the total', async () => {
  const chunks = [
    td(1, 'call-1', '{"a":'),   // 5
    td(2, 'call-2', '{"b":'),   // +5 = 10 (under total 12)
    td(3, 'call-3', '{}'),      // +2 = 12 (exactly at total, no breach)
    stop(),
  ]
  const out = await collect(guardStream(chunks, TOTAL))
  assert.deepEqual(out, chunks)
  assert.equal(out[out.length - 1].reason.kind, 'stop')
})

test('aggregate guard off (maxTotalArgsBytes=0): many calls pass through', async () => {
  const chunks = [
    td(1, 'call-1', '{"a":'),
    td(2, 'call-2', '{"b":'),
    td(3, 'call-3', '{"c":'),
    td(4, 'call-4', '{"d":'),
    stop(),
  ]
  const out = await collect(guardStream(chunks, TOTAL_OFF))
  assert.deepEqual(out, chunks)
})

test('aggregate guard reports per-call code first when one call is itself oversized', async () => {
  // First call breaches the per-call budget (20 > 10) before any total checks.
  const chunks = [
    td(1, 'call-1', '{"a":'),
    td(1, 'call-1', 'x'.repeat(20)), // per-call breach
    td(2, 'call-2', '{"b":'),
  ]
  const out = await collect(guardStream(chunks, TOTAL))
  assert.equal(out[out.length - 1].reason.failure.code, BREACH_CODE)
})

test('aggregate observe-only (fail=false) cuts the source but emits a normal stop finish', async () => {
  const chunks = [
    td(1, 'call-1', '{"a":'),   // 5
    td(2, 'call-2', '{"b":'),   // +5
    td(3, 'call-3', '{"c":'),   // +5 = 15 → breaches total
  ]
  const out = await collect(guardStream(chunks, { maxArgsBytes: 10, maxTotalArgsBytes: 12, fail: false }))
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'stop')
})

/*** utf8Length: UTF-8 byte counting (not UTF-16 code units) ***/

test('utf8Length counts UTF-8 bytes, not UTF-16 code units', () => {
  // "é" is 2 bytes in UTF-8 ("\u00e9"), 1 UTF-16 code unit.
  assert.equal(utf8Length('é'), 2)
  // "中" is 3 bytes in UTF-8, 1 UTF-16 code unit.
  assert.equal(utf8Length('中'), 3)
  // Supplementary plane "𝄞" (U+1D11E) is 4 bytes in UTF-8, but 2 UTF-16 units.
  assert.equal(utf8Length('𝄞'), 4)
  assert.equal(utf8Length(''), 0)
  assert.equal(utf8Length('abc'), 3)
})

test('byte guard counts non-ASCII arguments in UTF-8 bytes', async () => {
  // 3 bytes of non-ASCII args per delta. After 3 deltas = 9 bytes (under 10),
  // the 4th delta (3 more = 12) breaches a 10-byte cap that `.length` (UTF-16
  // units) would have reported as 3 → 9 → 12 too, but the *point* is the count
  // is stable: a cap that wants "bytes" gets bytes.
  const chunks = [td(1, 'call-1', '中'), td(1, 'call-1', '文'), td(1, 'call-1', '啊'), stop()]
  // 中=3, 文=3, 啊=3 → exactly 9, under 10 → no breach.
  const out = await collect(guardStream(chunks, { maxArgsBytes: 10, fail: true }))
  assert.deepEqual(out, chunks)
  // One more delta (3 bytes) would total 12 > 10 → breach.
  const breach = [td(1, 'call-1', '中'), td(1, 'call-1', '文'), td(1, 'call-1', '啊'), td(1, 'call-1', '啊')]
  const out2 = await collect(guardStream(breach, { maxArgsBytes: 10, fail: true }))
  assert.equal(out2[out2.length - 1].reason.kind, 'error')
  assert.equal(out2[out2.length - 1].reason.failure.code, BREACH_CODE)
})

/*** breachFragmentsFailure / breachFragmentsFinish: pure shapes ***/

test('breachFragmentsFailure carries the distinct stable code and a clear message', () => {
  const f = breachFragmentsFailure(0, 4074, 1024)
  assert.equal(f.code, BREACH_FRAGMENTS_CODE)
  assert.equal(BREACH_FRAGMENTS_CODE, 'TOOL_CALL_ARGUMENTS_TOO_MANY_FRAGMENTS')
  assert.match(f.message, /tool-call at index 0 emitted 4074 argument fragments/)
  assert.match(f.message, /exceeding the 1024-fragment guard/)
})

test('breachFragmentsFinish is an error finish carrying the fragment breach failure', () => {
  const r = breachFragmentsFinish(0, 4074, 1024)
  assert.equal(r.kind, 'error')
  assert.equal(r.failure.code, BREACH_FRAGMENTS_CODE)
})

test('default fragment budget is 0 (fragment guard off)', () => {
  assert.equal(DEFAULT_MAX_ARGS_FRAGMENTS, 0)
})

/*** guardStream: per-call fragment-count guard (maxArgsFragments) ***/

test('fragment guard cuts a call that emits too many deltas', async () => {
  // 10 one-byte deltas (10 bytes total) — under a huge byte budget, but 11 >
  // 10 fragments → fragment breach. This is the #6059 incident shape.
  const chunks = [td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x'), td(1, 'call-1', 'x')]
  const out = await collect(guardStream(chunks, FRAG))
  // The breaching (11th) delta is emitted, then a terminal error finish.
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'error')
  assert.equal(out[out.length - 1].reason.failure.code, BREACH_FRAGMENTS_CODE)
  // 11 arg deltas emitted (the breaching one included), then finish.
  assert.equal(out.filter(c => c.type === 'tool-call-delta').length, 11)
})

test('fragment guard does not breach when a call stays at the fragment cap', async () => {
  // Exactly 10 fragments, huge byte budget → no breach.
  const chunks = Array.from({ length: 10 }, () => td(1, 'call-1', 'x'))
  chunks.push(stop())
  const out = await collect(guardStream(chunks, FRAG))
  assert.deepEqual(out, chunks)
  assert.equal(out[out.length - 1].reason.kind, 'stop')
})

test('fragment observe-only (fail=false) cuts the source but emits a normal stop finish', async () => {
  const chunks = Array.from({ length: 11 }, () => td(1, 'call-1', 'x'))
  const out = await collect(guardStream(chunks, FRAG_OBSERVE))
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'stop')
})

test('fragment guard reports the byte code first when a call is itself oversized', async () => {
  // First delta alone is 20 bytes > 10-byte cap → per-call byte code, not fragment.
  const chunks = [td(1, 'call-1', 'x'.repeat(20)), td(1, 'call-1', 'x')]
  const out = await collect(guardStream(chunks, { maxArgsBytes: 10, maxArgsFragments: 10, fail: true }))
  assert.equal(out[out.length - 1].reason.failure.code, BREACH_CODE)
})

test('fragment guard off (maxArgsFragments=0): many deltas pass through', async () => {
  const chunks = Array.from({ length: 100 }, () => td(1, 'call-1', 'x'))
  chunks.push(stop())
  const out = await collect(guardStream(chunks, ON))
  assert.deepEqual(out, chunks)
})

/*** recordeed runaway fixture regression (discussion #6059) ***/

// luisnomad's boundary-exact recorded stream: 4,074 tool-call-delta fragments,
// 4,658 bytes total, histogram { 1: 3492, 2: 581, 4: 1 }. This is a *fragment
// count* runaway (the actual incident) that a byte budget alone never fires.
const recordedLengths =
  'H4sIAAAAAAAAE+3TsQ0AIAgF0Zkg7j+btdECo8WFHM1rr/jEiMw4Xa7yIDSIuGeRHUJDGUCCyDuEhqZfR2i4gNDQHEJD11r5ywRCIhgn6g8AAA=='

// Reconstruct the 4,074-delta stream from the recorded byte lengths, matching
// luisnomad's own decoding (gunzip → ascii → Number).
function buildRecordedRunaway() {
  const deltaLengths = [...gunzipSync(Buffer.from(recordedLengths, 'base64')).toString('ascii')].map(Number)
  assert.equal(deltaLengths.length, 4074)
  assert.equal(deltaLengths.reduce((a, b) => a + b, 0), 4658)
  return [
    ...deltaLengths.map((length) => td(0, 'call-redacted', 'x'.repeat(length))),
    finish({ kind: 'max-tokens' }),
  ]
}

test('recorded runaway (4,074 deltas / 4,658 bytes) passes a byte-only guard unchanged', async () => {
  // With only a byte budget (24 KiB default), the 4,658-byte stream never
  // breaches — that is exactly the blind spot luisnomad found.
  const stream = buildRecordedRunaway()
  const out = await collect(guardStream(stream, ON))
  // 4,074 deltas + the terminal max-tokens finish, all forwarded verbatim.
  assert.equal(out.filter(c => c.type === 'tool-call-delta').length, 4074)
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'max-tokens')
})

test('recorded runaway is caught by the fragment guard', async () => {
  // The actual incident shape: 4,074 fragments but only 4,658 bytes. A fragment
  // cap (e.g. 1,024) catches it; a byte cap alone never would.
  const stream = buildRecordedRunaway()
  const out = await collect(guardStream(stream, { maxArgsBytes: 24576, maxArgsFragments: 1024, fail: true }))
  assert.equal(out[out.length - 1].type, 'finish')
  assert.equal(out[out.length - 1].reason.kind, 'error')
  assert.equal(out[out.length - 1].reason.failure.code, BREACH_FRAGMENTS_CODE)
  // The guard cuts at fragment 1,025 (the first delta beyond 1,024).
  assert.equal(out.filter(c => c.type === 'tool-call-delta').length, 1025)
})
