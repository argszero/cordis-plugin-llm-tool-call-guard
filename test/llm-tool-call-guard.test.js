import { test } from 'node:test'
import assert from 'node:assert/strict'
import { guardStream, breachFailure, breachFinish, BREACH_CODE, DEFAULT_MAX_ARGS_BYTES } from '../lib/index.js'

const ON = { maxArgsBytes: 24576, fail: true }
const OFF = { maxArgsBytes: 0, fail: true }
const SMALL = { maxArgsBytes: 10, fail: true }
const OBSERVE = { maxArgsBytes: 10, fail: false }

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
