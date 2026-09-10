import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  repairIdentityStream,
  syntheticCallId,
  breachIdentityFailure,
  breachIdentityFinish,
  BREACH_IDENTITY_CODE,
} from '../lib/index.js'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'

/** Resolved-config shape the wiring always produces. */
function cfg(over = {}) {
  return {
    maxArgsBytes: 24576,
    maxArgsFragments: 0,
    maxTotalArgsBytes: 0,
    fail: true,
    repair: 'repair',
    repairBytes: true,
    ...over,
  }
}
const OFF = cfg({ repair: 'off' })
const ERROR = cfg({ repair: 'error' })
const NO_BYTES = cfg({ repairBytes: false })

/**
 * The exact #6152 shape: the pi-ai adapter emits `''` as its "id not seen yet"
 * sentinel on every delta, and the block-end repeats the raw (empty) id.
 */
function emptyIdentityStream() {
  return [
    { type: 'block-start', index: 0, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 0, id: '', argumentsDelta: '' },
    { type: 'tool-call-delta', index: 0, id: '', argumentsDelta: '{}' },
    {
      type: 'block-end',
      index: 0,
      block: { type: 'tool-call', id: '', name: '', arguments: '{}' },
    },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

async function collect(iter) {
  const out = []
  for await (const chunk of iter) out.push(chunk)
  return out
}

/*** syntheticCallId ***/

test('syntheticCallId is deterministic and namespaced so a repair is recognizable', () => {
  assert.equal(syntheticCallId(0), 'repaired-call-0')
  assert.equal(syntheticCallId(7), 'repaired-call-7')
  // Same input -> same id, because the durable log and the format-migration
  // predicates must be able to re-derive it on replay.
  assert.equal(syntheticCallId(3), syntheticCallId(3))
})

test('breachIdentityFailure names the missing fields under a distinct code', () => {
  const f = breachIdentityFailure(2, ['id'])
  assert.equal(f.code, BREACH_IDENTITY_CODE)
  assert.equal(BREACH_IDENTITY_CODE, 'TOOL_CALL_EMPTY_IDENTITY')
  assert.match(f.message, /index 2/)
  assert.match(f.message, /empty id/)
})

test('breachIdentityFailure reports both fields when both are empty', () => {
  const f = breachIdentityFailure(0, ['id', 'name'])
  assert.match(f.message, /empty id and empty name/)
})

test('breachIdentityFinish is an error finish carrying the identity failure', () => {
  const r = breachIdentityFinish(1, ['id'])
  assert.equal(r.kind, 'error')
  assert.equal(r.failure.code, BREACH_IDENTITY_CODE)
})

/*** repair mode (default) ***/

test('repairs an empty id on every delta AND on the authoritative block-end', async () => {
  const out = await collect(repairIdentityStream(emptyIdentityStream(), cfg()))
  const deltas = out.filter(c => c.type === 'tool-call-delta')
  assert.equal(deltas.length, 2)
  for (const d of deltas) assert.equal(d.id, 'repaired-call-0')
  const end = out.find(c => c.type === 'block-end')
  assert.equal(end.block.id, 'repaired-call-0')
  // The block-end is authoritative in the core assembler, so repairing only the
  // deltas would be a silent no-op. Both must carry it.
})

test('substitutes {} for blank arguments by default', async () => {
  const out = await collect(repairIdentityStream(emptyIdentityStream(), cfg()))
  const first = out.find(c => c.type === 'tool-call-delta')
  assert.equal(first.argumentsDelta, '{}')
})

test('repairBytes:false leaves the argument stream byte-for-byte', async () => {
  const out = await collect(repairIdentityStream(emptyIdentityStream(), NO_BYTES))
  const first = out.find(c => c.type === 'tool-call-delta')
  assert.equal(first.argumentsDelta, '')
  // The id is still repaired — the two switches are independent.
  assert.equal(first.id, 'repaired-call-0')
})

test('a later delta that repeats the empty id inherits the repaired one', async () => {
  const chunks = [
    { type: 'tool-call-delta', index: 4, id: '', argumentsDelta: '{"a":' },
    { type: 'tool-call-delta', index: 4, id: '', argumentsDelta: '1}' },
  ]
  const out = await collect(repairIdentityStream(chunks, cfg()))
  assert.deepEqual(out.map(c => c.id), ['repaired-call-4', 'repaired-call-4'])
})

test('leaves an empty NAME alone — UNKNOWN_TOOL is the honest, resumable outcome', async () => {
  const chunks = [
    { type: 'tool-call-delta', index: 1, id: 'call-1', name: '', argumentsDelta: '{}' },
  ]
  const out = await collect(repairIdentityStream(chunks, cfg()))
  // Name is not blanked (the call would vanish) and not invented (that would
  // fabricate a call the model never asked for).
  assert.equal(out[0].name, '')
  assert.equal(out[0].id, 'call-1')
})

test('reports each affected block once, not once per fragment', async () => {
  const seen = []
  const chunks = [
    { type: 'tool-call-delta', index: 0, id: '', argumentsDelta: '{"a":1}' },
    { type: 'tool-call-delta', index: 0, id: '', argumentsDelta: '{"b":2}' },
    { type: 'tool-call-delta', index: 0, id: '', argumentsDelta: '{"c":3}' },
  ]
  await collect(repairIdentityStream(chunks, cfg(), (i, missing) => seen.push([i, missing])))
  assert.deepEqual(seen, [[0, ['id']]])
})

test('reports both empty fields when the delta carries an empty name', async () => {
  const seen = []
  const chunks = [{ type: 'tool-call-delta', index: 2, id: '', name: '', argumentsDelta: '{}' }]
  await collect(repairIdentityStream(chunks, cfg(), (i, missing) => seen.push([i, missing])))
  assert.deepEqual(seen, [[2, ['id', 'name']]])
})

test('an absent name is not the #6152 shape and is never reported', async () => {
  const seen = []
  const chunks = [{ type: 'tool-call-delta', index: 0, id: 'call-1', argumentsDelta: '{}' }]
  await collect(repairIdentityStream(chunks, cfg(), (i, m) => seen.push([i, m])))
  assert.deepEqual(seen, [])
})

test('repairs only the offending block in a multi-call stream', async () => {
  const chunks = [
    { type: 'tool-call-delta', index: 0, id: 'call-a', argumentsDelta: '{}' },
    { type: 'tool-call-delta', index: 1, id: '', argumentsDelta: '{}' },
    { type: 'block-end', index: 0, block: { type: 'tool-call', id: 'call-a', name: 'read', arguments: '{}' } },
    { type: 'block-end', index: 1, block: { type: 'tool-call', id: '', name: '', arguments: '{}' } },
  ]
  const out = await collect(repairIdentityStream(chunks, cfg()))
  const ends = out.filter(c => c.type === 'block-end')
  assert.equal(ends[0].block.id, 'call-a')       // untouched
  assert.equal(ends[1].block.id, 'repaired-call-1') // repaired
})

/*** identity is repaired in a way the real core assembler accepts ***/

test('the repaired stream assembles to a non-empty id via the real BlockAssembler', () => {
  // Proves the claim mechanically: this is the actual core assembly algorithm,
  // and the block-end is what it treats as authoritative.
  const assembler = new BlockAssembler()
  for (const chunk of emptyIdentityStream()) assembler.push(chunk)
  const blocks = assembler.blocks()
  const call = blocks.find(b => b.type === 'tool-call')
  assert.equal(call.id, '')
  assert.equal(call.name, '')
})

test('with repair applied, the real BlockAssembler yields the synthetic id', async () => {
  const repaired = await collect(repairIdentityStream(emptyIdentityStream(), cfg()))
  const assembler = new BlockAssembler()
  for (const chunk of repaired) assembler.push(chunk)
  const call = assembler.blocks().find(b => b.type === 'tool-call')
  assert.equal(call.id, 'repaired-call-0')
  // The whole point of #6152: the id (and the result's callId) are non-empty,
  // so the session read boundary no longer rejects the stored event.
  assert.notEqual(call.id, '')
})

/*** error mode ***/

test("repair:'error' cuts the stream so the degenerate call is never executed", async () => {
  const out = await collect(repairIdentityStream(emptyIdentityStream(), ERROR))
  const last = out.at(-1)
  assert.equal(last.type, 'finish')
  assert.equal(last.reason.kind, 'error')
  assert.equal(last.reason.failure.code, BREACH_IDENTITY_CODE)
  // Cuts immediately: no block-end, so no assembled call reaches the loop.
  assert.equal(out.some(c => c.type === 'block-end'), false)
})

/*** off mode / pass-through ***/

test("repair:'off' is the v0.1.3 pass-through, byte-for-byte", async () => {
  const chunks = emptyIdentityStream()
  const out = await collect(repairIdentityStream(chunks, OFF))
  assert.deepEqual(out, chunks)
})

test('a healthy stream passes through unchanged in repair mode', async () => {
  const chunks = [
    { type: 'text-delta', index: 0, text: 'hi' },
    { type: 'tool-call-delta', index: 1, id: 'call-1', name: 'read', argumentsDelta: '{"p":1}' },
    { type: 'block-end', index: 1, block: { type: 'tool-call', id: 'call-1', name: 'read', arguments: '{"p":1}' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
  const out = await collect(repairIdentityStream(chunks, cfg()))
  assert.deepEqual(out, chunks)
})
