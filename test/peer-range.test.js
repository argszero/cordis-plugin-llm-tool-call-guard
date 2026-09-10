import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'))

/**
 * Guard the peer range against the two ways it has already been wrong.
 *
 * Every dsh release published today is a prerelease (0.1.2-rc.1, 0.1.5-rc.1,
 * ...), and a semver comparator only admits prereleases that share its own
 * major.minor.patch tuple. That makes two natural-looking ranges fail:
 *
 *   ">=0.1.2"             matches NOTHING -> npm install dies with ETARGET
 *   ">=0.1.2-rc.1 <0.2.0" matches only 0.1.2-rc.1 -> a 0.1.5-line user
 *                         gets ERESOLVE, because the "^0.1.5-x" tuple that
 *                         dsh-agent/dsh-llm publish peers against is not 0.1.2
 *
 * The npm latest tag for @deepseek-ai/dsh is 0.1.2-rc.1 while next/alpha point
 * at the 0.1.5 line, so BOTH are in active use and the range must carry one
 * comparator group per supported tuple line.
 */
test('peer range admits both supported dsh prerelease lines', () => {
  const range = pkg.peerDependencies['@deepseek-ai/dsh-llm']
  assert.ok(range, 'the dsh peer dependency must be declared')

  // Form 1: a bare release bound resolves to no published version at all.
  assert.ok(
    !/^(>=\^~)?\s*\d+\.\d+\.\d+\s*$/.test(range.trim()),
    `a bare non-prerelease comparator ("${range}") matches no published dsh version`,
  )

  // The lower bound must name a concrete prerelease, not a bare release.
  assert.match(range, /0\.1\.2-rc\.\d+/, 'expected a 0.1.2-rc.N lower bound')

  // Form 2: a single comparator group silently excludes the 0.1.5 line, which
  // the next/alpha dist-tags serve. Requiring an explicit 0.1.5 prerelease in
  // the range catches that regression.
  assert.match(
    range,
    /0\.1\.5-(alpha|beta|rc)\.\d+/,
    'the range must also admit the 0.1.5 prerelease line (next/alpha tags)',
  )
})
