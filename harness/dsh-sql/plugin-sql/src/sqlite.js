import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const CHILD = fileURLToPath(new URL('./query-child.js', import.meta.url))

const READ_ONLY = /^\s*(?:with|select|pragma\s+table_info|explain)\b/i

export class SqlError extends Error {}

export function assertReadOnly(sql) {
  const trimmed = String(sql ?? '').trim()
  if (!trimmed) throw new SqlError('query is empty')
  if (!READ_ONLY.test(trimmed)) {
    throw new SqlError('only read-only SELECT / WITH / EXPLAIN queries are allowed in this harness')
  }
  // One statement per call: a trailing `;` is fine, a second statement is not.
  const withoutTrailing = trimmed.replace(/;\s*$/, '')
  if (withoutTrailing.includes(';')) throw new SqlError('send exactly one statement per call')
  return withoutTrailing
}

// Runs one statement in a child process so `timeoutMs` is a real wall clock and
// a wedged query cannot take the agent down with it.
export function runQuery({ db, sql, maxRows = 50, maxCell = 300, timeoutMs = 60000, signal }) {
  return new Promise((resolve, reject) => {
    const child = execFile(
      process.execPath,
      [CHILD],
      { timeout: timeoutMs, maxBuffer: 32 * 1024 * 1024, killSignal: 'SIGKILL' },
      (error, stdout) => {
        if (error && error.killed) return reject(new SqlError(`query exceeded ${timeoutMs} ms and was killed`))
        if (error && !stdout) return reject(new SqlError(String(error.message)))
        try {
          resolve(JSON.parse(stdout))
        } catch {
          reject(new SqlError(`sqlite child returned unparsable output: ${stdout.slice(0, 400)}`))
        }
      },
    )
    const abort = () => child.kill('SIGKILL')
    signal?.addEventListener('abort', abort, { once: true })
    child.on('close', () => signal?.removeEventListener('abort', abort))
    child.stdin.end(JSON.stringify({ db, sql, maxRows, maxCell }))
  })
}

export function renderRows(result, { note } = {}) {
  if (!result.ok) return `ERROR: ${result.error}`
  if (!result.columns.length) return 'Query succeeded and returned no rows.'
  const header = result.columns.join(' | ')
  const body = result.rows.map((row) => row.map((c) => (c === null ? 'NULL' : c)).join(' | '))
  const shown = result.rows.length
  const tail =
    shown < result.total ? `\n... ${result.total} rows total, ${shown} shown.` : `\n(${result.total} rows)`
  return [note, header, '-'.repeat(Math.min(header.length, 120)), ...body].filter(Boolean).join('\n') + tail
}
