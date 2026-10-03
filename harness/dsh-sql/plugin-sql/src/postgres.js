import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { SqlError } from './sqlite.js'
const CHILD = fileURLToPath(new URL('./postgres-child.py', import.meta.url))

export function runPostgres(payload) {
  return new Promise((resolve, reject) => {
    const timeoutMs = payload.timeoutMs ?? 60000
    const child = execFile(process.env.DSH_SQL_PYTHON ?? 'python3', [CHILD], { timeout: timeoutMs + 15000, maxBuffer: 32 * 1024 * 1024, killSignal: 'SIGKILL' }, (error, stdout, stderr) => {
      if (error?.killed) return reject(new SqlError(`PostgreSQL helper exceeded ${timeoutMs} ms`))
      if (error && !stdout) return reject(new SqlError(String(stderr || error.message).slice(0, 1000)))
      try { resolve(JSON.parse(stdout)) } catch { reject(new SqlError(`invalid PostgreSQL helper output: ${(stdout || stderr).slice(0, 800)}`)) }
    })
    const abort = () => child.kill('SIGKILL'); payload.signal?.addEventListener('abort', abort, { once: true })
    child.on('close', () => payload.signal?.removeEventListener('abort', abort))
    child.stdin.end(JSON.stringify(payload))
  })
}

export function renderPostgres(result) {
  if (!result.ok) return `ERROR: ${result.error}`
  return (result.outputs ?? [result]).map((out) => {
    if (!out.columns?.length) return out.status ?? 'Command validated successfully (transaction rolled back).'
    const header = out.columns.join(' | ')
    return [header, '-'.repeat(Math.min(header.length, 120)), ...out.rows.map(r => r.map(v => v === null ? 'NULL' : String(v)).join(' | ')), out.truncated ? '... more rows not shown' : `(${out.rows.length} rows shown)`].join('\n')
  }).join('\n\n')
}
