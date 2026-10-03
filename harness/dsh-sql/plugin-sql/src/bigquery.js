import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'

import { SqlError } from './sqlite.js'

const CHILD = fileURLToPath(new URL('./bigquery-child.py', import.meta.url))

export function runBigQuery({
  mode,
  schemaDir,
  credentialPath,
  sql,
  tables,
  maxRows = 50,
  maxCell = 300,
  timeoutMs = 90000,
  maximumBytesBilled = 20 * 1024 ** 3,
  signal,
}) {
  return new Promise((resolve, reject) => {
    const python = process.env.DSH_SQL_PYTHON ?? 'python3'
    const child = execFile(
      python,
      [CHILD],
      {
        timeout: Math.max(timeoutMs + 30000, 90000),
        maxBuffer: 32 * 1024 * 1024,
        killSignal: 'SIGKILL',
      },
      (error, stdout, stderr) => {
        if (error?.killed) return reject(new SqlError(`BigQuery helper exceeded ${timeoutMs} ms and was killed`))
        if (error && !stdout) return reject(new SqlError(String(stderr || error.message).slice(0, 1000)))
        try {
          resolve(JSON.parse(stdout))
        } catch {
          reject(new SqlError(`BigQuery helper returned unparsable output: ${(stdout || stderr).slice(0, 800)}`))
        }
      },
    )
    const abort = () => child.kill('SIGKILL')
    signal?.addEventListener('abort', abort, { once: true })
    child.on('close', () => signal?.removeEventListener('abort', abort))
    child.stdin.end(
      JSON.stringify({
        mode,
        schemaDir,
        credentialPath,
        sql,
        tables,
        maxRows,
        maxCell,
        timeoutMs,
        maximumBytesBilled,
      }),
    )
  })
}

export function renderBigQueryRows(result) {
  if (!result.ok) return `ERROR: ${result.error}`
  if (!result.columns?.length) return 'Query succeeded and returned no rows.'
  const header = result.columns.join(' | ')
  const body = result.rows.map((row) => row.map((c) => (c === null ? 'NULL' : c)).join(' | '))
  const shown = result.rows.length
  const tail = shown < result.total ? `\n... ${result.total} rows total, ${shown} shown.` : `\n(${result.total} rows)`
  const gib = 1024 ** 3
  const usage = `-- BigQuery scan: ${(result.bytesProcessed / gib).toFixed(5)} GiB processed; ${(result.estimatedBytes / gib).toFixed(5)} GiB estimated by dry-run.`
  return [usage, header, '-'.repeat(Math.min(header.length, 120)), ...body].join('\n') + tail
}
