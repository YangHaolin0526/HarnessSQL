// Short-lived child that runs ONE sqlite statement and prints JSON on stdout.
// Isolated in its own process so a runaway query is killable by a wall clock.
import { DatabaseSync } from 'node:sqlite'

const chunks = []
for await (const chunk of process.stdin) chunks.push(chunk)
const { db, sql, maxRows, maxCell } = JSON.parse(Buffer.concat(chunks).toString('utf8'))

const clip = (value) => {
  if (value === null || value === undefined) return null
  if (typeof value === 'bigint') return value.toString()
  if (value instanceof Uint8Array) return `<blob ${value.byteLength} bytes>`
  const text = typeof value === 'string' ? value : String(value)
  return text.length > maxCell ? `${text.slice(0, maxCell)}…` : text
}

try {
  const handle = new DatabaseSync(db, { readOnly: true })
  const statement = handle.prepare(sql)
  const rows = statement.all()
  const columns = rows.length ? Object.keys(rows[0]) : []
  const body = rows.slice(0, maxRows).map((row) => columns.map((c) => clip(row[c])))
  process.stdout.write(JSON.stringify({ ok: true, columns, rows: body, total: rows.length }))
  handle.close()
} catch (error) {
  process.stdout.write(JSON.stringify({ ok: false, error: String(error?.message ?? error) }))
}
