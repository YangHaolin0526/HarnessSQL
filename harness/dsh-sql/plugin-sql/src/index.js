// A SQL-only model-facing toolset for the DeepSeek Harness.
//
// Everything this harness lets an agent do is here: inspect a SQLite database,
// run read-only queries against it, and submit one final answer query. There is
// no shell, no filesystem, no web — the composition disables those rows, so this
// plugin is the agent's entire capability surface.
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { assertReadOnly, renderRows, runQuery, SqlError } from './sqlite.js'
import { renderBigQueryRows, runBigQuery } from './bigquery.js'
import { renderPostgres, runPostgres } from './postgres.js'
import { applyBirdInteract } from './bird-interact.js'

export const name = 'dsh-plugin-sql'
export const inject = ['tools']

const text = (value) => [{ type: 'text', text: value }]
const stringOutput = { schema: { type: 'string' }, render: (_args, value) => text(value) }

export function apply(ctx, config = {}) {
  const backend = String(config.backend ?? process.env.DSH_SQL_BACKEND ?? 'sqlite').toLowerCase()
  if (backend === 'bird-interact') {
    applyBirdInteract(ctx, config)
    return
  }
  const db = config.db ?? process.env.DSH_SQL_DB
  const schemaDir = config.schemaDir ?? process.env.DSH_SQL_BQ_SCHEMA_DIR
  const credentialPath = config.credentialPath ?? process.env.DSH_SQL_BQ_CREDENTIAL
  const instanceId = config.instanceId ?? process.env.DSH_SQL_INSTANCE_ID
  const maximumBytesBilled = Number(
    config.maximumBytesBilled ?? process.env.DSH_SQL_BQ_MAX_BYTES ?? 20 * 1024 ** 3,
  )
  if (backend === 'sqlite' && !db) {
    throw new Error('dsh-plugin-sql: no SQLite database configured (config.db or DSH_SQL_DB)')
  }
  if (backend === 'bigquery' && (!schemaDir || !credentialPath)) {
    throw new Error('dsh-plugin-sql: BigQuery requires DSH_SQL_BQ_SCHEMA_DIR and DSH_SQL_BQ_CREDENTIAL')
  }
  if (!['sqlite', 'bigquery', 'postgres'].includes(backend)) throw new Error(`unsupported SQL backend: ${backend}`)
  const answerPath = config.answerPath ?? process.env.DSH_SQL_ANSWER
  const maxRows = config.maxRows ?? 50
  const timeoutMs = config.queryTimeoutMs ?? 60000
  const sampleRows = config.sampleRows ?? 3

  const query = (sql, exec, opts = {}) =>
    backend === 'sqlite'
      ? runQuery({ db, sql, maxRows: opts.maxRows ?? maxRows, timeoutMs, signal: exec.signal })
      : backend === 'bigquery' ? runBigQuery({
          mode: 'query', schemaDir, credentialPath, sql,
          maxRows: opts.maxRows ?? maxRows, timeoutMs, maximumBytesBilled, signal: exec.signal,
        })
      : runPostgres({ mode: 'query', database: db, sql, maxRows: opts.maxRows ?? maxRows, timeoutMs, signal: exec.signal })

  ctx.tools.register(
    defineTool({
      name: 'sql_list_tables',
      description:
        'List every table/view (or BigQuery sharded-table family) available for this task. Start here: you cannot see the database any other way.',
      parameters: {},
      output: stringOutput,
      async execute(_args, exec) {
        if (backend === 'bigquery') {
          const result = await runBigQuery({ mode: 'list', schemaDir, credentialPath, timeoutMs, signal: exec.signal })
          return result.ok ? result.text : `ERROR: ${result.error}`
        }
        if (backend === 'postgres') {
          const result = await runPostgres({ mode: 'list', database: db, timeoutMs, signal: exec.signal })
          if (!result.ok) return `ERROR: ${result.error}`
          return result.rows.map(([type, schema, table]) => `${type.padEnd(10)} ${schema}.${table}`).join('\n')
        }
        const listed = await query(
          "SELECT type, name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name",
          exec,
          { maxRows: 5000 },
        )
        if (!listed.ok) return `ERROR: ${listed.error}`
        if (!listed.rows.length) return 'The database has no tables.'
        const lines = []
        for (const [type, tableName] of listed.rows) {
          const counted = await query(`SELECT COUNT(*) AS n FROM "${tableName}"`, exec, { maxRows: 1 })
          const rows = counted.ok && counted.rows.length ? counted.rows[0][0] : '?'
          lines.push(`${type.padEnd(5)} ${tableName}  (${rows} rows)`)
        }
        return `${lines.length} objects in the database:\n${lines.join('\n')}`
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'sql_schema',
      description:
        'Show the CREATE statement and a few sample rows for one or more tables. Read the schema of every table you intend to query before writing SQL.',
      parameters: {
        tables: {
          type: 'array',
          items: { type: 'string' },
          required: true,
          description: 'Table or view names, exactly as sql_list_tables reported them.',
        },
      },
      output: stringOutput,
      async execute(args, exec) {
        if (!args.tables.length) throw new SqlError('pass at least one table name')
        if (backend === 'bigquery') {
          const result = await runBigQuery({
            mode: 'schema', schemaDir, credentialPath, tables: args.tables,
            timeoutMs, maxCell: 500, signal: exec.signal,
          })
          return result.ok ? result.text : `ERROR: ${result.error}`
        }
        if (backend === 'postgres') {
          const result = await runPostgres({ mode: 'schema', database: db, tables: args.tables, timeoutMs, signal: exec.signal })
          if (!result.ok) return `ERROR: ${result.error}`
          const groups = new Map()
          for (const [schema, table, column, type, nullable, defaultValue] of result.rows) {
            const key = `${schema}.${table}`; if (!groups.has(key)) groups.set(key, [])
            groups.get(key).push(`  "${column}" ${type}${nullable === 'NO' ? ' NOT NULL' : ''}${defaultValue ? ` DEFAULT ${defaultValue}` : ''}`)
          }
          const blocks = []
          for (const [table, columns] of groups) {
            const sample = await query(`SELECT * FROM ${table} LIMIT ${sampleRows}`, exec, { maxRows: sampleRows })
            blocks.push(`CREATE TABLE ${table} (\n${columns.join(',\n')}\n);\n\n-- sample rows\n${renderPostgres(sample)}`)
          }
          return blocks.length ? blocks.join('\n\n') : 'No matching tables.'
        }
        const blocks = []
        for (const tableName of args.tables.slice(0, 20)) {
          const ddl = await query(
            `SELECT sql FROM sqlite_master WHERE name = '${tableName.replace(/'/g, "''")}'`,
            exec,
            { maxRows: 1 },
          )
          if (!ddl.ok || !ddl.rows.length) {
            blocks.push(`-- ${tableName}: no such table or view`)
            continue
          }
          const sample = await query(`SELECT * FROM "${tableName}" LIMIT ${sampleRows}`, exec, {
            maxRows: sampleRows,
          })
          blocks.push(
            [`-- ${tableName}`, ddl.rows[0][0], '', renderRows(sample, { note: `-- sample rows` })].join('\n'),
          )
        }
        return blocks.join('\n\n')
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'sql_exec',
      description:
        `Run one read-only ${backend === 'bigquery' ? 'GoogleSQL BigQuery' : 'SQLite'} query and see its result. Use this to explore values and test your answer. BigQuery queries are dry-run first and capped at ${maximumBytesBilled / 1024 ** 3} GiB.`,
      parameters: {
        query: { type: 'string', required: true, description: `One ${backend === 'bigquery' ? 'GoogleSQL' : 'SQLite'} SELECT or WITH statement.` },
        limit: { type: 'number', description: `Max rows to display (default ${maxRows}).` },
      },
      output: stringOutput,
      async execute(args, exec) {
        const sql = backend === 'postgres' ? String(args.query ?? '').trim().replace(/;\s*$/, '') : assertReadOnly(args.query)
        const result = await query(sql, exec, { maxRows: args.limit ?? maxRows })
        return backend === 'bigquery' ? renderBigQueryRows(result) : backend === 'postgres' ? renderPostgres(result) : renderRows(result)
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'sql_submit',
      description:
        'Submit your final answer query. It is executed once to prove it runs; a query that errors or exceeds the scan cap is rejected and you must fix it. Call this exactly once when confident.',
      parameters: {
        query: { type: 'string', description: 'One final SQL statement.' },
        queries: { type: 'array', items: { type: 'string' }, description: 'Ordered final PostgreSQL SQL statements.' },
      },
      output: stringOutput,
      async execute(args, exec) {
        // `queries` is advertised on every backend but was only honoured on
        // postgres, so a non-postgres agent that filled it in -- which the
        // schema invites -- hit assertReadOnly(undefined), got back 'query is
        // empty' with no hint about which field to use, and retried the
        // identical call until the task ran out of turns (133 times on
        // local025). Accept either field everywhere; one statement per call
        // still applies off postgres, so take the first entry.
        const singleQuery = args.query ?? (Array.isArray(args.queries) ? args.queries[0] : undefined)
        const statements = backend === 'postgres' ? (args.queries?.length ? args.queries : [args.query]).map(s => String(s ?? '').trim()).filter(Boolean) : [assertReadOnly(singleQuery)]
        if (!statements.length) throw new SqlError('submit at least one SQL statement')
        const result = backend === 'postgres' ? await runPostgres({ mode: 'query', database: db, statements, maxRows: 10, timeoutMs, signal: exec.signal }) : await query(statements[0], exec, { maxRows: 10 })
        if (!result.ok) return `REJECTED — the query failed to execute:\n${result.error}\n\nFix it and submit again.`
        if (answerPath) {
          mkdirSync(dirname(answerPath), { recursive: true })
          writeFileSync(answerPath, backend === 'postgres' ? JSON.stringify({ instance_id: instanceId, predicted_sql: statements }, null, 2) + '\n' : `${statements[0]};\n`, 'utf8')
        }
        const preview = backend === 'bigquery' ? renderBigQueryRows(result) : backend === 'postgres' ? renderPostgres(result) : renderRows(result, { note: '-- result preview' })
        return `ACCEPTED. Recorded as the final answer.\n\n${preview}`
      },
    }),
  )
}
