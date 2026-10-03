import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import { defineTool } from '@deepseek-ai/dsh-tools'

const text = (value) => [{ type: 'text', text: String(value) }]
const stringOutput = { schema: { type: 'string' }, render: (_args, value) => text(value) }

const COSTS = {
  execute_sql: 1,
  get_schema: 1,
  get_all_column_meanings: 1,
  get_column_meaning: 0.5,
  get_all_external_knowledge_names: 0.5,
  get_knowledge_definition: 0.5,
  get_all_knowledge_definitions: 1,
  ask_user: 2,
  submit_sql: 3,
}

async function post(base, path, payload, signal) {
  const response = await fetch(`${base.replace(/\/$/, '')}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })
  const body = await response.text()
  if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}: ${body.slice(0, 1000)}`)
  return JSON.parse(body)
}

export function applyBirdInteract(ctx, config = {}) {
  const taskId = config.taskId ?? process.env.BIRD_TASK_ID
  const dbEnv = config.dbEnv ?? process.env.BIRD_DB_ENV_URL ?? 'http://127.0.0.1:6002'
  const userEnv = config.userEnv ?? process.env.BIRD_USER_ENV_URL ?? 'http://127.0.0.1:6001'
  const statePath = config.statePath ?? process.env.BIRD_STATE_PATH
  const initialBudget = Number(config.budget ?? process.env.BIRD_INITIAL_BUDGET ?? 12)
  const dialect = config.dialect ?? process.env.BIRD_SQL_DIALECT ?? 'PostgreSQL'
  if (!taskId) throw new Error('dsh-plugin-sql bird-interact backend requires BIRD_TASK_ID')

  const state = {
    task_id: taskId,
    initial_budget: initialBudget,
    budget_remaining: initialBudget,
    total_reward: 0,
    current_phase: 1,
    phase1_passed: false,
    phase2_passed: false,
    finished: false,
    actions: [],
    submissions: [],
  }

  const save = () => {
    if (!statePath) return
    mkdirSync(dirname(statePath), { recursive: true })
    writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, 'utf8')
  }
  save()

  const run = async (name, action, exec) => {
    if (state.finished) return 'The BIRD-Interact task is already finished. Do not call more tools.'
    if (state.budget_remaining <= 0 && name !== 'submit_sql') {
      return `Action blocked: the bird-coin budget is exhausted. You MUST call submit_sql with your best ${dialect} now.`
    }
    const before = state.budget_remaining
    const cost = state.budget_remaining <= 0 && name === 'submit_sql' ? 0 : COSTS[name]
    state.budget_remaining -= cost
    let result
    try {
      result = await action()
    } catch (error) {
      result = `Tool error: ${error?.message ?? error}`
    }
    state.actions.push({ name, cost, budget_before: before, budget_after: state.budget_remaining, result: String(result).slice(0, 2000) })
    save()
    const note = state.finished
      ? '\n\n[SYSTEM NOTE: Task finished. Do not call more tools.]'
      : state.budget_remaining <= 0
        ? '\n\n[SYSTEM NOTE: Budget exhausted. You MUST call submit_sql next.]'
        : `\n\n[SYSTEM NOTE: Remaining budget: ${state.budget_remaining.toFixed(1)}/${initialBudget.toFixed(1)} bird-coins.]`
    return `${result}${note}`
  }

  const register = (name, description, parameters, action) => ctx.tools.register(defineTool({
    name, description, parameters, output: stringOutput,
    async execute(args, exec) { return run(name, () => action(args, exec), exec) },
  }))

  register('execute_sql', `Execute one read-only ${dialect} SELECT/WITH/EXPLAIN query. Cost: 1 bird-coin.`, {
    sql: { type: 'string', required: true, description: `One ${dialect} query.` },
  }, async ({ sql }, exec) => {
    const r = await post(dbEnv, '/execute', { task_id: taskId, sql }, exec.signal)
    return r.success ? r.result : `SQL execution error: ${r.error}`
  })

  register('get_schema', 'Get the complete database schema with examples. Cost: 1 bird-coin.', {},
    async (_args, exec) => (await post(dbEnv, '/schema', { task_id: taskId }, exec.signal)).schema)

  register('get_all_column_meanings', 'Get meanings for every database column. Cost: 1 bird-coin.', {},
    async (_args, exec) => (await post(dbEnv, '/all_column_meanings', { task_id: taskId }, exec.signal)).column_meanings)

  register('get_column_meaning', 'Get the documented meaning of one column. Cost: 0.5 bird-coins.', {
    table_name: { type: 'string', required: true },
    column_name: { type: 'string', required: true },
  }, async (args, exec) => (await post(dbEnv, '/column_meaning', { task_id: taskId, ...args }, exec.signal)).meaning)

  register('get_all_external_knowledge_names', 'List accessible external-knowledge names. Cost: 0.5 bird-coins.', {},
    async (_args, exec) => JSON.stringify((await post(dbEnv, '/knowledge_names', { task_id: taskId }, exec.signal)).names))

  register('get_knowledge_definition', 'Get one accessible external-knowledge definition. Cost: 0.5 bird-coins.', {
    knowledge_name: { type: 'string', required: true },
  }, async ({ knowledge_name }, exec) => (await post(dbEnv, '/knowledge', { task_id: taskId, knowledge_name }, exec.signal)).knowledge)

  register('get_all_knowledge_definitions', 'Get all accessible external-knowledge definitions. Cost: 1 bird-coin.', {},
    async (_args, exec) => (await post(dbEnv, '/knowledge', { task_id: taskId }, exec.signal)).knowledge)

  register('ask_user', 'Ask the GPT-4o user simulator one concise clarification question. Cost: 2 bird-coins.', {
    question: { type: 'string', required: true, description: 'Ask exactly one clarification question.' },
  }, async ({ question }, exec) => {
    const r = await post(userEnv, '/ask', { task_id: taskId, question }, exec.signal)
    return `User response: ${r.answer}`
  })

  register('submit_sql', `Submit the final ${dialect} for hidden execution grading. Cost: 3 bird-coins. A successful phase-1 submission may return a phase-2 follow-up.`, {
    sql: { type: 'string', required: true, description: `The final ${dialect} statement for the current phase.` },
  }, async ({ sql }, exec) => {
    const r = await post(dbEnv, '/submit', { task_id: taskId, sql }, exec.signal)
    state.submissions.push({ phase: state.current_phase, sql, ...r })
    state.total_reward += Number(r.reward ?? 0)
    if (r.passed && r.phase_completed === 1) {
      state.phase1_passed = true
      if (r.has_follow_up) {
        state.current_phase = 2
        await post(userEnv, '/phase_transition', { task_id: taskId }, exec.signal)
        return `${r.message}\n\nUSER FOLLOW-UP QUERY (Phase 2):\n${r.follow_up_query}\n\nContinue working on this follow-up and submit its ${dialect}.`
      }
      state.finished = true
    } else if (r.passed && r.phase_completed === 2) {
      state.phase2_passed = true
      state.finished = true
    }
    if (!r.passed && state.budget_remaining <= 0) {
      state.finished = true
      return `${r.message}\n\nThe bird-coin budget is exhausted; this is the final submission and the task is finished.`
    }
    return r.message
  })
}
