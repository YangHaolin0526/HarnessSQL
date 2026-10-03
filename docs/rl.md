# RL runtime contract

The RL implementation is an integration layer for Slime rather than a fork of it. It expects Slime, Megatron-LM, and SGLang to be installed by the training environment.

Each prompt row carries `metadata.sample_id`. `text2sql.task_data` resolves that ID to a database, hidden oracle hashes, expected columns, and a KTX connection. `text2sql.generate` opens an isolated rollout session, launches the configured harness, records the exact sampled token IDs and log probabilities, extracts the submitted SQL, and assigns reward `1.0` only when execution matches the hidden result contract.

Required environment variables:

- `HARNESS_SQL_SYNTH_ROOT`: validated Spider2-derived task batches.
- `HARNESS_SQL_SPIDER_DB_DIR`: Spider2 SQLite database directory.
- `HARNESS_SQL_DBT_SYNTH_ROOT`: validated DBT-derived task batches.
- `HARNESS_SQL_DBT_DB_DIR`: DBT-derived SQLite database directory.
- `HARNESS_SQL_KTX_BIN`: KTX CLI entry point used by the rollout harness.
- `HARNESS_SQL_KTX_PROJECT`: KTX project containing the registered read-only connections.

Optional controls include `SQL_MAX_CONCURRENT_ROLLOUTS`, `SQL_ROLLOUT_THREADS`, `SQL_GENERATE_GUARD_SEC`, `SQL_MAX_TURNS`, `SQL_TURN_TIMEOUT_SEC`, `SQL_QUERY_TIMEOUT_SEC`, `MAX_TOOL_OUTPUT_CHARS`, `SHIM_BIND_HOST`, and `SHIM_PORT`.

Before training, replay every oracle with `training/rl/scripts/replay_all_oracles.py`. A task whose own oracle cannot earn reward is excluded rather than allowed to silently create an all-zero policy-gradient group.
