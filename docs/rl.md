# RL runtime contract

The RL implementation is the portable form of the production training pipeline. It
is an integration layer for Slime rather than a fork of it and expects Slime,
Megatron-LM, SGLang, Ray, and a torch-distributed reference checkpoint to be
provided by the training environment.

## Training profiles

| Profile | Advantage / ratio | Sampling filter | Clip low/high | Reference KL |
|---|---|---|---|---|
| `grpo` | token-level GRPO | reject aborted groups | 0.20 / 0.20 | 0.01 |
| `dapo` | group-relative | reject aborted and zero-variance groups | 0.20 / 0.28 | 0 |
| `gspo` | sequence-level ratio | reject aborted groups | 0.0003 / 0.0003 | 0 |

DAPO combines Slime's group-relative estimator, dynamic zero-signal filtering,
asymmetric clipping, and token-level loss. GSPO uses Slime's native `gspo`
estimator. `NUM_STEPS_PER_ROLLOUT` defaults to 2 so the second optimizer step is
off-policy with respect to the sampled rollout and the clipping behavior is not
degenerate.

Each prompt row carries `metadata.sample_id`. `text2sql.task_data` resolves that ID to a database, hidden oracle hashes, expected columns, and a KTX connection. `text2sql.generate` opens an isolated rollout session, launches the configured harness, records the exact sampled token IDs and log probabilities, extracts the submitted SQL, and assigns reward `1.0` only when execution matches the hidden result contract.

Required environment variables:

- `HARNESS_SQL_SYNTH_ROOT`: validated Spider2-derived task batches.
- `HARNESS_SQL_SPIDER_DB_DIR`: Spider2 SQLite database directory.
- `HARNESS_SQL_DBT_SYNTH_ROOT`: validated DBT-derived task batches.
- `HARNESS_SQL_DBT_DB_DIR`: DBT-derived SQLite database directory.
- `HARNESS_SQL_KTX_BIN`: KTX CLI entry point used by the rollout harness.
- `HARNESS_SQL_KTX_PROJECT`: KTX project containing the registered read-only connections.
- `SLIME_ROOT`: compatible Slime checkout containing `train.py`.
- `HF_CHECKPOINT`: Hugging Face policy/SFT checkpoint.
- `REF_LOAD`: matching Megatron torch-distributed reference checkpoint.
- `SAVE_DIR`: output checkpoint directory.
- `PROMPT_DATA`: training JSONL produced by `build_merged_rl_data.py`.

`HARNESS_SQL_SYNTH_BATCHES` and `HARNESS_SQL_DBT_BATCHES` optionally select
comma-separated synthesized batch directories. This allows RL data preparation
to consume arbitrary generated task counts instead of relying on the original
experiment sizes.

Training selection variables:

- `MODEL_SIZE=qwen3_8b|qwen3_14b` selects a shipped Megatron model config.
- `ALGORITHM=grpo|dapo|gspo` selects the objective profile.
- `SFT_CONTEXT_LEN=8192|16384|32768` pairs the SFT context and rollout budget.
- `EVAL_INTERVAL` plus `EVAL_DATA` enables held-out evaluation.

Optional controls include `SQL_MAX_CONCURRENT_ROLLOUTS`, `SQL_ROLLOUT_THREADS`, `SQL_GENERATE_GUARD_SEC`, `SQL_MAX_TURNS`, `SQL_TURN_TIMEOUT_SEC`, `SQL_QUERY_TIMEOUT_SEC`, `MAX_TOOL_OUTPUT_CHARS`, `SHIM_BIND_HOST`, and `SHIM_PORT`.

Before training, replay every oracle with `training/rl/scripts/replay_all_oracles.py`. A task whose own oracle cannot earn reward is excluded rather than allowed to silently create an all-zero policy-gradient group.

Run `training/rl/launch_ray.sh` in a configured Slime environment, or submit
`training/rl/slurm/train_rl_dsh.sbatch` after exporting the same variables.
The Ray launcher invokes `training/rl/train_rl_dsh.sh`, which contains the
actual Slime/Megatron training command. The training driver uses four actor
GPUs and four rollout GPUs by default;
these and the batch, sampling, optimizer, timeout, and context settings are all
environment-variable overrides.
