# HarnessSQL

HarnessSQL is an end-to-end research codebase for harness-aware text-to-SQL:

1. convert Spider 2.0-DBT databases from DuckDB to SQLite and build catalogs;
2. synthesize executable SQL tasks from Spider 2.0-Lite SQLite and DBT-derived SQLite;
3. run hosted APIs or local vLLM models through the `dsh-sql` tool harness and retain trajectories;
4. turn verified trajectories into SFT datasets and train Qwen-style causal LMs;
5. run execution-reward RL through Slime with the same SQL task and harness contract.

Generated data, databases, trajectories, model weights, checkpoints, credentials, and machine-specific configuration are intentionally excluded.

## Repository map

```text
data_generation/
  data_synthesis/   executable task synthesis and verification
  dbt_sqlite/       Spider 2.0-DBT download and DuckDB-to-SQLite conversion
harness/dsh-sql/    dsh bundle, SQL tools, profile, and benchmark runner
trajectories/       multi-seed dsh-sql rollout and SFT conversion
training/sft/       packed/split dataset assembly and supervised training
training/rl/        Slime rollout adapter, execution reward, and data preparation
configs/            portable training configuration
scripts/            local dsh-sql profile setup
```

## Install

Python 3.10+ and Node.js 22+ are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

cd harness/dsh-sql
npm ci
python ../../scripts/setup_dsh_profile.py
cd ../..
```

Install the optional training dependencies with `pip install -e '.[sft]'`. RL additionally requires a compatible Slime checkout and its Megatron/SGLang runtime; it is not vendored here.

## 1. Build source databases and tasks

Download the public Spider 2.0-DBT task payloads and convert their DuckDB databases:

```bash
python -m dbt_sqlite.download_harbor_tasks --output-dir artifacts/dbt/tasks
python -m dbt_sqlite.export_sqlite \
  --data-root artifacts/dbt/tasks \
  --output-root artifacts/dbt/sqlite/databases
```

Build catalogs and validate generated blueprints for either database pool:

```bash
python -m data_synthesis.pipeline catalog \
  --database-root /path/to/sqlite/databases \
  --metadata-root /path/to/sqlite/metadata \
  --catalog-dir artifacts/catalogs

export OPENAI_API_KEY=...
python -m data_synthesis.question_generation \
  --database-root /path/to/sqlite/databases \
  --catalog-dir artifacts/catalogs \
  --output-dir artifacts/blueprints \
  --backend openai-compatible --base-url https://your-endpoint.example/v1 --model your-model

python -m data_synthesis.pipeline pilot \
  --blueprints-file artifacts/blueprints/blueprints.jsonl \
  --database-root /path/to/sqlite/databases \
  --catalog-dir artifacts/catalogs \
  --output-dir artifacts/validated
```

API credentials are read only from environment variables. Local vLLM uses the same OpenAI-compatible backend with a loopback base URL.

## 2. Collect dsh-sql trajectories

Start vLLM, then run three seeds per synthesized task:

```bash
MODEL_PATH=/path/to/model VLLM_PORT=8000 \
  bash harness/dsh-sql/runner/serve_vllm.sh

python trajectories/dsh_synth1800_pipeline.py \
  --task-file /path/to/tasks.json \
  --db-dir /path/to/sqlite/databases \
  --dsh-root harness/dsh-sql \
  --out-dir artifacts/trajectories \
  --ports 8000 --num-votes 3 --concurrency 16
```

For a hosted endpoint, set `DSH_SQL_GATEWAY_BASE_URL` and `DSH_SQL_GATEWAY_KEY`, then use the Spider runner's `--provider gateway`. The dsh bundle exposes only `sql_list_tables`, `sql_schema`, `sql_exec`, and `sql_submit`; shell, filesystem, web, subagent, and telemetry rows are disabled.

Convert completed dsh session logs to packed and per-turn SFT JSON:

```bash
python trajectories/build_sft_from_dsh.py \
  --sessions 'artifacts/trajectories/seeds/*/homes/*/sessions/*/*/session.jsonl*' \
  --output-dir artifacts/sft
```

## 3. SFT

```bash
MODEL_PATH=/path/to/base-model \
DATA_PATH=artifacts/sft/train.packed.json \
OUTPUT_DIR=artifacts/checkpoints/sft \
DEEPSPEED_CONFIG=configs/deepspeed_zero3.json \
torchrun --nproc-per-node 8 training/sft/train_sft_clean.py
```

The trainer supports `MAX_LENGTH`, `CE_CHUNK`, `USE_LIGER`, `NUM_EPOCHS`, `LR`, `GRAD_ACCUM`, and related environment variables. No model or dataset path has a machine-specific default.

## 4. RL

Prepare both task pools by setting the four `HARNESS_SQL_*` data-root variables from `.env.example`, then run:

```bash
python training/rl/scripts/replay_all_oracles.py --output-dir artifacts/oracle-replay
python training/rl/scripts/build_merged_rl_data.py --out artifacts/rl/text2sql.jsonl
```

Configure Slime with:

```text
--custom-generate-function-path text2sql.generate.generate
--dynamic-sampling-filter-path text2sql.filters.check_no_aborted
```

`text2sql.generate` launches one isolated harness subprocess per sample, preserves the sampled token IDs/logprobs through an in-process OpenAI adapter, extracts the final SQL, and computes binary execution reward against hidden result hashes. See [docs/rl.md](docs/rl.md) for the runtime contract.

## Security before publishing

```bash
git grep -nE '(/aifs4su/|/work/[^/]+|sk-[A-Za-z0-9_-]{16,})'
git status --short
```

Keep credentials in environment variables, review the staged diff, and enable GitHub secret scanning for the published repository.
