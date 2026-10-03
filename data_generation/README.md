# Data generation

The pipeline separates untrusted model proposals from deterministic validation.

1. `dbt_sqlite.download_harbor_tasks` downloads the public Spider 2.0-DBT task payloads.
2. `dbt_sqlite.export_sqlite` materializes DuckDB tables and views into SQLite, maps types explicitly, preserves available dbt documentation, and records conversion provenance.
3. `data_synthesis.pipeline catalog` profiles live SQLite databases and constructs an offline schema/value/join catalog.
4. `data_synthesis.question_generation` asks an OpenAI-compatible API, Anthropic-compatible API, local vLLM endpoint, or local Transformers model for SQL-first task blueprints.
5. `data_synthesis.pipeline pilot` executes the proposed oracle SQL and mutation checks. Only executable, differentiating tasks are emitted.
6. `data_synthesis.trajectory` can collect a lightweight five-tool trajectory. For the training runs in this repository, use the `dsh-sql` collectors under `trajectories/` so inference and RL share the same harness contract.

Oracle SQL and result hashes remain verifier-side and are never added to the model prompt. All database, metadata, catalog, output, endpoint, and credential locations are passed through command-line options or environment variables.
