#!/usr/bin/env bash
#SBATCH --job-name=harnesssql-rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=5-00:00:00

# Slurm wrapper for the portable production training entrypoint. Configure paths
# and optional hyperparameters with --export=ALL,... or a sourced environment
# file; no cluster-specific path is embedded here.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec bash "$repo_root/training/rl/launch_ray.sh"
