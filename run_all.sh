#!/usr/bin/env bash
# Full experiment: Qwen 0.5B on every task, then Qwen 1.5B on a seeded subset.
# Resumable: rerun after an interruption and finished rows are skipped.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
python -u -m stale_rag.run --reader hf --model Qwen/Qwen2.5-0.5B-Instruct --threads 6
python -u -m stale_rag.run --reader hf --model Qwen/Qwen2.5-1.5B-Instruct --threads 6 --limit 300
echo "ALL DONE"
