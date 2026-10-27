#!/usr/bin/env bash
# Прогоны с Qwen3 по ревью: ~75 мин на каждый eval, экстракция дольше.
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://$(ip route show default | awk '{print $3}'):11434
export PYTHONUNBUFFERED=1
.venv/bin/python -m eval.run_eval --llm            > review/eval_llm.txt 2>&1
.venv/bin/python -m eval.run_eval --llm --no-graph > review/eval_llm_nograph.txt 2>&1
.venv/bin/python -m indexing.extract_llm           > review/extract_llm.txt 2>&1
.venv/bin/python -m eval.eval_extraction           > review/eval_extraction.txt 2>&1
echo done > review/llm_runs.done
