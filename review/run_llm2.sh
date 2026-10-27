#!/usr/bin/env bash
# Выборка для F1 экстракции (по одному продукту каждого типа) и повтор вопросов 6, 13, 15.
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://$(ip route show default | awk '{print $3}'):11434
export PYTHONUNBUFFERED=1
rm -f data/extracted/llm/*.json review/llm_runs2.done
for doc in balt_credit sever_install opora_debit kedr_student; do
  .venv/bin/python -m indexing.extract_llm --only "$doc" >> review/extract_llm.txt 2>&1
done
.venv/bin/python -m eval.eval_extraction > review/eval_extraction.txt 2>&1
.venv/bin/python -m eval.run_eval --llm --only 6 13 15 > review/eval_llm_rerun.txt 2>&1
cp eval/results_llm.json review/results_llm_rerun.json
echo done > review/llm_runs2.done
