#!/usr/bin/env bash
# Итоговые прогоны на коде из коммита (см. review/final_commit.txt), строго по одному процессу.
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://$(ip route show default | awk '{print $3}'):11434
export PYTHONUNBUFFERED=1
rm -f review/run_final.done
git rev-parse --short HEAD > review/final_commit.txt
s=$(date +%s)
.venv/bin/python -m eval.run_eval --llm > review/final_eval_llm.txt 2>&1
cp eval/results_llm.json review/results_final_llm.json
echo "секунд: $(( $(date +%s) - s ))" >> review/final_eval_llm.txt
s=$(date +%s)
.venv/bin/python -m eval.run_eval --llm --no-graph --only 4 5 9 > review/final_eval_llm_nograph_459.txt 2>&1
cp eval/results_llm_nograph_only_4_5_9.json review/results_final_llm_nograph_459.json
echo "секунд: $(( $(date +%s) - s ))" >> review/final_eval_llm_nograph_459.txt
s=$(date +%s)
.venv/bin/python -m indexing.extract_llm --mode classify --reuse-raw --force > review/final_extract_classify_all.txt 2>&1
echo "секунд: $(( $(date +%s) - s ))" >> review/final_extract_classify_all.txt
.venv/bin/python -m eval.eval_extraction --pred data/extracted/llm_classify > review/final_eval_extraction_classify_all.txt 2>&1
echo done > review/run_final.done
