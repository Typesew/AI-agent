#!/usr/bin/env bash
# Этап A, итерация 2: промпт с подсказками (хеш шаблона в имени кэша), выборка из 4 продуктов.
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://$(ip route show default | awk '{print $3}'):11434
export PYTHONUNBUFFERED=1
rm -f review/run_classify_v2.done
D="balt_credit sever_install opora_debit kedr_student"
start=$(date +%s)
.venv/bin/python -m indexing.extract_llm --mode classify --force --out data/extracted/llm_classify_v2 --only $D > review/extract_classify_v2.txt 2>&1
echo "всего секунд: $(( $(date +%s) - start ))" >> review/extract_classify_v2.txt
.venv/bin/python -m eval.eval_extraction --pred data/extracted/llm_classify_v2 > review/eval_extraction_classify_v2_rules.txt 2>&1
.venv/bin/python -m indexing.extract_llm --mode classify --reuse-raw --force --no-rules --out data/extracted/llm_classify_v2_norules --only $D > /dev/null 2>&1
.venv/bin/python -m eval.eval_extraction --pred data/extracted/llm_classify_v2_norules > review/eval_extraction_classify_v2_norules.txt 2>&1
echo done > review/run_classify_v2.done
