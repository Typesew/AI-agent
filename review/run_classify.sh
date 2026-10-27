#!/usr/bin/env bash
# Этап A: классификация пунктов на выборке из 4 продуктов, затем обе метрики.
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://$(ip route show default | awk '{print $3}'):11434
export PYTHONUNBUFFERED=1
rm -f review/run_classify.done
start=$(date +%s)
.venv/bin/python -m indexing.extract_llm --mode classify --force \
  --only balt_credit sever_install opora_debit kedr_student > review/extract_classify.txt 2>&1
echo "всего секунд: $(( $(date +%s) - start ))" >> review/extract_classify.txt
.venv/bin/python -m eval.eval_extraction --pred data/extracted/llm_classify > review/eval_extraction_classify.txt 2>&1
echo done > review/run_classify.done
