#!/bin/bash
LANG=$1 # Buryat
TREEBANK=$2
PROP=$3

BASE_MODEL=$HDD/experiments/distil/language-modeling/mbert-${LANG}-en-${PROP}
TASK_FT=$HDD/experiments/distil/pos_tagging/mbert-${LANG}-en-${PROP}-8p

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --task_ft $TASK_FT \
  --tokenizer_name bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name "${LANG}_${TREEBANK}" \
  --output_dir results/pos-tagging/${LANG} \
  --do_eval \
  --label_column_name upos \
  --per_device_eval_batch_size 8 \
  --task_name pos \
  --overwrite_output_dir \
  --eval_split test
  #--tokenizer_name xlm-roberta-base \
