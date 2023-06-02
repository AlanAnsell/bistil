#!/bin/bash
LANG=$1 # Buryat
TREEBANK=$2
#LANG_FT=../language-modeling/models/$LANG/pytorch_model.bin
BASE_MODEL=bert-base-multilingual-cased
TASK_FT=cambridgeltl/mbert-task-sft-pos  # Single-source task SFT

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --task_ft $TASK_FT \
  --dataset_name universal_dependencies \
  --dataset_config_name "${LANG}_${TREEBANK}" \
  --output_dir results/pos-tagging/${LANG} \
  --do_eval \
  --label_column_name upos \
  --per_device_eval_batch_size 8 \
  --task_name pos \
  --overwrite_output_dir \
  --eval_split test
  #--lang_ft $LANG_FT \
