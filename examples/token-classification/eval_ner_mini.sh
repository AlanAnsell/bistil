#!/bin/bash
LANG=$1 # Hausa
#LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small
#TASK_FT=cambridgeltl/mbert-task-sft-masakhaner
BASE_MODEL=$HDD/experiments/distil/language-modeling/mini-bilingual/en-${LANG}
TASK_FT=$HDD/experiments/distil/ner/mini-bilingual/en-${LANG}-8p
#TASK_FT=models/ner/en

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name $BASE_MODEL/tokenizer \
  --dataset_name ner_dataset.py \
  --dataset_config_name $LANG \
  --output_dir results/ner/${LANG} \
  --task_ft $TASK_FT \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_eval_batch_size 8 \
  --task_name ner \
  --overwrite_output_dir \
  --eval_split test
  #--lang_ft $LANG_FT \
