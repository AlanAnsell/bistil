#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3

#N="$SLURM_ARRAY_TASK_ID"
#LANGS=("hau" "ibo" "kin" "lug")
#TARGET_LANG=${LANGS[N]}
TARGET_LANG=$1
SOURCE_LANG=en
#LANG_FT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small

PROP=0.225

BASE_MODEL=$HDD/experiments/distil/language-modeling/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-tinybert-tmeval
OUTPUT_DIR=$HDD/experiments/distil/ner/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-tinybert-tmeval-8p
mkdir -p $OUTPUT_DIR

rm $BASE_MODEL/trainer_state.json

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name bert-base-multilingual-cased \
  --dataset_name ner_dataset.py \
  --dataset_config_name $SOURCE_LANG \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --task_name ner \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 3 \
  --sparse_ft_max_epochs_per_iteration 10 \
  --save_steps 1000000 \
  --ft_params_proportion 0.08 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 #> $OUTPUT_DIR/log.txt 2>&1
  #--lang_ft $LANG_FT \
