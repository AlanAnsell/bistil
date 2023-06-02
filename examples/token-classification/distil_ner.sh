#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

#N="$SLURM_ARRAY_TASK_ID"
N=0
LANGS=("ibo" "kin" "kin" "lug")
TARGET_LANG=${LANGS[N]}
#TARGET_LANG=$1
SOURCE_LANG=en
#LANG_FT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small

STUDENT_MODEL=$HDD/experiments/distil/language-modeling/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-trimmed-vocab-density-0.5
STUDENT_MODEL_TOKENIZER=$HDD/experiments/distil/language-modeling/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-trimmed-vocab
TEACHER_MODEL=bert-base-multilingual-cased
LANG_SFT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small
TASK_SFT=cambridgeltl/mbert-task-sft-masakhaner
OUTPUT_DIR=$HDD/experiments/distil/ner/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-trimmed-vocab-density-0.5-distilled
mkdir -p $OUTPUT_DIR

rm $STUDENT_MODEL/trainer_state.json

python distil_token_classification.py \
  --student_model_name_or_path $STUDENT_MODEL \
  --student_tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --teacher_model_name_or_path $TEACHER_MODEL \
  --lang_ft $LANG_SFT \
  --task_ft $TASK_SFT \
  --dataset_name ner_dataset.py \
  --dataset_config_name $SOURCE_LANG \
  --label_names labels \
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
  --remove_unused_columns no \
  --save_total_limit 2 #> $OUTPUT_DIR/log.txt 2>&1
  #--lang_ft $LANG_FT \
