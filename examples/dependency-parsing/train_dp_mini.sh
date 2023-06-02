#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3
#LANG=$1

N="$SLURM_ARRAY_TASK_ID"
LANGS=("mt" "ar" "swa" "hau")

SOURCE_LANG=en
TARGET_LANG=${LANGS[N]}
TREEBANK=en_ewt

BASE_MODEL=$HDD/experiments/distil/language-modeling/mini-bilingual/${SOURCE_LANG}-${TARGET_LANG}
OUTPUT_DIR=$HDD/experiments/distil/dependency-parsing/mini-bilingual/${SOURCE_LANG}-${TARGET_LANG}-8p
mkdir -p $OUTPUT_DIR

rm $BASE_MODEL/trainer_state.json

python run_dp.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name $BASE_MODEL/tokenizer \
  --dataset_name universal_dependencies \
  --dataset_config_name $TREEBANK \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 3 \
  --sparse_ft_max_epochs_per_iteration 10 \
  --save_steps 1000000 \
  --ft_params_proportion 0.08 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_las \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 > $OUTPUT_DIR/log.txt 2>&1
  #--lang_ft $LANG_FT \
  #--eval_steps 250 \
  #--full_l1_reg 1.0 \
  #--sparse_l1_reg 0.0 \
