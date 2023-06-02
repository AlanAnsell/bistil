#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3

N="$SLURM_ARRAY_TASK_ID"
#N=0
LANGS=("bm" "bxr" "gv" "hsb")
TARGET_LANG=${LANGS[N]}

SOURCE_LANG=en
TREEBANK=en_ewt

PROP=0.225

BASE_MODEL=$HDD/experiments/distil/language-modeling/mbert-${TARGET_LANG}-${SOURCE_LANG}-${PROP}
OUTPUT_DIR=$HDD/experiments/distil/pos_tagging/mbert-${TARGET_LANG}-${SOURCE_LANG}-${PROP}-8p
mkdir -p $OUTPUT_DIR

rm $BASE_MODEL/trainer_state.json

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name $TREEBANK \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --label_column_name upos \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --eval_accumulation_steps 20 \
  --task_name pos \
  --max_seq_length 256 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 3 \
  --sparse_ft_max_epochs_per_iteration 10 \
  --ft_params_proportion 0.08 \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --eval_split validation \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --save_total_limit 2 > $OUTPUT_DIR/log.txt 2>&1
  #--optim adafactor 
  #--lang_ft $LANG_FT \
  #--sparse_l1_reg 0.0 \
  #--tokenizer_name xlm-roberta-base \
  #--save_steps 1000000 \
