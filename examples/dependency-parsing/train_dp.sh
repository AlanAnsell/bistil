#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-2
#TARGET_LANG=$1
SOURCE_LANG=en
TREEBANK=en_ewt
#LANG_FT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small

#N="$SLURM_ARRAY_TASK_ID"
N=0
#PROPS=("0.2" "0.225")
LANGS=("mt" "ar")
TARGET_LANG=${LANGS[N]}
#PROP=${PROPS[N]}
#PROP=0.225

BASE_MODEL=$HDD/experiments/distil/language-modeling/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-384-dims
OUTPUT_DIR=$HDD/experiments/distil/dependency-parsing/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-384-dims-8p
#BASE_MODEL=bert-base-multilingual-cased
#OUTPUT_DIR=models/prune-all-test-${TARGET_LANG}
mkdir -p $OUTPUT_DIR

rm $BASE_MODEL/trainer_state.json

python run_dp.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name $TREEBANK \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 10 \
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
  --save_total_limit 2 #> $OUTPUT_DIR/log.txt 2>&1
  #--lang_ft $LANG_FT \
  #--eval_steps 250 \
  #--full_l1_reg 1.0 \
  #--sparse_l1_reg 0.0 \
  #--load_weights $HDD/experiments/distil/language-modeling/mbert-${TARGET_LANG}-en-hc-test/pytorch_model.bin \
  #--relufy yes \
  #--hc_lambda 1 \
