#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3

N="$SLURM_ARRAY_TASK_ID"
#N=0
TARGET_LANGS=("mt" "ar" "hau" "swa")
VALIDATION_SPLIT_PERCENTAGES=(5 1 5 5)
TARGET_LANG=${TARGET_LANGS[N]}
VALIDATION_SPLIT_PERCENTAGE=${VALIDATION_SPLIT_PERCENTAGES[N]}
SOURCE_LANG=en
SOURCE_SFT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small
TARGET_SFT=cambridgeltl/mbert-lang-sft-${TARGET_LANG}-small

OUTDIR=$HDD/experiments/distil/language-modeling/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-6-layers-384-dims-withce
#OUTDIR=$HDD/experiments/distil/language-modeling/distil/mbert-${TARGET_LANG}-tinybert-tmeval
mkdir -p $OUTDIR

python distil_mlm.py \
  --model_name_or_path bert-base-multilingual-cased \
  --source_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${SOURCE_LANG}.txt" \
  --source_sft ${SOURCE_SFT} \
  --target_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt" \
  --target_sft ${TARGET_SFT} \
  --layer_reduction_factor 2 \
  --hidden_size_reduction_factor 2 \
  --output_dir ${OUTDIR} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 256 \
  --save_steps 10000000 \
  --max_steps 200000 \
  --overwrite_output_dir \
  --learning_rate 1e-4 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --validation_split_percentage ${VALIDATION_SPLIT_PERCENTAGE} \
  --preprocessing_num_workers 15 \
  --load_best_model_at_end \
  --remove_unused_columns no \
  --save_total_limit 2 > $OUTDIR/log.txt 2>&1
  #--optim adafactor \
