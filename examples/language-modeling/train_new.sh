#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3
#LANG=$1

N="$SLURM_ARRAY_TASK_ID"
LANGS=("mt" "ar" "swa" "hau")

SOURCE_LANG=en
TARGET_LANG=${LANGS[N]}
SOURCE_CORPUS="$HOME/projects/composable-sft/examples/language-modeling/corpora/${SOURCE_LANG}.txt"
TARGET_CORPUS="$HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt"

OUTDIR=$HDD/experiments/distil/language-modeling/mini-bilingual/${SOURCE_LANG}-${TARGET_LANG}
mkdir -p $OUTDIR/tokenizer
VOCAB_FILE=$OUTDIR/tokenizer/

python learn_vocab.py \
  --source_file $SOURCE_CORPUS \
  --target_file $TARGET_CORPUS \
  --output_file $VOCAB_FILE > $OUTDIR/log.txt 2>&1

python run_mlm.py \
  --config_name mbert-small.json \
  --tokenizer_name $OUTDIR/tokenizer \
  --train_file "${SOURCE_CORPUS},${TARGET_CORPUS}" \
  --output_dir $OUTDIR \
  --sparse_ft no \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 256 \
  --preprocessing_num_workers 5 \
  --warmup_ratio 0.1 \
  --save_steps 10000000 \
  --overwrite_output_dir \
  --learning_rate 1e-4 \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --max_steps 200000 \
  --validation_split_percentage 5 \
  --load_best_model_at_end \
  --save_total_limit 2 >> $OUTDIR/log.txt 2>&1
  #--optim adafactor \
  #--load_best_model_at_end \
  #--full_ft_min_steps_per_iteration 1000 \
  #--sparse_ft_min_steps_per_iteration 1000 \
  #--freeze_layer_norm \
  #--evaluation_strategy steps \
  #--eval_steps 5000 \
  #--full_ft_max_epochs_per_iteration 100 \
  #--sparse_ft_max_epochs_per_iteration 100 \
