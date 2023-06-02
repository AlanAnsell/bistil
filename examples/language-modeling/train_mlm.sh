#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
##SBATCH --array=0-1
#LANG=$1

#N="$SLURM_ARRAY_TASK_ID"
N=0
#PROPS=("0.15" "0.175")
LANGS=("mt" "ar")

#PROP=${PROPS[N]}
#PROP=0.225
LANG=${LANGS[N]}
#LANG=mt

OUTDIR=$HDD/experiments/distil/language-modeling/mbert-${LANG}-en-prune-3phase-test
mkdir -p $OUTDIR

python run_mlm.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/en.txt,$HOME/projects/composable-sft/examples/language-modeling/corpora/${LANG}.txt" \
  --output_dir $OUTDIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --full_l1_reg 10.0 \
  --sparse_l1_reg 0.0 \
  --max_seq_length 256 \
  --save_steps 10000000 \
  --overwrite_output_dir \
  --ft_params_proportion 0.5 \
  --freeze_decoder \
  --learning_rate 5e-5 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --hc_lambda 1 \
  --full_ft_max_steps_per_iteration 100000 \
  --sparse_ft_max_steps_per_iteration 50000 \
  --validation_split_percentage 5 \
  --load_best_model_at_end \
  --save_total_limit 2 > $OUTDIR/log.txt 2>&1
  #--optim adam \
  #--load_best_model_at_end \
  #--full_ft_min_steps_per_iteration 1000 \
  #--sparse_ft_min_steps_per_iteration 1000 \
  #--freeze_layer_norm \
  #--evaluation_strategy steps \
  #--eval_steps 5000 \
  #--full_ft_max_epochs_per_iteration 100 \
  #--sparse_ft_max_epochs_per_iteration 100 \
  #--relufy yes \
