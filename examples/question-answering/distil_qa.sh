#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3

N="$SLURM_ARRAY_TASK_ID"
#N=0
TARGET_LANGS=("ar" "de" "el" "es")
SOURCE_LANG=en
TARGET_LANG=${TARGET_LANGS[N]}
BASE_MODEL_SHORT=mdeberta
LAYER_REDUCTION_FACTOR=1
HIDDEN_DIM_REDUCTION_FACTOR=2

STUDENT_MODEL="../language-modeling/model/${BASE_MODEL_SHORT}-${SOURCE_LANG}-${TARGET_LANG}-state-projection-test"
STUDENT_MODEL_TOKENIZER="$HDD/experiments/distil/run1/bilingual_distil/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-2_drf-1_trim-1e6_steps-200000/trimmed-vocab"
rm "$STUDENT_MODEL/trainer_state.json"
OUTDIR=models/${BASE_MODEL_SHORT}-${SOURCE_LANG}-${TARGET_LANG}-state-projection-test
mkdir -p $OUTDIR

python run_qa.py \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --dataset_name squad \
  --output_dir $OUTDIR \
  --do_train \
  --do_eval \
  --max_seq_length 384 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --ft_params_proportion 0.08 \
  --save_steps 10000000 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 1
if [[ $? != 0 ]]; then
    exit 1
fi
  #--lang_ft $LANG_SFT \
  #--eval_accumulation_steps 15 \
  #--full_ft_max_steps_per_iteration 100 \
  #--sparse_ft_max_steps_per_iteration 100 \

python run_qa.py \
  --do_eval \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --task_ft $OUTDIR \
  --dataset_name xquad \
  --dataset_config_name "xquad.${TARGET_LANG}" \
  --output_dir $OUTDIR/results/$TARGET_LANG \
  --per_device_eval_batch_size 8 \
  --eval_split validation
