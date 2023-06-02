#!/bin/bash
cd "$SRC_DIR/examples/question-answering"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

STUDENT_MODEL="$HDD/experiments/distil/run1/bilingual_distil/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-${HIDDEN_DIM_REDUCTION_FACTOR}_trim-${TRIM_THRESH_SHORT}_steps-${PRETRAINING_MAX_STEPS}"
STUDENT_MODEL_TOKENIZER="$STUDENT_MODEL/trimmed-vocab"

rm "$STUDENT_MODEL/trainer_state.json"

python run_qa.py \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --dataset_name squad \
  --output_dir $DIR \
  --do_train \
  --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration $FULL_FT_EPOCHS \
  --sparse_ft_max_epochs_per_iteration $SPARSE_FT_EPOCHS \
  --ft_params_proportion $SFT_DENSITY \
  --save_steps 10000000 \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  --learning_rate $LEARNING_RATE \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 1
if [[ $? != 0 ]]; then
    exit 1
fi
  #--eval_accumulation_steps 15 \
  #--full_ft_max_steps_per_iteration 100 \
  #--sparse_ft_max_steps_per_iteration 100 \

python run_qa.py \
  --do_eval \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --task_ft $DIR \
  --dataset_name xquad \
  --dataset_config_name "xquad.${TARGET_LANG}" \
  --output_dir $DIR/results/$TARGET_LANG \
  --per_device_eval_batch_size $BATCH_SIZE \
  --eval_split validation
