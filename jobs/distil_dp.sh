#!/bin/bash
cd "$SRC_DIR/examples/dependency-parsing"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

STUDENT_MODEL="$HDD/experiments/distil/run1/bilingual_distil/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-${HIDDEN_DIM_REDUCTION_FACTOR}_trim-${TRIM_THRESH_SHORT}_steps-${PRETRAINING_MAX_STEPS}"
STUDENT_MODEL_TOKENIZER="$STUDENT_MODEL/trimmed-vocab"

TEACHER_MODEL="$BASE_MODEL"
LANG_SFT=cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${SOURCE_LANG}-small
#if [[ "$MULTISOURCE" == "yes" ]]; then
#    TASK_SFT=cambridgeltl/${BASE_MODEL_SHORT}-task-sft-dp
#else
TASK_SFT=cambridgeltl/${BASE_MODEL_SHORT}-task-sft-dp
#fi

rm "$STUDENT_MODEL/trainer_state.json"

python distil_dp.py \
  --student_model_name_or_path $STUDENT_MODEL \
  --student_tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --teacher_model_name_or_path $TEACHER_MODEL \
  --lang_ft $LANG_SFT \
  --task_ft $TASK_SFT \
  --dataset_name universal_dependencies \
  --dataset_config_name $SOURCE_TREEBANK \
  --output_dir $DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ_LENGTH \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration $FULL_FT_EPOCHS \
  --sparse_ft_max_epochs_per_iteration $SPARSE_FT_EPOCHS \
  --ft_params_proportion $SFT_DENSITY \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --learning_rate $LEARNING_RATE \
  --metric_for_best_model eval_las \
  --load_best_model_at_end \
  --eval_split validation \
  --remove_unused_columns no \
  --save_total_limit 1
if [[ $? != 0 ]]; then
    exit 1
fi

python run_dp.py \
  --do_eval \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --task_ft $DIR \
  --dataset_name universal_dependencies \
  --dataset_config_name $TARGET_TREEBANK \
  --output_dir $DIR/results/$TARGET_LANG \
  --per_device_eval_batch_size $BATCH_SIZE \
  --eval_split test
