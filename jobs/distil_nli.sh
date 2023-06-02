#!/bin/bash
cd "$SRC_DIR/examples/text-classification"
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
TASK_SFT=cambridgeltl/${BASE_MODEL_SHORT}-task-sft-nli

rm "$STUDENT_MODEL/trainer_state.json"

#python distil_nli.py \
#  --student_model_name_or_path $STUDENT_MODEL \
#  --student_tokenizer_name $STUDENT_MODEL_TOKENIZER \
#  --teacher_model_name_or_path $TEACHER_MODEL \
#  --lang_ft $LANG_SFT \
#  --task_ft $TASK_SFT \
#  --dataset_name multi_nli \
#  --dataset_config_name $SOURCE_LANG \
#  --output_dir $DIR \
#  --do_train \
#  --do_eval \
#  --per_device_train_batch_size $BATCH_SIZE \
#  --per_device_eval_batch_size $BATCH_SIZE \
#  --overwrite_output_dir \
#  --full_ft_max_epochs_per_iteration $FULL_FT_EPOCHS \
#  --sparse_ft_max_epochs_per_iteration $SPARSE_FT_EPOCHS \
#  --save_steps 1000000 \
#  --ft_params_proportion $SFT_DENSITY \
#  --evaluation_strategy steps \
#  --eval_steps $EVAL_STEPS \
#  --freeze_layer_norm \
#  --learning_rate $LEARNING_RATE \
#  --metric_for_best_model eval_accuracy \
#  --load_best_model_at_end \
#  --validation_split validation_matched \
#  --eval_split validation \
#  --label_names labels \
#  --remove_unused_columns no \
#  --save_total_limit 1
#if [[ $? != 0 ]]; then
#    exit 1
#fi

if [[ "$TARGET_LANG" == "el" ]]; then
    python run_nli.py \
      --do_eval \
      --model_name_or_path $STUDENT_MODEL \
      --tokenizer_name $STUDENT_MODEL_TOKENIZER \
      --dataset_name xnli \
      --dataset_config_name $TARGET_LANG \
      --task_ft $DIR \
      --output_dir $DIR/results/${TARGET_LANG} \
      --per_device_eval_batch_size $BATCH_SIZE
else
    python run_nli.py \
      --do_eval \
      --model_name_or_path $STUDENT_MODEL \
      --tokenizer_name $STUDENT_MODEL_TOKENIZER \
      --validation_file data/AmericasNLI/test/${TARGET_LANG}.tsv \
      --label_file data/labels.txt \
      --task_ft $DIR \
      --output_dir $DIR/results/${TARGET_LANG} \
      --per_device_eval_batch_size $BATCH_SIZE
fi
