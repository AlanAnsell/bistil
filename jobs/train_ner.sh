#!/bin/bash
cd "$SRC_DIR/examples/token-classification"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

STUDENT_MODEL="$HDD/experiments/distil/run1/bilingual_distil/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-${HIDDEN_DIM_REDUCTION_FACTOR}_trim-${TRIM_THRESH_SHORT}_steps-${PRETRAINING_MAX_STEPS}"
STUDENT_MODEL_TOKENIZER="$STUDENT_MODEL/trimmed-vocab"
rm "$STUDENT_MODEL/trainer_state.json"

python run_token_classification.py \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --dataset_name ner_dataset.py \
  --dataset_config_name $SOURCE_LANG \
  --label_names labels \
  --label_column_name ner_tags \
  --task_name ner \
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
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 1
if [[ $? != 0 ]]; then
    exit 1
fi

python run_token_classification.py \
  --do_eval \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --task_ft $DIR \
  --dataset_name ner_dataset.py \
  --dataset_config_name $TARGET_LANG \
  --output_dir $DIR/results/$TARGET_LANG \
  --per_device_eval_batch_size $BATCH_SIZE \
  --eval_split test
