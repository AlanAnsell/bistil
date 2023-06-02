#!/bin/bash
cd "$SRC_DIR/examples/token-classification"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

LANG_SFT="$HDD/experiments/distil/run1/sft_distilbert/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${LANG_SFT_DENSITY}_steps-${LANG_SFT_STEPS}_${LANG_SFT_REG}"

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --lang_ft $LANG_SFT \
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
