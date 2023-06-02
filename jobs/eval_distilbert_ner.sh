#!/bin/bash
cd "$SRC_DIR/examples/token-classification"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

LANG_SFT="$HDD/experiments/distil/run1/sft_distilbert/${BASE_MODEL_SHORT}_${TARGET_LANG}_${LANG_SFT_DENSITY}_steps-${LANG_SFT_STEPS}_${LANG_SFT_REG}"
TASK_SFT="$HDD/experiments/distil/run1/distilbert_ner/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${LANG_SFT_DENSITY}_steps-${LANG_SFT_STEPS}_${LANG_SFT_REG}_${SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"
OUTPUT_DIR="${TASK_SFT}/results/${TARGET_LANG}"
mkdir -p "$OUTPUT_DIR"

python run_token_classification.py \
  --model_name_or_path $BASE_MODEL \
  --lang_ft $LANG_SFT \
  --task_ft $TASK_SFT \
  --dataset_name ner_dataset.py \
  --dataset_config_name $TARGET_LANG \
  --label_names labels \
  --label_column_name ner_tags \
  --task_name ner \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ_LENGTH \
  --eval_split test
