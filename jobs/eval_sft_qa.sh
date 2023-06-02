#!/bin/bash
cd "$SRC_DIR/examples/question-answering"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

LANG_SFT=cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${TARGET_LANG}-small
TASK_SFT="$HDD/experiments/distil/run1/sft_qa/${BASE_MODEL_SHORT}_${LANG_SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}"
OUTPUT_DIR="${TASK_SFT}/results/${TARGET_LANG}"
mkdir -p "$OUTPUT_DIR"

python run_qa.py \
  --model_name_or_path $BASE_MODEL \
  --task_ft $TASK_SFT \
  --dataset_name xquad \
  --dataset_config_name "xquad.${TARGET_LANG}" \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --per_device_eval_batch_size $BATCH_SIZE \
  --eval_split validation
  #--lang_ft $LANG_SFT \
