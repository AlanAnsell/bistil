#!/bin/bash
cd "$SRC_DIR/examples/text-classification"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

LANG_SFT=cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${TARGET_LANG}-small
TASK_SFT=cambridgeltl/${BASE_MODEL_SHORT}-task-sft-nli
OUTPUT_DIR="$DIR/results/$TARGET_LANG"

python run_nli.py \
  --model_name_or_path $BASE_MODEL \
  --lang_ft $LANG_SFT \
  --task_ft $TASK_SFT \
  --validation_file data/AmericasNLI/test/${TARGET_LANG}.tsv \
  --label_file data/labels.txt \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --per_device_eval_batch_size $BATCH_SIZE
