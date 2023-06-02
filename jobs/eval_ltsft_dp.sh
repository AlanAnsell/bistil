#!/bin/bash
cd "$SRC_DIR/examples/dependency-parsing"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

LANG_SFT=cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${TARGET_LANG}-small
TASK_SFT=cambridgeltl/${BASE_MODEL_SHORT}-task-sft-dp
OUTPUT_DIR="${DIR}/results/${TARGET_LANG}"
mkdir -p "$OUTPUT_DIR"

python run_dp.py \
  --model_name_or_path $BASE_MODEL \
  --lang_ft $LANG_SFT \
  --task_ft $TASK_SFT \
  --dataset_name universal_dependencies \
  --dataset_config_name $TARGET_TREEBANK \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ_LENGTH \
  --eval_split test
