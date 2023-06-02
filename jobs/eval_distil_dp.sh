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
TASK_SFT="$HDD/experiments/distil/run1/distil_dp/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-${HIDDEN_DIM_REDUCTION_FACTOR}_trim-${TRIM_THRESH_SHORT}_steps-${PRETRAINING_MAX_STEPS}_${SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"
rm "$STUDENT_MODEL/trainer_state.json"

python run_dp.py \
  --do_eval \
  --model_name_or_path $STUDENT_MODEL \
  --tokenizer_name $STUDENT_MODEL_TOKENIZER \
  --task_ft $TASK_SFT \
  --dataset_name universal_dependencies \
  --dataset_config_name $SOURCE_TREEBANK \
  --output_dir $TASK_SFT/source-results/$TARGET_LANG \
  --per_device_eval_batch_size $BATCH_SIZE \
  --eval_split test
