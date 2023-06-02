#!/bin/bash
cd "$SRC_DIR/examples/dependency-parsing"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

if [[ "$MODEL_TYPE" == "sftdistilbert" ]]; then
    LANG_SFT="--lang_ft $HDD/experiments/distil/run1/sft_distilbert/${BASE_MODEL_SHORT}_${TARGET_LANG}_${LANG_SFT_DENSITY}_steps-${LANG_SFT_STEPS}_${LANG_SFT_REG}"
    TASK_SFT="--task_ft $HDD/experiments/distil/run1/distilbert_dp/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${LANG_SFT_DENSITY}_steps-${LANG_SFT_STEPS}_${LANG_SFT_REG}_${SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"
    TOKENIZER_NAME=$BASE_MODEL
elif [[ "$MODEL_TYPE" == "mmt" ]]; then
    LANG_SFT="--lang_ft cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${TARGET_LANG}-small"
    TASK_SFT="--task_ft cambridgeltl/${BASE_MODEL_SHORT}-task-sft-dp"
    TOKENIZER_NAME=$BASE_MODEL
elif [[ "$MODEL_TYPE" == "bidistil" ]]; then
    LANG_SFT=""
    BASE_MODEL="$HDD/experiments/distil/run1/bilingual_distil/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-1_trim-1e6_steps-${DISTILLATION_STEPS}"
    TOKENIZER_NAME="$BASE_MODEL/trimmed-vocab"
    TASK_SFT="--task_ft $HDD/experiments/distil/run1/distil_dp/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-1_trim-1e6_steps-${DISTILLATION_STEPS}_${SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"
else
    echo "Unrecognised model type $MODEL_TYPE" >&2
    exit 1
fi

if [[ "$DEVICE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

OUTPUT_DIR="$DIR/results/$TARGET_LANG"

python run_dp.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name $TOKENIZER_NAME \
  $LANG_SFT \
  $TASK_SFT \
  --count_flops ${COUNT_FLOPS} \
  --num_threads 1 \
  --replicate_eval_dataset $REPLICATE_EVAL_DATASET \
  --dataset_name universal_dependencies \
  --dataset_config_name $TARGET_TREEBANK \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ_LENGTH \
  --eval_split test
