#!/bin/bash
cd "$SRC_DIR/examples/question-answering"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

if [[ "$MODEL_TYPE" == "sftmmt" ]]; then
    TASK_SFT="--task_ft $HDD/experiments/distil/run1/sft_qa/${BASE_MODEL_SHORT}_${SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}"
    TOKENIZER_NAME=$BASE_MODEL
elif [[ "$MODEL_TYPE" == "bidistil" ]]; then
    BASE_MODEL="$HDD/experiments/distil/run1/bilingual_distil/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-1_trim-1e6_steps-${DISTILLATION_STEPS}"
    TOKENIZER_NAME="$BASE_MODEL/trimmed-vocab"
    TASK_SFT="--task_ft $HDD/experiments/distil/run1/distil_qa/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_lrf-${LAYER_REDUCTION_FACTOR}_drf-1_trim-1e6_steps-${DISTILLATION_STEPS}_${SFT_DENSITY}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"
else
    echo "Unrecognised model type $MODEL_TYPE" >&2
    exit 1
fi

if [[ "$DEVICE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

OUTPUT_DIR="$DIR/results/$TARGET_LANG"
mkdir -p $OUTPUT_DIR

python run_qa.py \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name $TOKENIZER_NAME \
  $TASK_SFT \
  --count_flops ${COUNT_FLOPS} \
  --compute_metrics no \
  --num_threads 1 \
  --replicate_eval_dataset $REPLICATE_EVAL_DATASET \
  --dataset_name xquad \
  --dataset_config_name "xquad.${TARGET_LANG}" \
  --output_dir $OUTPUT_DIR \
  --do_eval \
  --per_device_eval_batch_size $BATCH_SIZE \
  --eval_split validation
