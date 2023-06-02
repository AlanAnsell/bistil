#!/bin/bash
cd "$SRC_DIR/examples/language-modeling"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

VOCAB_DIR="$DIR/trimmed-vocab"
mkdir -p "$VOCAB_DIR"

python trim_vocab.py \
  --model_name_or_path "$BASE_MODEL" \
  --source_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${SOURCE_LANG}.txt" \
  --target_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt" \
  --probability_threshold "$TRIM_THRESH" \
  --output_dir "$VOCAB_DIR" \
  --overwrite_output_dir \
  --preprocessing_num_workers 5
if [[ $? != 0 ]]; then
    exit 1
fi
if [[ "$BASE_MODEL_SHORT" == "mbert" ]]; then
    rm "${VOCAB_DIR}/tokenizer.json"
fi

if [[ "$BASE_MODEL_SHORT" == "mdeberta" ]]; then
    SOURCE_SFT_ARG=""
    TARGET_SFT_ARG=""
else
    SOURCE_SFT_ARG="--source_sft cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${SOURCE_LANG}-small"
    TARGET_SFT_ARG="--target_sft cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-${TARGET_LANG}-small"
fi

python distil_mlm.py \
  --model_name_or_path "$VOCAB_DIR" \
  --sft_tokenizer "$BASE_MODEL" \
  --source_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${SOURCE_LANG}.txt" \
  $SOURCE_SFT_ARG \
  --target_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt" \
  $TARGET_SFT_ARG \
  --layer_reduction_factor "$LAYER_REDUCTION_FACTOR" \
  --hidden_size_reduction_factor "$HIDDEN_DIM_REDUCTION_FACTOR" \
  --freeze_embeddings "$FREEZE_EMBEDDINGS" \
  --output_dir "$DIR" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_steps "$MAX_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_steps 1000000 \
  --validation_split_percentage "$VALIDATION_SPLIT_PERCENTAGE" \
  --preprocessing_num_workers 5 \
  --load_best_model_at_end \
  --remove_unused_columns no \
  --save_total_limit 2
  #--do_mlm $DO_MLM \
