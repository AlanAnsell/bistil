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

python bilingual_mlm.py \
  --config_name "$VOCAB_DIR" \
  --tokenizer_name "$VOCAB_DIR" \
  --source_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${SOURCE_LANG}.txt" \
  --target_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt" \
  --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
  --do_mlm $DO_MLM \
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
  --save_total_limit 2
