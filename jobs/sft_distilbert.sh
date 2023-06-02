#!/bin/bash
cd "$SRC_DIR/examples/language-modeling"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

python run_mlm.py \
  --model_name_or_path $BASE_MODEL \
  --train_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${LANG}.txt" \
  --output_dir $DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --preprocessing_num_workers 5 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --save_steps 10000000 \
  --overwrite_output_dir \
  --ft_params_proportion $SFT_DENSITY \
  --freeze_layer_norm \
  --freeze_decoder \
  --full_l1_reg $L1_REG \
  --sparse_l1_reg $L1_REG \
  --learning_rate $LEARNING_RATE \
  --n_ft_iterations 1 \
  --full_ft_min_steps_per_iteration $FULL_FT_MIN_STEPS \
  --sparse_ft_min_steps_per_iteration $SPARSE_FT_MIN_STEPS \
  --full_ft_max_steps_per_iteration $FULL_FT_MAX_STEPS \
  --sparse_ft_max_steps_per_iteration $SPARSE_FT_MAX_STEPS \
  --full_ft_max_epochs_per_iteration $FULL_FT_MAX_EPOCHS \
  --sparse_ft_max_epochs_per_iteration $FULL_FT_MAX_EPOCHS \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  --validation_split_percentage $VALIDATION_SPLIT_PERCENTAGE \
  --load_best_model_at_end \
  --save_total_limit 1
