#!/bin/bash
cd "$SRC_DIR/examples/question-answering"
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment $CONDA_ENV" >&2
    exit 1
fi

printenv

#LANG_SFT=cambridgeltl/${BASE_MODEL_SHORT}-lang-sft-en-small

python run_qa.py \
  --model_name_or_path $BASE_MODEL \
  --dataset_name squad \
  --output_dir $DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --save_steps 10000000 \
  --overwrite_output_dir \
  --ft_params_proportion $SFT_DENSITY \
  --freeze_layer_norm \
  --learning_rate $LEARNING_RATE \
  --full_ft_max_epochs_per_iteration $FULL_FT_MAX_EPOCHS \
  --sparse_ft_max_epochs_per_iteration $FULL_FT_MAX_EPOCHS \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 1
  #--eval_steps 100 \
  #--lang_ft $LANG_SFT \
