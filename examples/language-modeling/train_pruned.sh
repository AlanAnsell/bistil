#!/bin/bash

OUTDIR=models/en-pruned-0.5-continued
mkdir -p $OUTDIR

python run_mlm.py \
  --model_name_or_path bert-base-multilingual-cased \
  --load_weights models/en-pruned-0.5/pytorch_model.bin \
  --train_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/en.txt" \
  --output_dir $OUTDIR \
  --sparse_ft no \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 256 \
  --save_steps 10000000 \
  --overwrite_output_dir \
  --learning_rate 5e-5 \
  --max_steps 100000 \
  --preprocessing_num_workers 15 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --validation_split_percentage 1 \
  --load_best_model_at_end \
  --save_total_limit 2 #> $OUTDIR/log.txt 2>&1
