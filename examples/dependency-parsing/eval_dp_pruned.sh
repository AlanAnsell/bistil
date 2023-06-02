#!/bin/bash
LANG=$1 # Buryat
TREEBANK=$2
#BASE_MODEL=models/${LANG}-en-${PROP}-8p
BASE_MODEL=bert-base-multilingual-cased
TASK_MODEL=models/prune-all-test-mt
#BASE_MODEL=../language-modeling/models/${LANG}-${PROP}-taught #/pytorch_model.bin
#BASE_MODEL=../language-modeling/models/${LANG}-en-${PROP}
#BASE_MODEL=../language-modeling/models/${LANG}-${PROP}
#TASK_FT=models/en-0.2-ltsft-0.08 #pytorch_subnetwork.bin # Single-source task SFT
#TASK_FT=cambridgeltl/mbert-task-sft-dp
#BASE_MODEL=models/full-reg-1/checkpoint-15680
#BASE_MODEL=bert-base-multilingual-cased
#LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small

if [[ "$LANG" == "wol" ]]; then
    LANG=wo
fi

python run_dp.py \
  --model_name_or_path $BASE_MODEL \
  --load_weights $TASK_MODEL/pytorch_model.bin \
  --relufy yes \
  --tokenizer_name bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name "${LANG}_${TREEBANK}" \
  --output_dir results/${LANG} \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir \
  --eval_split test
  #--lang_ft $LANG_FT \
