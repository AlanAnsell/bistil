#!/bin/bash
#SBATCH --gres=gpu:rtx_3080:1
#SBATCH --array=0-3

#N="$SLURM_ARRAY_TASK_ID"
N=0
TARGET_LANGS=("mt" "fo" "ibo" "kin")
TARGET_LANG=${TARGET_LANGS[N]}
SOURCE_LANG=en

OUTDIR=$HDD/experiments/distil/language-modeling/distil/mbert-${SOURCE_LANG}-${TARGET_LANG}-vocab
#OUTDIR=$HDD/experiments/distil/language-modeling/distil/mbert-${TARGET_LANG}-tinybert-tmeval
mkdir -p $OUTDIR

python trim_vocab.py \
  --model_name_or_path bert-base-multilingual-cased \
  --source_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${SOURCE_LANG}.txt" \
  --target_file "$HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt" \
  --output_dir ${OUTDIR} \
  --overwrite_output_dir \
  --preprocessing_num_workers 15 #> $OUTDIR/log.txt 2>&1

rm "${OUTDIR}/tokenizer.json"
