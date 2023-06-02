#!/bin/bash
SOURCE_FILE=$1
TARGET_LANG=$2
OUTPUT_FILE=$3

python learn_vocab.py \
  --source_file $SOURCE_FILE \
  --target_file $TARGET_FILE \
  --output_file $OUTPUT_FILE
