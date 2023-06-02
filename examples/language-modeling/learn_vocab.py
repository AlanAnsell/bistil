import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer, HfArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Arguments:

    source_file: str = field(
        default=None,
        metadata={
            "help": "File containing source language corpus."
        },
    )
    target_file: str = field(
        default=None,
        metadata={
            "help": "File containing target language corpus."
        },
    )
    output_file: str = field(
        default=None,
        metadata={
            "help": "Where to save tokenizer."
        },
    )



def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    source_lines = []
    with open(args.source_file, 'r') as f:
        for line in f:
            source_lines.append(line.strip())

    target_lines = []
    with open(args.target_file, 'r') as f:
        for line in f:
            target_lines.append(line.strip())

    if len(target_lines) < len(source_lines):
        upscaling = int(round(float(len(source_lines)) / float(len(target_lines))))
        #source_lines = source_lines[:len(target_lines)]
    else:
        upscaling = 1


    old_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    #tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
    #tokenizer.pre_tokenizer = BertPreTokenizer()
    #trainer = WordPieceTrainer(
    #    vocab_size=30000,
    #    limit_alphabet=1000,
    #    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    #)
    #tokenizer.train_from_iterator(
    #    source_lines + upscaling * target_lines,
    #    trainer=trainer
    #)
    tokenizer = old_tokenizer.train_new_from_iterator(
        source_lines + upscaling * target_lines,
        30000
    )
    tokenizer.save_pretrained(args.output_file)

if __name__ == "__main__":
    main()
