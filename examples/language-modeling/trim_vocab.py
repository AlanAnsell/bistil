# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by the Cambridge Language Technology Lab
import copy
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import tokenizers
import transformers
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file for the source language."}
    )
    target_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file for the target language."}
    )
    probability_threshold: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Minimum token probability to be kept."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #if training_args.per_device_train_batch_size != 42:
    #    raise ValueError("Batch size must be 42!!!")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    #data_files = data_args.train_file.split(',')
    #data_files = {'target': parts[0]}
    #if len(parts) > 1:
    #    data_files['source'] = parts[1]
    #logger.info(data_files)

    raw_datasets = {}
    if data_args.source_file:
        raw_datasets['source'] = load_dataset(
            'text',
            data_files={'train': data_args.source_file},
            cache_dir=model_args.cache_dir,
        )
    if data_args.target_file:
        raw_datasets['target'] = load_dataset(
            'text',
            data_files={'train': data_args.target_file},
            cache_dir=model_args.cache_dir,
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        **tokenizer_kwargs
    )

    text_column_name = "text" #if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            add_special_tokens=False,
            max_length=int(1e9),
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = {
            k: v.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                #remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )
            for k, v in raw_datasets.items()
        }

    def plot(prob, label):
        prob = list(prob)
        prob.sort(reverse=True)
        plt.plot(list(range(len(tokenizer))), prob, label=label)

    combined = np.zeros([len(tokenizer)])
    for split, examples in tokenized_datasets.items():
        examples = examples['train']
        logger.info(examples[0])
        counts = np.zeros([len(tokenizer)], dtype=np.int64)
        for i, example in enumerate(examples['input_ids']):
            for input_id in example:
                counts[input_id] += 1
            if (i+1) % 10000000 == 0:
                logger.info(f'{i+1}/{len(examples)}')
        counts = counts / counts.sum()
        #plot(counts, split)
        combined = np.maximum(combined, counts)

    #plot(combined, 'combined')

    #plt.yscale('log')
    #plt.legend()
    #plt.show()
    
    keep = combined >= data_args.probability_threshold
    for special_token in tokenizer.special_tokens_map.values():
        keep[tokenizer.vocab[special_token]] = True

    keep_tokens = np.squeeze(np.argwhere(keep))
    logger.info(
        f'Retaining {len(keep_tokens)} tokens with probability >= '
        f'{data_args.probability_threshold:.10f}.'
    )
    old2new = {k: i for i, k in enumerate(keep_tokens)}

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_output_embeddings()
    embeddings.weight.data = embeddings.weight.data[keep, :]
    embeddings.bias.data = embeddings.bias.data[keep]

    model.config.vocab_size = len(keep_tokens)

    model.save_pretrained(training_args.output_dir)

    tokenizer_state = json.loads(tokenizer.backend_tokenizer.__getstate__())
    vocab = tokenizer_state['model']['vocab']
    #vocab_type = type(tokenizer_state['vocab'])
    #logger.info(f'vocab is a {vocab_type}.')
    #logger.info(tokenizer_state['vocab'])
    if isinstance(vocab, dict):
        tokenizer_state['model']['vocab'] = {
            k: i for i, (k, v) in enumerate(vocab.items())
            if keep[v]
        }
    elif isinstance(vocab, list):
        tokenizer_state['model']['vocab'] = [
            tuple(item) for i, item in enumerate(vocab)
            if keep[i]
        ]
    else:
        raise TypeError(
            f"tokenizer_state['vocab'] must be a dict or list, but was type {vocab_type}."
        )
    for added_token in tokenizer_state['added_tokens']:
        added_token['id'] = old2new[added_token['id']]

    #tokenizer_class = getattr(tokenizers.models, tokenizer_state.pop('type'))
    #logger.info(f'tokenizer_class: {tokenizer_class}')
    tokenizer._tokenizer = tokenizers.Tokenizer.from_str(json.dumps(tokenizer_state))
    #tokenizer.backend_tokenizer.model = tokenizer_class(**tokenizer_state)
    logger.info(f'added vocab: {tokenizer.get_added_vocab()}')
    tokenizer.save_pretrained(training_args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
