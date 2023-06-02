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
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch import nn

from datasets import ClassLabel, load_dataset, load_metric

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dp.modeling_biaffine import BiaffineParser
from dp.utils_udp import (
    DataCollatorForDependencyParsing,
    DependencyParsingTrainer,
    dataset_preprocessor,
    UD_HEAD_LABELS,
    UDTrainingArguments,
)

from sft import (
    density,
    sparsify,
    load_multisource_dataset,
    load_single_dataset,
    LotteryTicketPruner,
    LotteryTicketSparseFineTuner,
    MultiSourcePlugin,
    ParamGroup,
    SFT,
    SftArguments,
    TeacherStudentModel,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    student_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    student_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path for student model."}
    )
    teacher_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path for student model."}
    )
    load_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Weights to apply to model."},
    )
    relufy: bool = field(
        default=False,
        metadata={"help": "Whether to change hidden activation function to relu."},
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
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

    multisource_data: Optional[str] = field(
        default=None, metadata={"help": "File describing JSON descriptor of multi-source data."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_split: Optional[str] = field(
        default='validation', metadata={"help": "The split to evaluate on."}
    )
    max_examples: int = field(
        default=None,
        metadata={'help': 'Sets the maximum number of examples that may be used in training.'}
    )
    eval_split: Optional[str] = field(
        default='validation',
        metadata={"help": "validation or test."},
    )
    eval_languages: Optional[str] = field(
        default=None,
        metadata={"help": "An optional CSV file containing eval language treebanks and ft paths."},
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum sequence length; longer sequences will be discarded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SftArguments, UDTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, sft_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, sft_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ['labels_arcs', 'labels_rels']

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s.%(msecs)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    labels = UD_HEAD_LABELS
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.student_tokenizer_name if model_args.student_tokenizer_name else model_args.student_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    padding = "max_length" if data_args.pad_to_max_length else False
    preprocessor = dataset_preprocessor(
        tokenizer,
        label2id,
        padding,
    )

    if data_args.multisource_data is None:
        dataset_descriptor = {
            'name': data_args.dataset_name,
            'config_name': data_args.dataset_config_name,
        }

        if not training_args.do_train:
            dataset_descriptor['train_split'] = None
        if not training_args.do_eval:
            dataset_descriptor['validation_split'] = None

        data = load_single_dataset(
            dataset_descriptor,
            training_args,
            preprocessor=preprocessor,
            cache_dir=model_args.cache_dir,
            max_seq_length=data_args.max_seq_length,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache,
            remove_original_columns=True,
        )
        lang_sfts = None

    else:
        with open(data_args.multisource_data) as f:
            multisource_json = json.load(f)

        data, lang_sfts = load_multisource_dataset(
            multisource_json,
            training_args,
            preprocessor=preprocessor,
            cache_dir=model_args.cache_dir,
            max_seq_length=data_args.max_seq_length,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache,
            remove_original_columns=True,
        )

    train_dataset = data['train'] if training_args.do_train else None
    eval_dataset = data[data_args.eval_split] if training_args.do_eval else None

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
        pad_token_id=tokenizer.pad_token_id,
    )
    teacher_model = BiaffineParser(teacher_config).from_pretrained(
        model_args.teacher_model_name_or_path,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
    )
    if sft_args.lang_ft is not None:
        lang_ft = SFT(sft_args.lang_ft)
        logger.info(f'Applying language fine-tuning {sft_args.lang_ft} to teacher model.')
        lang_ft.apply(teacher_model, with_abs=False)
    if sft_args.task_ft is not None:
        task_ft = SFT(sft_args.task_ft)
        logger.info(f'Applying task fine-tuning {sft_args.task_ft} to teacher model.')
        task_ft.apply(teacher_model, with_abs=True)
    teacher_model.requires_grad_(False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        model_args.teacher_tokenizer_name if model_args.teacher_tokenizer_name else model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    teacher_embeddings = teacher_model.get_input_embeddings()
    assert len(teacher_tokenizer) == teacher_embeddings.weight.size(0)
    if len(teacher_tokenizer) != len(tokenizer):
        embeddings_to_keep = [False] * len(teacher_tokenizer)
        vocab = tokenizer.vocab
        teacher_vocab = teacher_tokenizer.vocab
        for k in vocab.keys():
            v = teacher_vocab.get(k, None)
            if v is not None:
                embeddings_to_keep[v] = True
        teacher_embeddings.weight.data = teacher_embeddings.weight.data[embeddings_to_keep, :]
        assert teacher_embeddings.weight.size(0) == len(tokenizer)
        logger.info('Removed embeddings from teacher model which are not present in student vocabulary.')

    student_config = AutoConfig.from_pretrained(
        model_args.student_model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
        pad_token_id=tokenizer.pad_token_id,
    )
    student_model = BiaffineParser(student_config).from_pretrained(
        model_args.student_model_name_or_path,
        config=student_config,
        cache_dir=model_args.cache_dir,
    )

    model = TeacherStudentModel(
        teacher_model,
        student_model,
        alpha_attn=1.0,
        alpha_ce=1.0,
        alpha_hidden=1.0,
        #alpha_task=1.0,
        mask_logits=False,
    )

    if sft_args.freeze_layer_norm:
        for n, p in model.named_parameters():
            if 'LayerNorm' in n:
                p.requires_grad = False

    param_groups = [
        ParamGroup(
            model,
            [
                f'student_model.{n}' for n, p in student_model.named_parameters()
                if p.requires_grad and n.startswith(student_model.base_model_prefix)
            ],
            origin='pretrained',
            density=sft_args.ft_params_proportion,
            structure=None,
            reset=True,
        ),
        ParamGroup(
            model,
            [
                f'student_model.{n}' for n, p in student_model.named_parameters()
                if p.requires_grad and not n.startswith(student_model.base_model_prefix)
            ],
            sparsify=False,
            reset=True,
        )
    ]

    for param_group in param_groups:
        logger.info(param_group)
    #maskable_params = [
    #    n for n, p in model.named_parameters()
    #    if p.requires_grad
    #    #if n.startswith(model.base_model_prefix) and p.requires_grad
    #]

    data_collator = DataCollatorForDependencyParsing(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    training_args.use_legacy_prediction_loop = True
    # Initialize our Trainer
    trainer_cls = LotteryTicketSparseFineTuner
    #trainer_cls = Trainer
    #trainer_cls = LotteryTicketPruner
    trainer_cls = DependencyParsingTrainer(trainer_cls)
    # Optional if using single-source training
    #trainer_cls = MultiSourcePlugin(trainer_cls)

    trainer = trainer_cls(
        sft_args=sft_args,
        param_groups=param_groups,
        #maskable_params=maskable_params,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        #source_sfts=lang_sfts,
    )

    for n, p in model.named_parameters():
        logger.info(f'{n}: {p.size()}, {p.requires_grad}')

    # Training
    if training_args.do_train:
        trainer.train()
        #trainer.save_model()
        #model.student_model.save_pretrained(training_args.output_dir)

        if training_args.local_rank <= 0:
            #torch.save(trainer.subnetwork(), os.path.join(training_args.output_dir, 'pytorch_subnetwork.bin'))
            trainer.sft(submodule='student_model').save(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
