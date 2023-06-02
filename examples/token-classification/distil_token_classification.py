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
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import pandas as pd
from datasets import ClassLabel, load_dataset, load_metric
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sft import (
    density,
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

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    eval_languages: Optional[str] = field(
        default=None,
        metadata={"help": "An optional CSV file containing eval language treebanks and ft paths."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )


    def __post_init__(self):
        if self.multisource_data is self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


#class MyDataCollator(DataCollatorForTokenClassification):
#
#    def __call__(self, examples):
#        logger.info(examples[0])
#        super().__call__(examples)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SftArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, sft_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, sft_args, training_args = parser.parse_args_into_dataclasses()

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.multisource_data is None:
        dataset_descriptor = {
            'name': data_args.dataset_name,
            'config_name': data_args.dataset_config_name,
        }

        if not training_args.do_train:
            dataset_descriptor['train_split'] = None
        if not training_args.do_eval:
            dataset_descriptor['validation_split'] = None

        raw_datasets = load_single_dataset(
            dataset_descriptor,
            training_args,
            cache_dir=model_args.cache_dir,
            overwrite_cache=data_args.overwrite_cache,
        )
        lang_sfts = None

    else:
        with open(data_args.multisource_data) as f:
            multisource_json = json.load(f)

        raw_datasets, lang_sfts = load_multisource_dataset(
            multisource_json,
            training_args,
            cache_dir=model_args.cache_dir,
            overwrite_cache=data_args.overwrite_cache,
        )

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets[data_args.eval_split].column_names
        features = raw_datasets[data_args.eval_split].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            if label != '_':
                unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = sorted(features[label_column_name].feature.names)
        if '_' in label_list:
            label_list.remove('_')
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer_name_or_path = model_args.student_tokenizer_name if model_args.student_tokenizer_name else model_args.student_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_model_name_or_path,
        num_labels=num_labels,
        #label2id=label_to_id,
        #id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    teacher_model = AutoModelForTokenClassification.from_pretrained(
        model_args.teacher_model_name_or_path,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
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
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
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
        #label2id=label_to_id,
        #id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    student_model = AutoModelForTokenClassification.from_pretrained(
        model_args.student_model_name_or_path,
        config=student_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = TeacherStudentModel(
        teacher_model,
        student_model,
        alpha_attn=1.0,
        alpha_ce=1.0,
        alpha_hidden=1.0,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        for idx in range(len(examples[text_column_name])):
            examples[label_column_name][idx] = [
                features[label_column_name].feature.names[label]
                for label in examples[label_column_name][idx]
            ]
            invalid_indices = set(
                i for i, pos in enumerate(examples[label_column_name][idx])
                if pos == '_'
            )
            for col in [text_column_name, label_column_name]:
                examples[col][idx] = [
                    v for i, v in enumerate(examples[col][idx])
                    if i not in invalid_indices
                ]
        
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            if data_args.max_seq_length is not None:
                train_dataset = train_dataset.filter(
                    lambda example: len(example['input_ids']) <= data_args.max_seq_length
                )

    if training_args.do_eval:
        if data_args.eval_split not in raw_datasets:
            raise ValueError("--do_eval requires an evaluation dataset")
        eval_dataset = raw_datasets[data_args.eval_split]
        with training_args.main_process_first(desc="evaluation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on evaluation dataset",
            )
            if data_args.max_seq_length is not None:
                eval_dataset = eval_dataset.filter(
                    lambda example: len(example['input_ids']) <= data_args.max_seq_length
                )

    # Data collator
    #data_collator = MyDataCollator(
    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        #padding='max_length',
        #max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Metrics
    metric = load_metric("seqeval")

    def print_thing(x, tab=0):
        if isinstance(x, tuple):
            for item in x:
                print_thing(item, tab+1)
        elif isinstance(x, np.ndarray):
            rep = str(x)
            rep = rep.split('\n')
            for line in rep:
                print((4*tab)*' ' + line)
        else:
            print(f'x is a {type(x)}')

    def compute_metrics(p):
        predictions, labels = p
        #print_thing(predictions)
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    if sft_args.freeze_layer_norm:
        for n, p in student_model.named_parameters():
            if 'LayerNorm' in n:
                p.requires_grad = False

    #if training_args.do_train:
    #    with torch.no_grad():
    #        for n, p in model.named_parameters():
    #            if p.requires_grad and not n.startswith(model.base_model_prefix):
    #                p.zero_()

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
    #def is_linear_weight(n):
    #    param_path_segments = n.split('.')
    #    parent_module_path = '.'.join(param_path_segments[:-1])
    #    parent_module = model.get_submodule(parent_module_path)
    #    return n.endswith('weight') and isinstance(parent_module, torch.nn.Linear)

    #param_groups = [
    #    ParamGroup(
    #        model,
    #        [n],
    #        density=0.25,
    #        structure='block',
    #        origin='zeros',
    #        regularization_strength=sft_args.full_l1_reg,
    #        regularization_masking=True,
    #        reset=False,
    #    )
    #    for n, p in model.named_parameters()
    #    if (
    #        n.startswith(model.base_model_prefix) and
    #        is_linear_weight(n) and
    #        p.requires_grad
    #    )
    #]
    #param_groups.append(
    #    ParamGroup(
    #        model,
    #        [
    #            n for n, p in model.named_parameters()
    #            if (
    #                n.startswith(model.base_model_prefix) and
    #                not is_linear_weight(n) and
    #                p.requires_grad
    #            )
    #        ],
    #        density=0.08,
    #        #nnz=sft_args.ft_params_num,
    #        structure=None,
    #        reset=False,
    #    )
    #)
    #param_groups.append(
    #    ParamGroup(
    #        model,
    #        [
    #            n for n, p in model.named_parameters()
    #            if not n.startswith(model.base_model_prefix) and p.requires_grad
    #        ],
    #        density=0.25,
    #        origin='zeros',
    #        reset=False
    #    )
    #)
    for param_group in param_groups:
        logger.info(param_group)

    #maskable_params = [
    #    n for n, p in model.named_parameters()
    #    if n.startswith(model.base_model_prefix) and p.requires_grad
    #]

    # Initialize our Trainer
    trainer_cls = LotteryTicketSparseFineTuner
    #trainer_cls = LotteryTicketPruner
    #trainer_cls = MultiSourcePlugin(trainer_cls)
    trainer = trainer_cls(
        sft_args=sft_args,
        param_groups=param_groups,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #source_sfts=lang_sfts,
    )
    logger.info(train_dataset)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        #trainer.save_model()  # Saves the tokenizer too for easy upload
        #model.student_model.save_pretrained(training_args.output_dir)

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        #trainer.save_state()

        if training_args.local_rank <= 0:
            #torch.save(trainer.subnetwork(), os.path.join(training_args.output_dir, 'pytorch_subnetwork.bin'))
            trainer.sft(submodule='student_model').save(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        #max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        #metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
