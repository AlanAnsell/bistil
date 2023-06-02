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
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sft import (
    load_multisource_dataset,
    load_single_dataset,
    LotteryTicketSparseFineTuner,
    MultiSourcePlugin,
    ParamGroup,
    SFT,
    SftArguments,
    TeacherStudentModel,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    multisource_data: Optional[str] = field(
        default=None, metadata={"help": "JSON multi-source dataset descriptor."}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "Name of NLI dtaaset."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train split."},
    )
    validation_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the validation split."},
    )
    test_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the test split."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train data file (tsv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation data file (tsv file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data (tsv file)."},
    )
    label_file: Optional[str] = field(
        default=None,
        metadata={"help": "File containing label names."},
    )

    eval_split: Optional[str] = field(
        default='validation', metadata={"help": "The split to evaluate on."}
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SftArguments, TrainingArguments))
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
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.multisource_data is None:
        dataset_descriptor = {}
        if data_args.dataset_name:
            dataset_descriptor['name'] = data_args.dataset_name
            if data_args.dataset_config_name:
                dataset_descriptor['config_name'] = data_args.dataset_config_name
        else:
            if data_args.train_file:
                dataset_descriptor['train_file'] = data_args.train_file
            if data_args.validation_file:
                dataset_descriptor['validation_file'] = data_args.validation_file
            if data_args.test_file:
                dataset_descriptor['test_file'] = data_args.test_file

            dataset_descriptor['file_type'] = 'csv'
            dataset_descriptor['load_kwargs'] = {
                'delimiter': '\t',
            }

        if data_args.train_split is not None:
            dataset_descriptor['train_split'] = data_args.train_split
        if data_args.validation_split is not None:
            dataset_descriptor['validation_split'] = data_args.validation_split
        if data_args.test_split is not None:
            dataset_descriptor['test_split'] = data_args.test_split

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

    def get_labels(dataset):
        if isinstance(dataset.features['label'], ClassLabel):
            return dataset.features['label'].names

        if data_args.label_file is None:
            raise ValueError('--label_file must be provided if dataset is not a datasets.Dataset')

        label_list = []
        with open(data_args.label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    label_list.append(line)
        return label_list

    logger.info(raw_datasets)
    logger.info(raw_datasets['validation'][0])

    if training_args.do_train:
        train_dataset = raw_datasets['train']
        label_list = get_labels(train_dataset)
    if training_args.do_eval:
        eval_dataset = raw_datasets[data_args.eval_split]
        label_list = get_labels(eval_dataset)

    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.student_tokenizer_name if model_args.student_tokenizer_name else model_args.student_model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
    )
    student_config = AutoConfig.from_pretrained(
        model_args.student_model_name_or_path,
        num_labels=num_labels,
        finetuning_task="nli",
        cache_dir=model_args.cache_dir,
        pad_token_id=tokenizer.pad_token_id,
    )
    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.student_model_name_or_path,
        config=student_config,
        cache_dir=model_args.cache_dir,
    )

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_model_name_or_path,
        num_labels=num_labels,
        finetuning_task="nli",
        cache_dir=model_args.cache_dir,
        pad_token_id=tokenizer.pad_token_id,
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
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
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
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

    model = TeacherStudentModel(
        teacher_model,
        student_model,
        alpha_attn=1.0,
        alpha_ce=1.0,
        alpha_hidden=1.0,
        #alpha_task=1.0,
        mask_logits=False,
        eval_bypass_teacher=True,
    )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        tokenized_examples = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        tokenized_examples['labels'] = [
            label2id.get(label, label)
            for label in examples['label']
        ]
        return tokenized_examples

    if training_args.do_train:
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Get the metric function
    metric = load_metric("xnli")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

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

    for n, p in model.named_parameters():
        logger.info(f'{n}: {p.requires_grad}')

    # Initialize our Trainer
    trainer_cls = LotteryTicketSparseFineTuner
    #trainer_cls = MultiSourcePlugin(trainer_cls)
    trainer = trainer_cls(
        sft_args=sft_args,
        #maskable_params=maskable_params,
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

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        #max_train_samples = (
        #    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        #)
        #metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        #trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        #trainer.save_state()

        if training_args.local_rank <= 0:
            trainer.sft(submodule='student_model').save(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
