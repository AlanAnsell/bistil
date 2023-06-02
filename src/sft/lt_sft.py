import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)


class LotteryTicketSparseFineTuner(SparseFineTuner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.setLevel(self.args.get_process_log_level())

    def train(self, *args, **kwargs):
        self.freeze()
        
        self.disable_masking()
        self.optimizer = None
        self.lr_scheduler = None
        self.set_training_len(
            self.sft_args.full_ft_min_steps_per_iteration,
            self.sft_args.full_ft_max_steps_per_iteration,
            self.sft_args.full_ft_max_epochs_per_iteration,
        )
        self.args.weight_decay = self.sft_args.full_weight_decay
        super().train(*args, **kwargs)

        self.sparsify()

        self.enable_masking()
        self.optimizer = None
        self.lr_scheduler = None
        self.set_training_len(
            self.sft_args.sparse_ft_min_steps_per_iteration,
            self.sft_args.sparse_ft_max_steps_per_iteration,
            self.sft_args.sparse_ft_max_epochs_per_iteration,
        )
        self.args.weight_decay = self.sft_args.sparse_weight_decay
        return super().train(*args, **kwargs)
        #self.finish()
        #self.enable_masking()
        #self.optimizer = None
        #self.lr_scheduler = None
        #self.set_training_len(
        #    self.sft_args.sparse_ft_min_steps_per_iteration,
        #    self.sft_args.sparse_ft_max_steps_per_iteration,
        #    self.sft_args.sparse_ft_max_epochs_per_iteration,
        #)
        #self.args.weight_decay = self.sft_args.sparse_weight_decay
        #output = super().train(*args, **kwargs)
        #return output

