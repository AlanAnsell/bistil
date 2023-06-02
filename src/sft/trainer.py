import itertools
import logging
import os
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

#from pytorch_block_sparse import BlockSparseMatrix

from transformers import Trainer, TrainerCallback

from .param_group import ParamGroup
from .sft import SFT
from .sft_args import SftArguments
from .utils import (
    apply_attn_state_analysis,
    apply_ff_state_analysis,
    #CustomBlockSparseLinear,
    DataCollatorWithConsistentEvalMasking,
    prune_attn_heads,
    prune_ff_neurons,
    remove_attn_state_analysis,
    remove_ff_state_analysis,
    replace_with_block_sparse,
)
from .hard_concrete import (
    apply_hard_concrete_linear,
    remove_hard_concrete_linear,
    prune_hard_concrete_ff,
    apply_hard_concrete_attention,
    remove_hard_concrete_attention,
    prune_hard_concrete_attention,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _RegLossCallback(TrainerCallback):

    def __init__(self, sft):
        self._sft = sft

    def on_step_begin(self, *args, **kwargs):
        self._sft.calculate_reg_loss = True

    def on_epoch_begin(self, *args, **kwargs):
        if not self._sft._sparsified:
            self._sft.compute_regularization_masks()


class SparseFineTuner(Trainer):
    """ Superclass for Trainers that learn sparse fine-tunings. Keeps track
    of original model parameters so that difference vectors can be calculated
    at the end of training, and which parameters are masked so that gradients
    of fixed parameters can be zeroed.

    Args:
        sft_args: an SftArguments object containing SFT training options.
        maskable_params: a list of parameter names; the model parameters which
            are to be sparsely fine-tuned. Parameters not included in
            maskable_params but have requires_grad=True will be fully
            fine-tuned (this is typically preferable for model heads, for
            instance). If None, all parameters will be sparsely fine-tuned.
        **kwargs: arguments to pass to Trainer constructor.
    """
    def __init__(
        self,
        *args,
        sft_args=None,
        param_groups=None,
        sparsified=False,
        model=None,
        **kwargs
    ):
        if sft_args is None:
            self.sft_args = SftArguments()
        else:
            self.sft_args = sft_args

        if param_groups is None:
            self._param_groups = []
            #self._param_groups = [
            #    ParamGroup(
            #        model,
            #        n,
            #        density=self.sft_args.ft_params_proportion,
            #        nnz=self.sft_args.ft_params_num,
            #    )
            #    for n, p in model.named_parameters()
            #    if p.requires_grad
            #]
        else:
            self._param_groups = param_groups

        self._param_to_group = {}
        self._original_params = {}
        self._masks = {}
        self._regularization_masks = {}
        super().__init__(*args, model=model, **kwargs)
        for param_group in self._param_groups:
            if param_group.structure in ['prune_intermediate', 'prune_ff', 'prune_attn', 'hard_concrete_linear', 'hard_concrete_attention']:
                continue
            for n in param_group.names:
                p = model.get_parameter(n)
                self._param_to_group[n] = param_group
                if param_group.requires_originals():
                    self._original_params[n] = torch.zeros_like(
                        p.data,
                        device=param_group.originals_device(),
                        requires_grad=False,
                    ).copy_(p.data)
                if param_group.requires_mask():
                    self._masks[n] = torch.ones_like(
                        p,
                        dtype=torch.bool,
                        requires_grad=False,
                    )
                if (
                    param_group.requires_regularization() and 
                    param_group.regularization_masking
                ):
                    self._regularization_masks[n] = torch.ones_like(
                        p,
                        dtype=torch.bool,
                        requires_grad=False,
                    )

        #logger.setLevel(self.args.get_process_log_level())

        # Since the regularization loss is dependent only on the parameter
        # values, we can get away with calculating it only once per full step
        # rather than at every gradient accumulation step. This flag gets set
        # by a _RegLossCalculationCallback at the start of each full step to
        # tell us to do so.
        self.calculate_reg_loss = False
        self._reg_loss = 0.0 # Keeps track of the reg loss for logging purposes.
        self._hc_sparsity = 0.0
        self.add_callback(_RegLossCallback(self))

        self._sparsified = sparsified
        self._masking_enabled = True

    def analyse(self):
        self._move_model_to_device(self.model, self.args.device)
        self.model.zero_grad()
        self.model.train()
        batch_size_tmp = self.args.per_device_eval_batch_size
        self.args.per_device_eval_batch_size = 1
        eval_dataloader = self.get_eval_dataloader()

        for inputs in tqdm(
            eval_dataloader,
            desc="Analysing",
            disable=self.args.local_rank > 0
        ):
            inputs = self._prepare_inputs(inputs)
            #logger.info(inputs)
            outputs = self.model(**inputs)
            loss = outputs[0]
            #for n, p in self.model.named_parameters():
            #    logger.info(f'{n}: {p.grad_fn}')
            #logger.info(f'loss = {loss}')
            #logger.info(loss)
            loss.backward()
            #logger.info(torch.autograd.grad(loss, [self.model.bert.encoder.layer[11].output.dense.weight]))
            #logger.info([
            #    (t == 0.0).all().item()
            #    for t in torch.autograd.grad(loss, [self.model.word_outputs_deps, self.model.word_outputs_heads])
            #])
        self.args.per_device_eval_batch_size = batch_size_tmp

    def prune_intermediate_neurons(self, param_group):
        activation_probs = []
        total_neurons = 0
        for layer1_name, _ in param_group.names:
            module = self.model.get_submodule(layer1_name)
            total_neurons += module.module.out_features
            activation_probs.append(module.activation_probability())
        activation_probs = torch.cat(activation_probs).tolist()
        activation_probs.sort(reverse=True)
        threshold = activation_probs[int(param_group.density * len(activation_probs))]
        neurons_removed = 0
        for layer1_name, layer2_name in param_group.names:
            neurons_removed += prune_ff_neurons(self.model, layer1_name, layer2_name, threshold)
        logger.info(
            f'Pruned {neurons_removed}/{total_neurons} '
            f'({100 * neurons_removed / total_neurons:.3f}%) '
            f'of intermediate neurons in feed-forward layers with threshold '
            f'{threshold:.6f}.'
        )

    def prune_attn_heads(self, param_group):
        head_importances = []
        total_heads = 0
        for module_name in param_group.names:
            module = self.model.get_submodule(module_name)
            total_heads += module.module.self.num_attention_heads
            head_importances.append(module.head_importance)
            logger.info(f'Importances of attn heads in {module_name}: {module.head_importance}')
        head_importances = torch.cat(head_importances).tolist()
        head_importances.sort(reverse=True)
        threshold = head_importances[int(param_group.density * len(head_importances))]
        heads_removed = 0
        for module_name in param_group.names:
            heads_removed += prune_attn_heads(self.model, module_name, threshold)
        logger.info(
            f'Pruned {heads_removed}/{total_heads} '
            f'({100 * heads_removed / total_heads:.3f}%) '
            f'of attention heads with threshold {threshold:.6f}.'
        )

    def prune_hc_linear(self, param_group):
        log_alphas = []
        for layer1_name, _ in param_group.names:
            module = self.model.get_submodule(layer1_name)
            log_alphas.append(module.mask.log_alpha)
        log_alphas = torch.cat(log_alphas).tolist()
        log_alphas.sort(reverse=True)
        threshold = log_alphas[int(param_group.density * len(log_alphas))]
        for layer1_name, layer2_name in param_group.names:
            prune_hard_concrete_ff(self.model, layer1_name, layer2_name, threshold)
        for layer1_name, _ in param_group.names:
            remove_hard_concrete_linear(self.model, layer1_name)

    def prune_hc_attention(self, param_group):
        log_alphas = []
        for module_name in param_group.names:
            module = self.model.get_submodule(module_name)
            log_alphas.append(module.mask.log_alpha)
        log_alphas = torch.cat(log_alphas).tolist()
        log_alphas.sort(reverse=True)
        threshold = log_alphas[int(param_group.density * len(log_alphas))]
        for module_name in param_group.names:
            prune_hard_concrete_attention(self.model, module_name, threshold)
        for module_name in param_group.names:
            remove_hard_concrete_attention(self.model, module_name)

    def unstructured_sparsify(self, param_group):
        with torch.no_grad():
            diffs = []
            for n in tqdm(
                param_group.names,
                desc='Finding masking threshold',
                disable=self.args.local_rank > 0 or self.args.disable_tqdm,
            ):
                p = self.model.get_parameter(n)
                if param_group.origin == 'pretrained':
                    delta = p - self._original_params[n].to(p.device)
                else:
                    delta = p
                delta = delta.view(-1)
                #valid_indices = self._masks[n].view(-1)
                #valid_deltas = delta[valid_indices]
                abs_deltas = torch.abs(delta)
                diffs.extend(abs_deltas.tolist())
            
            if  param_group.nnz > len(diffs): 
                raise ValueError(
                    f'Was requested to sparsify param group with '
                    f'{param_group.nnz} non-zero elements/diffs, but group '
                    f'has only {len(diffs)} params in total.'
                )
            diffs = np.partition(diffs, len(diffs) - param_group.nnz)
            thresh = diffs[len(diffs) - param_group.nnz]
            
            n_masked = 0
            for n in tqdm(
                param_group.names,
                desc='Updating masks',
                disable=self.args.local_rank > 0 or self.args.disable_tqdm,
            ):
                p = self.model.get_parameter(n)
                if param_group.origin == 'pretrained':
                    abs_delta = (p - self._original_params[n].to(p.device)).abs()
                else:
                    abs_delta = p.abs()
                to_mask = (abs_delta >= thresh) #& (~self._masks[n])
                self._masks[n] = to_mask #| self._masks[n]
                n_masked += to_mask.sum()

                if param_group.reset:
                    p.copy_(self._original_params[n])

                if param_group.origin == 'zeros':
                    p.mul_(self._masks[n])

            logger.info(
                f'Masked {n_masked}/{len(diffs)} params with threshold '
                f'{thresh:.6f} in {param_group}'
            )

    def _get_active_blocks(self, param_group):
        block_descriptors = []
        br, bc = param_group.block_shape

        with torch.no_grad():
            for n in param_group.names:
                p = self.model.get_parameter(n)
                grid_rows = p.size(0) // br
                grid_cols = p.size(1) // bc
                for r in range(grid_rows):
                    for c in range(grid_cols):
                        block_descriptors.append((
                            torch.sum(torch.abs(p[r*br:(r+1)*br, c*bc:(c+1)*bc])),
                            n,
                            (r, c)
                        ))
        block_descriptors.sort(reverse=True)

        blocks_to_select = int(param_group.density * len(block_descriptors))
        return block_descriptors[:blocks_to_select]

    def block_sparsify(self, param_group):
        selected_blocks = self._get_active_blocks(param_group)
        selected_blocks.sort(key=lambda d: d[2])
        blocks_by_tensor = {n: [] for n in param_group.names}
        for value, n, coor in selected_blocks:
            blocks_by_tensor[n].append((value, coor))

        br, bc = param_group.block_shape
        for n, blocks in sorted(list(blocks_by_tensor.items())):
            p = self.model.get_parameter(n)
            grid_rows = p.size(0) // br
            grid_cols = p.size(1) // bc
            block_mask = torch.zeros(
                [grid_rows, grid_cols],
                dtype=torch.bool,
                device=p.device
            )
            data = []
            block_value = 0.0
            data_source = (
                self._original_params[n].to(p.device) if param_group.reset
                else p
            )
            with torch.no_grad():
                for value, (r, c) in blocks:
                    block_value += value
                    block_mask[r, c] = True
                    data.append(
                        data_source[r*br:(r+1)*br, c*bc:(c+1)*bc].transpose(0, 1)
                    )
                data_tensor = torch.concat(data, 0)
            block_sparse_p = BlockSparseMatrix(
                p.size(),
                block_mask,
                data_tensor,
                param_group.block_shape
            )
            replace_with_block_sparse(self.model, n, block_sparse_p)
            logger.info(
                f'Parameter {n} was replaced with a block-sparse tensor '
                f'consisting of {len(data)}/{grid_rows*grid_cols} blocks '
                f'of size {param_group.block_shape}. Value preserved = '
                f'{block_value:.4f}/{p.abs().sum():.4f}.'
            )

    def compute_regularization_masks(self):
        logger.info('Recomputing regularization masks.')
        for param_group in self._param_groups:
            if (
                param_group.requires_regularization() and 
                param_group.regularization_masking
            ):
                for n in param_group.names:
                    self._regularization_masks[n].fill_(True)
                active_blocks = self._get_active_blocks(param_group)
                br, bc = param_group.block_shape
                for _, n, (r, c) in active_blocks:
                    self._regularization_masks[n][r*br:(r+1)*br, c*bc:(c+1)*bc] = False

    def sparsify(self):
        prunable_ffs = []
        prunable_attns = []
        for param_group in self._param_groups:
            if param_group.sparsify:
                if param_group.structure == 'prune_ff':
                    prunable_ffs.append(param_group)
                elif param_group.structure == 'prune_attn':
                    prunable_attns.append(param_group)

        if prunable_ffs or prunable_attns:
            for param_group in prunable_ffs:
                for layer1_name, _ in param_group.names:
                    logger.info(f'Applying FF state analyser to {layer1_name}.')
                    apply_ff_state_analysis(self.model, layer1_name)
            for param_group in prunable_attns:
                for module_name in param_group.names:
                    logger.info(f'Applying attention state analyser to {module_name}.')
                    apply_attn_state_analysis(self.model, module_name)
            logger.info('Analysing states')
            self.analyse()
            for param_group in prunable_ffs:
                self.prune_intermediate_neurons(param_group)
            for param_group in prunable_attns:
                self.prune_attn_heads(param_group)
            for param_group in prunable_ffs:
                for layer1_name, _ in param_group.names:
                    remove_ff_state_analysis(self.model, layer1_name)
            for param_group in prunable_attns:
                for module_name in param_group.names:
                    remove_attn_state_analysis(self.model, module_name)

        for param_group in self._param_groups:
            if param_group.sparsify:
                if param_group.structure is None:
                    self.unstructured_sparsify(param_group)
                elif param_group.structure == 'block':
                    self.block_sparsify(param_group)
                elif param_group.structure == 'hard_concrete_linear':
                    for layer1_name, _ in param_group.names:
                        apply_hard_concrete_linear(self.model, layer1_name)
                elif param_group.structure == 'hard_concrete_attention':
                    for module_name in param_group.names:
                        apply_hard_concrete_attention(self.model, module_name)
                elif param_group.structure not in ['prune_ff', 'prune_attn']:
                    raise ValueError(
                        f'Unrecognised sparsity structure: {param_group.structure}'
                    )
            elif param_group.reset:
                with torch.no_grad():
                    for n in param_group.names:
                        self.model.get_parameter(n).copy_(self._original_params[n])

        for n, p in self.model.named_parameters():
            logger.info(f'{n}: {p.size()}')
            p.grad = None
        self.model.to(self.args.device)
        metrics = self.evaluate()
        self.log_metrics('eval', metrics)
        self._sparsified = True

    def finish(self):
        for param_group in self._param_groups:
            if param_group.structure == 'hard_concrete_linear':
                self.prune_hc_linear(param_group)
            elif param_group.structure == 'hard_concrete_attention':
                self.prune_hc_attention(param_group)

    def enable_masking(self):
        self._masking_enabled = True

    def disable_masking(self):
        self._masking_enabled = False

    def reset(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self._original_params[n])

    def freeze(self):
        for p in self._masks.values():
            p.data.zero_()

    def sft(self, eps=1e-7, submodule=None):
        """ Calculates the sparse difference vector between the current
        parameter values and the pre-trained values.

        Args:
            eps: differences smaller than this amount will be treated as zero,
            i.e. excluded from the SFT.

        Returns:
            An SFT containing the differences.
        """
        with torch.no_grad():
            _sft = SFT()
            for param_group in self._param_groups:
                for n in param_group.names:
                    if submodule is not None:
                        if not n.startswith(submodule + '.'):
                            continue
                        mapped_n = n[len(submodule)+1:]
                    else:
                        mapped_n = n
                    logger.info(n)
                    if param_group.sparsify:
                        if param_group.structure is None:
                            p = self.model.get_parameter(n)
                            if param_group.origin == 'pretrained':
                                delta = p.cpu() - self._original_params[n].cpu()
                                abs_delta = torch.abs(delta)
                                significant = abs_delta > eps
                                delta = delta * significant
                                delta = delta.to_sparse().coalesce()
                                _sft.add_diff(mapped_n, delta)
                            else:
                                _sft.add_abs(
                                    mapped_n,
                                    p.to_sparse().coalesce()
                                )
                        elif param_group.structure == 'block':
                            _sft.add_block_sparse(mapped_n, self.model)
                        else:
                            raise ValueError(
                                'Unrecognised sparsity structure: '
                                f'{param_group.structure}.'
                            )
                    else:
                        _sft.add_abs(mapped_n, self.model.get_parameter(n))

            return _sft

    def subnetwork(self):
        params = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.maskable_params:
                    params[n] = (p * self._mask[n]).cpu()
                elif p.requires_grad:
                    params[n] = p.cpu()
        return params

    def set_training_len(self, min_steps, max_steps, max_epochs):
        if max_steps is None and max_epochs is None:
            raise ValueError('Length of sft training not specified.')
        if min_steps is not None and max_steps is not None and min_steps > max_steps:
            raise ValueError('min_steps cannot be > max_steps')

        if max_epochs is None:
            self.args.max_steps = max_steps
        else:
            n_steps = max_epochs * len(self.train_dataset) // (
                self.args.per_device_train_batch_size *
                self.args.gradient_accumulation_steps
            )
            logger.info(f'{max_epochs} epochs = {n_steps} steps')
        
            if max_steps is None or n_steps < max_steps:
                if min_steps is not None and n_steps < min_steps:
                    self.args.max_steps = min_steps
                else:
                    self.args.num_train_epochs = max_epochs
                    self.args.max_steps = -1
            else:
                self.args.max_steps = max_steps

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        #for n, p in self.model.named_parameters():
        #    logger.info(f'{n}: {p.grad}.')

        if self.calculate_reg_loss:
            diffs = []
            numel = 0
            for n, p in self.model.named_parameters():
                param_group = self._param_to_group.get(n)
                if param_group is None:
                    continue
                reg_strength = (
                    param_group.sparse_regularization_strength if self._sparsified
                    else param_group.full_regularization_strength
                )
                if reg_strength != 0.0:
                    if param_group.origin == 'pretrained':
                        diff = torch.abs(p - self._original_params[n])
                    else:
                        diff = torch.abs(p)
                    if param_group.regularization_masking:
                        diff.mul_(self._regularization_masks[n])
                    diff = torch.sum(diff)
                    diffs.append(reg_strength * diff)
                    numel += p.numel()
            if diffs:
                reg_loss = torch.sum(torch.stack(diffs)) / numel
                reg_loss.backward()
                self._reg_loss += float(reg_loss)

            #if self._sparsified: # and self.state.global_step > 50000:
            #    sparsity_loss = []
            #    all_expected_params = 0
            #    all_total_params = 0
            #    for param_group in self._param_groups:
            #        if param_group.structure == 'hard_concrete_linear':
            #            expected_params = sum(
            #                self.model.get_submodule(layer1_name).expected_ones()
            #                for layer1_name, _ in param_group.names
            #            )
            #            total_params = sum(
            #                self.model.get_submodule(layer1_name).total_params()
            #                for layer1_name, _ in param_group.names
            #            )
            #            sparsity_loss.append(self.sft_args.hc_lambda * expected_params)
            #            all_expected_params += expected_params.item()
            #            all_total_params += total_params
            #        if param_group.structure == 'hard_concrete_attention':
            #            expected_params = sum(
            #                self.model.get_submodule(module_name).expected_ones()
            #                for module_name in param_group.names
            #            )
            #            total_params = sum(
            #                self.model.get_submodule(module_name).total_params()
            #                for module_name in param_group.names
            #            )
            #            sparsity_loss.append(self.sft_args.hc_lambda * expected_params)
            #            all_expected_params += expected_params.item()
            #            all_total_params += total_params
            #    if sparsity_loss:
            #        sparsity_loss = torch.sum(torch.stack(sparsity_loss))
            #        sparsity_loss.backward()
            #        self._hc_sparsity += all_expected_params / all_total_params

            self.calculate_reg_loss = False

        #l1_reg = (
        #    self.sft_args.sparse_l1_reg
        #    if self._masking_enabled
        #    else self.sft_args.full_l1_reg
        #)
        #if l1_reg != 0.0 and self.calculate_reg_loss:
        #    # Since we only calculate reg loss once per full step.
        #    l1_reg *= self.args.gradient_accumulation_steps
        #    l1_dists = []
        #    for n, p in self.model.named_parameters():
        #        if (
        #            p.requires_grad and
        #            (n in self.maskable_params or
        #                not self.sft_args.apply_reg_to_sparse_only)
        #        ):
        #            l1_dists.append(
        #                torch.sum(torch.abs(p))
        #            )
        #    reg_loss = l1_reg * torch.sum(torch.stack(l1_dists)) / self._num_params
        #    reg_loss.backward()
        #    self._reg_loss += float(reg_loss)
        #    self.calculate_reg_loss = False

        if self._masking_enabled:
            # set gradients for non-trainable parametres to zero.
            for n, p in self.model.named_parameters():
                param_group = self._param_to_group.get(n)
                if param_group is None:
                    continue
                if param_group.requires_mask() and p.grad is not None:
                    p.grad *= self._masks[n]

        return loss

    def evaluate(self, *args, **kwargs):
        if isinstance(self.data_collator, DataCollatorWithConsistentEvalMasking):
            self.data_collator.eval()
        output = super().evaluate(*args, **kwargs)
        if isinstance(self.data_collator, DataCollatorWithConsistentEvalMasking):
            self.data_collator.train()
        return output

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self._reg_loss != 0.0:
                logs['l1_reg_loss'] = round(self._reg_loss / (self.state.global_step - self._globalstep_last_logged), 4)
                self._reg_loss = 0.0
            if self._hc_sparsity != 0.0:
                logs['hc_sparsity'] = round(self._hc_sparsity / (self.state.global_step - self._globalstep_last_logged), 4)
                self._hc_sparsity = 0.0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

