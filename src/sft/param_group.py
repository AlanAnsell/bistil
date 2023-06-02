import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ParamGroup:

    def __init__(
        self,
        model,
        names, 
        density=None,
        nnz=None,
        origin='pretrained',
        full_regularization_strength=0.0,
        sparse_regularization_strength=0.0,
        regularization_masking=False,
        reset=True,
        sparsify=True,
        structure=None,
        block_shape=(32, 32),
    ):
        self.names = names
        self.origin = origin
        self.full_regularization_strength = full_regularization_strength
        self.sparse_regularization_strength = sparse_regularization_strength
        self.regularization_masking = regularization_masking
        self.reset = reset
        self.sparsify = sparsify
        self.structure = structure
        self.block_shape = block_shape

        if self.sparsify:
            if density is None and nnz is None:
                raise ValueError('One of density or nnz must be specified.')

            if density is not None and nnz is not None:
                logger.warning('Both density and nnz were specified; nnz will be used.')

            if structure in ['prune_intermediate', 'hard_concrete_linear', 'hard_concrete_attention', 'prune_ff', 'prune_attn']:
                if density is None:
                    raise ValueError(
                        'density must be specified when structure is prune_intermediate.'
                    )
                self.density = density
                self.nnz = None
            else:
                numel = 0
                for n in self.names:
                    numel += model.get_parameter(n).numel()
                if nnz is not None:
                    if nnz > numel:
                        raise ValueError(
                            f'nnz ({nnz}) exceeds total number of '
                            f'parameters in group {(numel)}.'
                        )
                    self.nnz = nnz
                    self.density = self.nnz / numel
                else:
                    self.density = density
                    self.nnz = int(numel * density)

        if self.structure == 'block':
            if self.block_shape != (32, 32):
                logger.warning(
                    f'Block sparsity with block shape {self.block_shape} '
                    'will have degraded performance; use shape (32, 32) '
                    'for best performance.'
                )
            
            for n in self.names:
                self._validate_block_structure(model, n)

    def __str__(self):
        sections = ['ParamGroup(']
        sections.append('\tparams = [')
        for n in sorted(list(self.names)):
            sections.append('\t\t' + str(n))
        sections.append('\t]')
        if self.sparsify:
            if self.structure is None:
                sparsification = 'unstructured'
            elif self.structure == 'block':
                sparsification = f'blocked with shape {self.block_shape}'
            else:
                sparsification = self.structure
            sections.append(f'\tsparsification = {sparsification}')
            sections.append(f'\torigin = {self.origin}')
            sections.append(f'\tdensity = {self.density:.4f}')
            sections.append(f'\tnnz = {self.nnz}')
        else:
            sections.append('\tsparsification = none')
        if self.full_regularization_strength != 0.0 or self.sparse_regularization_strength != 0.0:
            sections.append(
                f'\tfull regularization strength = {self.full_regularization_strength:.4f}'
            )
            sections.append(
                f'\tsparse regularization strength = {self.sparse_regularization_strength:.4f}'
            )
        sections.append(')')
        return '\n'.join(sections)

    def _validate_block_structure(self, model, n):
        p = model.get_parameter(n)
        if p.dim() != 2:
            raise ValueError(
                'Block sparsity can only be applied to parameters of dimension 2, '
                f'but {n} has dimension {p.dim()}.'
            )

        if p.size(0) % self.block_shape[0] != 0 or p.size(1) % self.block_shape[1] != 0:
            raise ValueError(
                f'Parameter {n} has shape {tuple(p.size())} which cannot be tiled by '
                f'block shape {self.block_shape}.'
            )

        param_path_segments = n.split('.')
        parent_module_path = '.'.join(param_path_segments[:-1])
        parent_module = model.get_submodule(parent_module_path)

        if not isinstance(parent_module, nn.Linear):
            raise ValueError(
                f'Block sparsity was requested for parameter {n}, '
                'but it does not belong to a nn.Linear module.'
            )

    def requires_originals(self):
        return self.reset or self.origin == 'pretrained'

    def originals_device(self):
        if self.requires_regularization() and self.origin == 'pretrained':
            return None
        return 'cpu'

    def requires_mask(self):
        return self.sparsify and self.structure is None

    def requires_regularization(self):
        return self.full_regularization_strength != 0.0 or self.sparse_regularization_strength != 0.0
