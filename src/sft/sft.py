import logging
import os

import numpy as np
import torch

from .hf_utils import pull_from_hf_model_hub
from .utils import CustomBlockSparseLinear, replace_with_block_sparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SFT_FILE_NAME = 'pytorch_diff.bin'


def encode_tensor(tensor):
    """
    Compresses a sparse tensor by flattening indices into a single dimension to
    reduce size of saved SFTs.
    """
    if tensor.is_sparse:
        multipliers = np.cumprod([1] + list(tensor.size())[:-1], dtype=np.int64)
        coordinates = np.array(tensor.indices().to('cpu'), dtype=np.int64)
        indices = np.matmul(multipliers, coordinates)
        perm = list(range(len(indices)))
        perm.sort(key=lambda x: indices[x])
        indices = indices[perm]
        index_steps = indices[1:] - indices[:-1]
        index_steps = index_steps.tolist()
        if len(indices) > 0:
            index_steps = [indices[0]] + index_steps
        values = tensor.values().to('cpu')[perm]
        return {
            'size': tensor.size(),
            'index_steps': index_steps,
            'values': values,
        }
    return tensor


def decode_tensor(encoding):
    """
    Inverse of encode_sparse_tensor.
    """
    if isinstance(encoding, torch.Tensor):
        return encoding

    size = encoding['size']
    index_steps = encoding['index_steps']
    values = encoding['values']

    indices = np.cumsum(index_steps, dtype=np.int32)
    divisors = np.concatenate([[1], np.cumprod(list(size))[:-1]], dtype=np.int32)
    modulos = np.array(list(size), dtype=np.int32)
    
    coordinates = np.expand_dims(indices, 0) // np.expand_dims(divisors, 1)
    coordinates = coordinates % np.expand_dims(modulos, 1)

    return torch.sparse_coo_tensor(coordinates, values, size=size).coalesce()


class SFT:
    """
    Represents a sparse fine-tuning of a pre-trained base model. Contains two
    sets of tensors, "diffs", the difference tensors for sparsely fine-tuned
    parameters, and "abs", the fully-specified values of fully fine-tuned
    parameters, e.g. model head parameters.

    Args:
        name_or_path: if supplied, specifies where to load the SFT from. If
        a valid directory path, the SFT is loaded from the local file system,
        otherwise, will attempt to load a SFT of the given name from huggingface
        models. If not supplied, an empty SFT will be created.
        version: version number on huggingface models.
        cache_dir: huggingface cache directory.
    """
    def __init__(self,
        name_or_path=None,
        version=None,
        cache_dir=None,
    ):
        if name_or_path is not None:
            if os.path.isdir(name_or_path):
                sft_dir = name_or_path
            else:
                sft_dir = pull_from_hf_model_hub(
                    name_or_path,
                    version=version,
                    cache_dir=cache_dir
                )

            sft_file = os.path.join(sft_dir, SFT_FILE_NAME)
            tensors = torch.load(sft_file)
            
            if 'diffs' in tensors:
                self.diffs = {
                    p: decode_tensor(d)
                    for p, d in tensors['diffs'].items()
                }
            else:
                self.diffs = {}

            if 'abs' in tensors:
                self.abs = {
                    p: decode_tensor(d)
                    for p, d in tensors['abs'].items()
                }
            else:
                self.abs = {}

            if 'block_sparses' in tensors:
                self.block_sparses = tensors['block_sparses']
            else:
                self.block_sparses = {}

            if not self.diffs and not self.abs and not self.block_linears:
                logger.warn(f'Empty SFT {name_or_path}')
        else:
            self.diffs = {}
            self.abs = {}
            self.block_sparses = {}

    def add_diff(self, name, tensor):
        """
        Adds a new parameter tensor to the SFT. 

        Args:
            name: the parameter name, e.g. bert.embeddings.word_embeddings.weight.
            tensor: the tensor of differences/values.
            diff: bool, if true the tensor contains the differences between the
            fine-tuned and original parameter values, otherwise it contains
            fully-specified dense values (i.e. an "abs" parameter).
        """
        self.diffs[name] = tensor.cpu()


    def add_abs(self, name, tensor):
        self.abs[name] = tensor.cpu()

    def add_block_sparse(self, name, model):
        param_path_segments = name.split('.')
        parent_module_path = '.'.join(param_path_segments[:-1])
        parent_module = model.get_submodule(parent_module_path)

        if not isinstance(parent_module, CustomBlockSparseLinear):
            raise ValueError(
                f'Parameter {name} does not belong to a BlockSparseLinear module.'
            )

        parameters = {
            f'{parent_module_path}.{n}': p.cpu()
            for n, p in parent_module.named_parameters()
        }
        buffers = {
            f'{parent_module_path}.{n}': p.cpu()
            for n, p in parent_module.named_buffers()
        }
        self.block_sparses[name] = {
            'parameters': parameters,
            'buffers': buffers,
        }
        #if diff:
        #    self.diffs[name] = tensor.to_sparse().coalesce()
        #else:
        #    self.abs[name] = tensor.to('cpu')

    def save(self, save_dir):
        encoded_diffs = {
            n: encode_tensor(p)
            for n, p in self.diffs.items()
        }
        encoded_abs = {
            n: encode_tensor(p)
            for n, p in self.abs.items()
        }
        tensors = {
            'diffs': encoded_diffs,
            'abs': encoded_abs,
            'block_sparses': self.block_sparses,
        }
        save_path = os.path.join(save_dir, SFT_FILE_NAME)
        torch.save(tensors, save_path)

    def apply(self, model, with_abs=True):
        """
        Applies SFT to a model by adding the sparse parameter differences to the
        model's parameters, and setting the value of the "abs" parameters to the
        SFT's values if with_abs is True.

        Args:
            model: an nn.Module, the model.
            with_abs: determines whether the SFT's "abs" parameters are applied.
            This should be False when applying a language SFT to a task-oriented
            model such as ...ModelForTokenClassification, because otherwise a
            crash will occur when the SFT tries to copy the values of the
            language modelling head parameters to the task model which lacks 
            these parameters.
        """
        with torch.no_grad():
            for name in self.diffs.keys():
                diff = self.diffs[name]
                tensor = model.get_parameter(name)
                if diff.device != tensor.device:
                    # Permanently copy the diff tensor to the parameter tensor's
                    # device so that future applications and reversions can
                    # be carried out faster. This is important for good
                    # performance in multi-source training.
                    diff = diff.to(tensor.device)
                    self.diffs[name] = diff
                tensor += diff

            for old_name, block_linear in self.block_sparses.items():
                logger.info(f'Applying block sparse for {old_name}.')
                replace_with_block_sparse(model, old_name)
                for new_name, data in block_linear['parameters'].items():
                    param = model.get_parameter(new_name)
                    param.data = data.to(param.device)
                for new_name, data in block_linear['buffers'].items():
                    param = model.get_buffer(new_name)
                    param.data = data.to(param.device)

            if with_abs:
                for name in self.abs.keys():
                    value = self.abs[name]
                    if value.is_sparse:
                        value = value.to_dense()
                    tensor = model.get_parameter(name)
                    if value.device != tensor.device:
                        value = value.to(tensor.device)
                        self.abs[name] = value
                    tensor.data = value

    def revert(self, model):
        """
        Removes SFT from a model by subtracting the sparse parameter
        differences.
        """
        with torch.no_grad():
            for name in self.diffs.keys():
                diff = self.diffs[name]
                tensor = model.get_parameter(name)
                if diff.device != tensor.device:
                    diff = diff.to(tensor.device)
                    self.diffs[name] = diff
                tensor -= diff

