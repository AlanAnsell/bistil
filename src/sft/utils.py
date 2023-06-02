import io
import logging
import random
import subprocess
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
#import block_linear_cpp
#from pytorch_block_sparse import BlockSparseLinear, BlockSparseMatrix
#from pytorch_block_sparse.block_sparse import BlockSparseMatrixBase
#from pytorch_block_sparse.block_sparse_linear import BlockSparseLinearFunction

from transformers import DataCollatorForLanguageModeling

#block_linear_cpp = load(name='block_linear_cpp', sources=['../../../blockify/block_linear/block_linear.cpp'])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FFStateAnalyser(nn.Module):

    def __init__(self, module, dimension):
        super().__init__()
        self.module = module
        self.dimension = dimension
        self.count = 0
        self.n_activations = torch.zeros([dimension], dtype=torch.long)

    def forward(self, *args, **kwargs):
        result = self.module(*args, **kwargs)
        with torch.no_grad():
            flattened = result.view(-1, self.dimension)
            self.count += flattened.size(0)
            active = (flattened > 0).to(dtype=torch.long)
            if self.n_activations.device != active.device:
                self.n_activations = self.n_activations.to(active.device)
            self.n_activations += torch.sum(active, 0)
        return result

    def activation_probability(self):
        return self.n_activations / self.count

    def prune(self, threshold):
        assert isinstance(self.module, nn.Linear)


def apply_ff_state_analysis(model, module_name):
    module = model.get_submodule(module_name)
    if not isinstance(module, nn.Linear):
        raise ValueError(f'{module_name} must be an nn.Linear module')
    module_path = module_name.split('.')
    parent_module_name = '.'.join(module_path[:-1])
    parent_module = model.get_submodule(parent_module_name)
    state_analyser = FFStateAnalyser(module, module.out_features)
    setattr(
        parent_module,
        module_path[-1],
        state_analyser
    )

def remove_ff_state_analysis(model, module_name):
    module_path = module_name.split('.')
    parent_module_name = '.'.join(module_path[:-1])
    module = model.get_submodule(module_name)
    parent_module = model.get_submodule(parent_module_name)
    setattr(
        parent_module,
        module_path[-1],
        module.module
    )

def prune_ff_neurons(model, layer1_name, layer2_name, threshold):
    linear1 = model.get_submodule(layer1_name)
    assert isinstance(linear1, FFStateAnalyser)
    activation_probs = linear1.activation_probability()
    linear1 = linear1.module
    linear2 = model.get_submodule(layer2_name)
    assert isinstance(linear2, nn.Linear)
     
    neuron_mask = activation_probs > threshold
    linear1.weight.data = linear1.weight[neuron_mask, :]
    linear1.weight.grad = None
    linear1.bias.data = linear1.bias[neuron_mask]
    linear1.bias.grad = None
    linear2.weight.data = linear2.weight[:, neuron_mask]
    linear2.weight.grad = None
    return (~neuron_mask).sum()


class AttentionStateAnalyser(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.head_mask = nn.Parameter(
            torch.ones(
                [1, self.module.self.num_attention_heads, 1, 1]
            )
        )
        self.head_importance = torch.zeros([self.module.self.num_attention_heads])

        def update_importance(grad):
            #logger.info(f'update_importance was called with grad {grad}.')
            #logger.info(f'head_mask grad = {self.head_mask.grad}.')
            #logger.info(f'head_mask = {self.head_mask}.')
            #logger.info(f'grad_fn = {self.head_mask.grad_fn}.')
            if self.head_importance.device != grad.device:
                self.head_importance = self.head_importance.to(grad.device)
            self.head_importance += grad.view(-1).abs().detach()
            return grad

        self.head_mask.register_hook(update_importance)

    def forward(self, *args, **kwargs):
        #self.head_mask = self.head_mask.to(args[0].device)
        if len(args) >= 3:
            args = list(args)
            args[2] = self.head_mask
        else:
            kwargs['head_mask'] = self.head_mask
        return self.module(*args, **kwargs)


def apply_attn_state_analysis(model, module_name):
    module = model.get_submodule(module_name)
    module_path = module_name.split('.')
    parent_module_name = '.'.join(module_path[:-1])
    parent_module = model.get_submodule(parent_module_name)
    state_analyser = AttentionStateAnalyser(module)
    setattr(
        parent_module,
        module_path[-1],
        state_analyser
    )

def remove_attn_state_analysis(model, module_name):
    module_path = module_name.split('.')
    parent_module_name = '.'.join(module_path[:-1])
    module = model.get_submodule(module_name)
    assert isinstance(module, AttentionStateAnalyser)
    parent_module = model.get_submodule(parent_module_name)
    setattr(
        parent_module,
        module_path[-1],
        module.module
    )

def prune_attn_heads(model, module_name, threshold):
    module = model.get_submodule(module_name)
    assert isinstance(module, AttentionStateAnalyser)
    heads_to_prune = torch.argwhere(module.head_importance < threshold)
    module.module.prune_heads(heads_to_prune)
    return heads_to_prune.size(0)


class DataCollatorWithConsistentEvalMasking(DataCollatorForLanguageModeling):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_generator = torch.Generator()
        self._generator = self._train_generator

    def train(self):
        self._generator = self._train_generator

    def eval(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(2147483647)
        logger.info('Setting data_collator into eval mode')

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=self._generator).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8), generator=self._generator).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5), generator=self._generator).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self._generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def density(model, include_head=True):
    non_zeros = 0
    n_params = 0
    for n, p in model.named_parameters():
        if include_head or n.startswith(model.base_model_prefix):
            non_zeros += torch.sum(p != 0.0)
            n_params += p.numel()
    return non_zeros / n_params


class SparseLinear(nn.Module):

    def __init__(self, dense_linear):
        super().__init__()
        sparse_csc = dense_linear.weight.data.to_sparse_csc()
        self.weight = nn.Parameter(sparse_csc)
        #self.weight = nn.Parameter(torch.sparse_csr_tensor(
        #    sparse_csr.crow_indices().to(dtype=torch.int32),
        #    sparse_csr.col_indices().to(dtype=torch.int32),
        #    sparse_csr.values(),
        #    size=sparse_csr.size(),
        #    device=sparse_csr.device,
        #    requires_grad=sparse_csr.requires_grad,
        #))
        self.bias = dense_linear.bias

    def __call__(self, x):
        x_shape = x.size()
        #print(f'x: {x_shape}, W: {self.weight.size()}')
        #x = x.view(-1, x_shape[-1])
        #x = x.transpose(0, 1)
        #x = torch.matmul(self.weight, x)
        #x = x.transpose(0, 1)
        #x = x.view(*x_shape[:-1], self.weight.size(0))
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

def sparsify(model):
    for parent_module in list(model.modules()):
        for name, child_module in list(parent_module.named_children()):
            if isinstance(child_module, nn.Linear):
                print(name)
                setattr(
                    parent_module,
                    name,
                    SparseLinear(child_module)
                )


class Block(nn.Module):

    def __init__(self, weight, inputs, outputs):
        super().__init__()
        self.register_buffer('input_indices', inputs)
        self.register_buffer('output_indices', outputs)
        self.weight = nn.Parameter(weight)

    def __call__(self, x):
        x = x[..., self.input_indices]
        #return torch.matmul(x, self.weight)
        return F.linear(x, self.weight)


class BlockLinear(nn.Module):

    def __init__(self, dense_linear, block_descriptors):
        super().__init__()
        self.n_blocks = len(block_descriptors)
        self.in_features = dense_linear.in_features
        self.out_features = dense_linear.out_features
        blocks = []
        input_indices = []
        output_indices = []
        with torch.no_grad():
            full_weights = dense_linear.weight.clone()
            for inputs, outputs in block_descriptors:
                weight_submatrix = full_weights[outputs, :][:, inputs]
                #weight_submatrix = torch.transpose(weight_submatrix, 0, 1)
                blocks.append(weight_submatrix)
                input_indices.append(inputs)
                output_indices.append(outputs)
                for i in outputs:
                    for j in inputs:
                        full_weights[i, j] = 0.0
            self.weights = nn.Parameter(torch.stack(blocks, 0))
            self.register_buffer('input_indices', torch.stack(input_indices, 0))
            self.register_buffer('output_indices', torch.stack(output_indices, 0))

        if dense_linear.bias is not None:
            self.bias = dense_linear.bias
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        result = block_linear_cpp.forward(x, self.weights, self.input_indices, self.output_indices, self.out_features)
        #result = torch.zeros(x.size()[:-1] + (self.out_features,), device=x.device)
        #for block in self.blocks:
        #    block_output = block(x)
        #    indices = torch.tile(block.output_indices, x.size()[:-1] + (1,))
        #    #print(f'result: {result.size()}')
        #    #print(f'indices: {block.output_indices.size()}')
        #    #print(f'output: {block_output.size()}')
        #    result.scatter_add_(-1, indices, block_output)
        if self.bias is not None:
            result += self.bias
        return result


class CustomBlockSparseLinear(nn.Module):

    def __init__(
        self,
        weight,
        bias=None,
    ):
        super().__init__()
        self.fn = BlockSparseLinearFunction.apply

        self.weight = weight
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = self.fn(x, self.weight.get_differentiable_data(), self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


def replace_with_block_sparse(model, param_name, block_sparse_matrix=None):
    param_path_segments = param_name.split('.')
    parent_module_path = '.'.join(param_path_segments[:-1])
    parent_module = model.get_submodule(parent_module_path)

    if not isinstance(parent_module, nn.Linear):
        raise ValueError(
            f'Block sparsity was requested for parameter {param_name}, '
            'but it does not belong to a nn.Linear module.'
        )

    if block_sparse_matrix is None:
        print(
            f'Replacing {param_name} which is of size ' +
            str(tuple(model.get_parameter(param_name).size()))
        )
        block_sparse_matrix = BlockSparseMatrix.zeros(
            tuple(model.get_parameter(param_name).size())
        )
    block_sparse_linear_module = CustomBlockSparseLinear(
        block_sparse_matrix,
        bias=parent_module.bias,
    )

    grandparent_module_path = '.'.join(param_path_segments[:-2])
    grandparent_module = model.get_submodule(grandparent_module_path)

    setattr(
        grandparent_module,
        param_path_segments[-2],
        block_sparse_linear_module
    )
    #parent_module = self.model.get_submodule(parent_module_path)
    #assert parent_module.weight.data.requires_grad


def select_blocks(X, block_shape=(32, 32), density=0.25):
    assert X.dim() == 2
    n, m = block_shape
    assert X.size(0) % n == 0
    assert X.size(1) % m == 0
    
    block_rows = X.size(0) // n
    block_cols = X.size(1) // m
    block_sums = []
    for r in range(block_rows):
        for c in range(block_cols):
            block_sums.append((torch.sum(torch.abs(X[r*n:(r+1)*n, c*m:(c+1)*m])), (r, c)))
    block_sums.sort(reverse=True)

    total_blocks = len(block_sums)
    blocks_to_select = int(density * total_blocks)
    selected_blocks = block_sums[:blocks_to_select]
    selected_blocks.sort(key=lambda val_and_coor: val_and_coor[1])

    block_mask = torch.zeros([block_rows, block_cols], dtype=torch.bool, device=X.device)
    block_value = 0.0
    data = []
    for value, (r, c) in selected_blocks:
        block_value += value
        block_mask[r, c] = True
        data.append(X[r*n:(r+1)*n, c*m:(c+1)*m].transpose(0, 1))
    data = torch.concat(data, 0)

    total_value = torch.sum(torch.abs(X))
    print(f'Value: {block_value:.4f}/{total_value:.4f}')

    return BlockSparseMatrixBase(X.size(), block_mask, data, block_shape=block_shape)


class BlockSparseLinearWithPerm(nn.Module):

    def __init__(self, bsl):
        super().__init__()
        self.bsl = bsl
        self.register_buffer('in_perm', torch.randint(0, bsl.in_features, [bsl.in_features], device=bsl.weight.data.device))
        self.register_buffer('out_perm', torch.randint(0, bsl.out_features, [bsl.out_features], device=bsl.weight.data.device))

    def forward(self, x):
        x = x[..., self.in_perm].contiguous()
        x = self.bsl(x)
        return x[..., self.out_perm].contiguous()


def blockify(linear):
    #torch.save(linear.weight.data, 'test.pt')
    input_buffer = io.BytesIO()
    torch.save(torch.transpose(linear.weight.data, 0, 1), input_buffer)
    input_bytes = input_buffer.getvalue()
    input_buffer.close()
    print(f'{len(input_bytes)} bytes')
    with subprocess.Popen(['blockify', '4', '4'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
        output_bytes, _ = p.communicate(input=input_bytes)
        if p.returncode != 0:
            raise RuntimeError(f'blockify failed with exit code {p.returncode}')
    output_bytes = io.BytesIO(output_bytes)
    output = torch.load(output_bytes)

    block_linear = BlockLinear(linear, output)
    for n, p in sorted(list(block_linear.named_parameters())):
        print(f'{n}: {p.size()}')

    linear.to('cuda:0')
    sparse_weight = select_blocks(linear.weight, density=0.20)
    block_sparse_linear = CustomBlockSparseLinear(
        linear.in_features,
        linear.out_features,
        sparse_weight,
        bias=linear.bias
    )
    bsl_with_perm = BlockSparseLinearWithPerm(block_sparse_linear)

    X = torch.rand([8, 256, 768], device='cuda:0')
    #torch.set_num_threads(1)
    with torch.no_grad():
        block_sparse_res = block_sparse_linear(X)
        start_time = time.time()
        for _ in range(10000):
            block_sparse_res = block_sparse_linear(X)
        end_time = time.time()
        print(f'Block sparse matmuls took {end_time - start_time :.3f}s')
        dense_res = linear(X)
        start_time = time.time()
        for _ in range(10000):
            dense_res = linear(X)
        end_time = time.time()
        print(f'Dense matmuls took {end_time - start_time :.3f}s')
        #start_time = time.time()
        #for _ in range(100):
        #    block_res = block_linear(X)
        #end_time = time.time()
        #print(f'Block matmuls took {end_time - start_time :.3f}s')
        bslp_res = bsl_with_perm(X)
        start_time = time.time()
        for _ in range(10000):
            bslp_res = bsl_with_perm(X)
        end_time = time.time()
        print(f'Block sparse matmuls took {end_time - start_time :.3f}s')
        #print(dense_res)
        #print(block_sparse_res)

