import copy
import inspect
import logging
import re

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def factorize_embeddings(full_embeddings, factorized_embeddings):
    d = factorized_embeddings.size(1)
    logger.info(f'Projecting to rank {d}')
    U, S, V = torch.linalg.svd(full_embeddings)
    newU = U[:, :d]
    newS = torch.diag(S[:d])
    newV = V[:d, :]
    logger.info(f'Full embeddings: {full_embeddings.size()}')
    logger.info(f'Factorized embeddings: {factorized_embeddings.size()}')
    logger.info(f'U: {U.size()}')
    logger.info(f'S: {S.size()}')
    logger.info(f'V: {V.size()}')
    logger.info(f'newU: {newU.size()}')
    logger.info(f'newS: {newS.size()}')
    logger.info(f'newV: {newV.size()}')
    factorized_embeddings.data = newU #@ torch.sqrt(newS) #@ newV


def compute_downprojection(E, d):
    U, S, Vt = torch.linalg.svd(E)
    return Vt[:d, :].transpose(0, 1)


def compute_per_layer_representation_mappings(model, tokenizer, trainer_args, dataset, d):
    trainer = Trainer(
        model=model,
        args=trainer_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
    )
    logger.info(f'model is a {type(model)}')
    model.eval()
    hidden_states = [[] for _ in range(model.config.num_hidden_layers + 1)]
    with torch.no_grad():
        for batch in tqdm(
            trainer.get_eval_dataloader(),
            desc='Evaluating states',
        ):
            batch = trainer._prepare_inputs(batch)
            #logger.info(sorted(list(batch.keys())))
            outputs = model(**batch, output_hidden_states=True)
            for layer_states, saved_states in zip(outputs.hidden_states, hidden_states):
                input_ids = batch['input_ids']
                isnt_padding = input_ids != tokenizer.pad_token_id
                seq_lengths = torch.sum(isnt_padding, dim=1).to(torch.float32)
                random_indices = (torch.rand(input_ids.size(0), device=seq_lengths.device) * seq_lengths).to(input_ids.dtype)
                flattened_indices = random_indices + input_ids.size(1) * torch.arange(input_ids.size(0), device=random_indices.device)
                layer_states = layer_states.view(-1, layer_states.size(-1))
                saved_states.append(layer_states[flattened_indices, :])
        hidden_states = [
            torch.concat(h, dim=0).detach().cpu()
            for h in hidden_states
        ]
        return [
            compute_downprojection(h, d) for h in hidden_states
        ]


def compress_model(
    teacher_model,
    auto_model_type,
    layer_reduction_factor=1,
    hidden_size_reduction_factor=1,
    layer_initialization_strategy='regular',
):
    teacher_config = teacher_model.config
    assert teacher_config.num_hidden_layers % layer_reduction_factor == 0
    student_config = copy.deepcopy(teacher_config)
    student_config.num_hidden_layers //= layer_reduction_factor
    student_config.hidden_size //= hidden_size_reduction_factor
    student_config.intermediate_size //= hidden_size_reduction_factor
    if hasattr(student_config, 'hidden_dropout_prob'):
        student_config.hidden_dropout_prob = 0.0
    if hasattr(student_config, 'attention_probs_dropout_prob'):
        student_config.attention_probs_dropout_prob = 0.0
    student_model = auto_model_type.from_config(student_config)

    #if hidden_size_reduction_factor != 1:
    #    mapper = torch.normal(
    #        0.0,
    #        teacher_model.config.initializer_range,
    #        [teacher_model.config.hidden_size, student_model.config.hidden_size]
    #    )
    #    if torch.cuda.is_available():
    #        teacher_model.to('cuda:0')
    #        student_model.to('cuda:0')
    #        mapper = mapper.to('cuda:0')
    #else:
    #    mapper = None

    #if hidden_size_reduction_factor == 1:
    def param_mapper(teacher_param_name):
        match = re.search(r'layer\.(\d+)\.', teacher_param_name)
        if match is None:
            return teacher_param_name
        layer_num = int(match.group(1))
        if layer_initialization_strategy == 'regular':
            if layer_num % layer_reduction_factor == layer_reduction_factor - 1:
                student_layer = layer_num // layer_reduction_factor
                return teacher_param_name.replace(
                    match.group(0),
                    f'layer.{student_layer}.'
                )
        elif layer_initialization_strategy == 'bottom':
            if layer_num < student_config.num_hidden_layers:
                return teacher_param_name
        else:
            raise RuntimeError(f'Invalid layer mapping strategy "{layer_initialization_strategy}"')

        return None

    with torch.no_grad():
        for teacher_param_name, teacher_param in teacher_model.named_parameters():
            student_param_name = param_mapper(teacher_param_name)
            student_display_name = student_param_name if student_param_name else ''
            logger.info(f'Mapping {teacher_param_name} -> {student_display_name}')
            if student_param_name:
                student_param = student_model.get_parameter(student_param_name)
                if student_param is student_model.get_input_embeddings().weight:
                    logger.info('Factorizing teacher embeddings')
                    factorize_embeddings(teacher_param, student_param)
                    #student_param.requires_grad = False
                #else:
                #    indices = []
                #    for teacher_dim, student_dim in zip(teacher_param.size(), student_param.size()):
                #        reduction_factor = teacher_dim // student_dim
                #        indices.append(list(range(0, teacher_dim, reduction_factor)))
                #    student_param.data = teacher_param[np.ix_(*indices)].clone()
                    #if hidden_size_reduction_factor == 1:
                    #    student_param.data = teacher_param.data.clone()
                    #elif 'embeddings' in teacher_param_name:
                    #    student_param.data = torch.matmul(teacher_param, mapper)
                    #elif tuple(teacher_param.size()) == 2 * (teacher_config.hidden_size,):
                    #    logger.info('Solving least squares problem.')
                    #    student_param.data = torch.linalg.lstsq(
                    #        mapper, 
                    #        torch.matmul(teacher_param.transpose(0, 1), mapper)
                    #    )[0].transpose(0, 1)

        output_embeddings = student_model.get_output_embeddings()
        if output_embeddings:
            output_embeddings.weight = student_model.get_input_embeddings().weight

    for n, p in student_model.named_parameters():
        logger.info(f'{n}: {p.size()}')

    return student_model


class TeacherStudentModel(nn.Module):

    def __init__(
        self,
        teacher_model,
        student_model,
        alpha_ce=0.0,
        alpha_task=0.0,
        alpha_hidden=0.0,
        alpha_attn=0.0,
        softmax_temperature=2.0,
        mask_logits=True,
        eval_bypass_teacher=False,
        align_layers='all',
        attn_loss='mse',
        eps=1e-20,
        mapping_dataset=None,
        mapping_tokenizer=None,
        mapping_args=None,
    ):
        super().__init__()
        assert teacher_model.config.num_hidden_layers % student_model.config.num_hidden_layers == 0

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.student_model = student_model
        self.config = self.student_model.config

        if self.teacher_model.config.hidden_size != self.student_model.config.hidden_size:
            self.layer_mappings = nn.ParameterList(
                [
                    nn.Parameter(m, requires_grad=False)
                    for m in compute_per_layer_representation_mappings(
                        teacher_model,
                        mapping_tokenizer,
                        mapping_args,
                        mapping_dataset,
                        self.student_model.config.hidden_size
                    )
                ]
            )
        else:
            self.layer_mappings = None

        self.alpha_ce = alpha_ce
        self.alpha_task = alpha_task
        self.alpha_hidden = alpha_hidden
        self.alpha_attn = alpha_attn
        self.softmax_temperature = softmax_temperature
        self.mask_logits = mask_logits
        self.eval_bypass_teacher = eval_bypass_teacher
        self.align_layers = align_layers
        self.eps = eps
        self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.hidden_loss_fct = nn.MSELoss()
        if attn_loss == 'mse':
            self.attn_loss_fct = nn.MSELoss()
        elif attn_loss == 'kl':
            self.attn_loss_fct = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f'Unsupported attention loss type {attn_loss}')

        signature = inspect.signature(self.teacher_model.forward)
        #self.labels_required = 'labels' in signature.parameters
        self.signature_columns = set(signature.parameters.keys())

    def train(self, mode=True):
        # Always keep teacher model locked in eval mode.
        self.training = mode
        self.student_model.train(mode=mode)

    def soft_cross_entropy(self, predictions, targets):
        return -torch.mean(torch.sum(
            F.log_softmax(predictions / self.softmax_temperature, dim=-1) *
            F.softmax(targets / self.softmax_temperature, dim=-1),
            dim=-1
        ))
        #return self.ce_loss_fct(
        #    F.log_softmax(predictions / self.softmax_temperature, dim=-1),
        #    F.softmax(targets / self.softmax_temperature, dim=-1)
        #) #* self.softmax_temperature ** 2

    def forward(self, *args, **kwargs):
        return_dict = kwargs.get('return_dict', None)
        output_attentions = kwargs.get('output_attentions', None)
        output_hidden_states = kwargs.get('output_hidden_states', None)
        #if not self.training:
        #    logger.info(sorted(list(kwargs.keys())))
        kwargs = {k: v for k, v in kwargs.items() if k in self.signature_columns}
        #if not self.training:
        #    logger.info(sorted(list(kwargs.keys())))
        if not self.training and self.eval_bypass_teacher: # and ('labels' in kwargs or 'start_positions' in kwargs):
            #logger.info('Bypassing student-teacher evaluation')
            #assert 'labels' not in kwargs
            return self.student_model(*args, **kwargs)

        kwargs['output_attentions'] = True
        kwargs['output_hidden_states'] = True
        kwargs['return_dict'] = True

        #if not self.labels_required:
        #    kwargs.pop('labels', None)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(*args, **kwargs)
        student_outputs = self.student_model(*args, **kwargs)

        loss = 0.0

        if self.alpha_attn != 0.0:
            attn_loss = 0.0
            teacher_attns = teacher_outputs.attentions
            student_attns = student_outputs.attentions
            assert len(teacher_attns) % len(student_attns) == 0
            factor = len(teacher_attns) // len(student_attns)
            if self.align_layers == 'all':
                student_attns = list(enumerate(student_attns))
            else:
                student_attns = [(len(student_attns) - 1, student_attns[-1])]
            for i, student_attn in student_attns:
                student_attn = student_attn.view(-1)
                teacher_attn = teacher_attns[(i+1) * factor - 1]
                teacher_attn = teacher_attn.view(-1)
                is_padded = teacher_attn == 0.0
                student_attn = student_attn[~is_padded]
                teacher_attn = teacher_attn[~is_padded]
                attn_loss += self.attn_loss_fct(student_attn, teacher_attn)
            loss += self.alpha_attn * attn_loss / len(student_attns)

        if self.alpha_hidden != 0.0:
            hidden_loss = 0.0
            teacher_reps = teacher_outputs.hidden_states
            student_reps = student_outputs.hidden_states
            assert (len(teacher_reps) - 1) % (len(student_reps) - 1) == 0
            factor = (len(teacher_reps) - 1) // (len(student_reps) - 1)
            if self.align_layers == 'all':
                student_reps = list(enumerate(student_reps))
            else:
                student_reps = [(len(student_reps) - 1, student_reps[-1])]
            for i, student_rep in student_reps:
                teacher_rep = teacher_reps[i * factor]
                if self.layer_mappings is not None:
                    with torch.no_grad():
                        teacher_rep = torch.matmul(teacher_rep, self.layer_mappings[i * factor])
                #if i == 0:
                #    logger.info(f'student_rep: {student_rep}')
                #    logger.info(f'teacher_rep: {teacher_rep}')
                hidden_loss += self.hidden_loss_fct(student_rep, teacher_rep)
            loss += self.alpha_hidden * hidden_loss / len(student_reps)

        if self.alpha_task != 0.0:
            loss += self.alpha_task * student_outputs.loss

        if self.alpha_ce != 0.0:
            any_logits = False
            for logits_output in ['logits', 'start_logits', 'end_logits']:
                if not hasattr(teacher_outputs, logits_output):
                    continue
                any_logits = True
                teacher_logits = getattr(teacher_outputs, logits_output)
                student_logits = getattr(student_outputs, logits_output)
                if not isinstance(teacher_logits, tuple):
                    teacher_logits = (teacher_logits,)
                    student_logits = (student_logits,)
                
                for t_logits, s_logits in zip(teacher_logits, student_logits):
                    #logger.info(f't_logits: {t_logits}')
                    #logger.info(f's_logits: {s_logits}')
                    if self.mask_logits:
                        mask = (kwargs['labels'] > -1).unsqueeze(-1).expand_as(s_logits)
                        t_logits = torch.masked_select(t_logits, mask).view(-1, s_logits.size(-1))
                        s_logits = torch.masked_select(s_logits, mask).view(-1, s_logits.size(-1))
                        assert t_logits.size() == s_logits.size() 
                    #logger.info(f't_logits: {t_logits}')
                    #logger.info(f's_logits: {s_logits}')
                    loss += self.alpha_ce * self.soft_cross_entropy(s_logits, t_logits)

                    #loss += (
                    #    self.ce_loss_fct(
                    #        F.log_softmax(s_logits/ self.softmax_temperature, dim=-1),
                    #        F.softmax(t_logits/ self.softmax_temperature, dim=-1),
                    #    )
                    #    * (self.softmax_temperature) ** 2
                    #)

            assert any_logits

        if not output_attentions:
            student_outputs.attentions = None
        else:
            assert False
        if not output_hidden_states:
            student_outputs.hidden_states = None
        else:
            assert False

        #if not self.training:
        #    logger.info(student_outputs)
        
        if not return_dict:
            return (loss,) + student_outputs[1:]

        student_outputs.loss = loss
        return student_outputs


