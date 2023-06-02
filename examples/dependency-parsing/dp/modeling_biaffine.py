from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, BertModel, BertPreTrainedModel, PreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, load_tf_weights_in_bert
from transformers.utils.generic import ModelOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DependencyParsingOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    rel_preds: Optional[torch.FloatTensor] = None
    arc_preds: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Credit:
# Class taken from https://github.com/yzhangcs/biaffine-parser
class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.init_weights()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def init_weights(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        return s


def BiaffineParser(_config):

    class _BiaffineParser(PreTrainedModel):
        """
        Credit: G. Glavaš & I. Vulić
        Based on paper "Is Supervised Syntactic Parsing Beneficial for Language Understanding? An Empirical Investigation"
        (https://arxiv.org/pdf/2008.06788.pdf)
        """
        config_class = type(_config)
        base_model_prefix = _config.model_type

        def __init__(self, config):
            super().__init__(config)
            setattr(
                self,
                self.base_model_prefix,
                AutoModel.from_config(config)
            )

            self.biaffine_arcs = Biaffine(n_in=config.hidden_size, bias_x=True, bias_y=False)
            self.biaffine_rels = Biaffine(n_in=config.hidden_size, n_out=config.num_labels, bias_x=True, bias_y=True)

            dropout_prob = config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else config.dropout
            self.dropout = nn.Dropout(dropout_prob)
            self.loss_fn = CrossEntropyLoss()

        def _init_weights(self, module):
            if hasattr(module, 'init_weights'):
                module.init_weights()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels_arcs=None,
            labels_rels=None,
            word_starts=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            # run through BERT encoder and get vector representations
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                #token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            reps = self.dropout(outputs[0])
            word_outputs_deps = self._merge_subword_tokens(reps, word_starts)

            # adding the CLS representation as the representation for the "root" parse token
            #word_outputs_heads = torch.cat([outputs[1].unsqueeze(1), word_outputs_deps], dim=1)
            word_outputs_heads = torch.cat([reps[:, 0, ...].unsqueeze(1), word_outputs_deps], dim=1)

            arc_preds = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
            arc_preds = arc_preds.squeeze()

            rel_preds = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
            rel_preds = rel_preds.permute(0, 2, 3, 1)

            if len(arc_preds.shape) == 2:
                arc_preds = arc_preds.unsqueeze(0)
            
            mask = labels_arcs.ne(self.config.pad_token_id)
            arc_scores, arcs = arc_preds[mask], labels_arcs[mask]
            loss = self.loss_fn(arc_scores, arcs)

            rel_scores, rels = rel_preds[mask], labels_rels[mask]
            rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
            rel_loss = self.loss_fn(rel_scores, rels)
            loss += rel_loss

            if not return_dict:
                return loss, rel_preds, arc_preds

            return DependencyParsingOutput(
                loss=loss,
                rel_preds=rel_preds,
                arc_preds=arc_preds,
                logits=(rel_scores, arc_scores),
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def _merge_subword_tokens(self, subword_outputs, word_starts):
            instances = []
            max_seq_length = subword_outputs.shape[1]

            # handling instance by instance
            for i in range(len(subword_outputs)):
                subword_vecs = subword_outputs[i]
                word_vecs = []
                starts = word_starts[i]
                mask = starts.ne(self.config.pad_token_id)
                starts = starts[mask]
                for j in range(len(starts) - 1):
                    if starts[j + 1] <= 0:
                        break

                    start = starts[j]
                    end = starts[j + 1]
                    vecs_range = subword_vecs[start:end]
                    word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

                instances.append(word_vecs)

            t_insts = []
            zero_tens = torch.zeros(self.config.hidden_size).unsqueeze(0)
            zero_tens = zero_tens.to("cuda" if torch.cuda.is_available() else "cpu")

            for inst in instances:
                if len(inst) < max_seq_length:
                    for i in range(max_seq_length - len(inst)):
                        inst.append(zero_tens)
                t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

            w_tens = torch.cat(t_insts, dim=0)
            return w_tens

    return _BiaffineParser
