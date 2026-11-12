# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
from torchcrf import CRF

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaPreTrainedModel, RobertaOutput, RobertaSelfOutput, RobertaIntermediate

class RobertaSelfEncoder(nn.Module):
    def __init__(self, config):
        super(RobertaSelfEncoder, self).__init__()
        layer = RobertaLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class RobertaCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(RobertaCrossEncoder, self).__init__()
        layer = RobertaCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class RobertaCoAttention(nn.Module):
    def __init__(self, config):
        super(RobertaCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):

        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class RobertaCrossAttention(nn.Module):
    def __init__(self, config):
        super(RobertaCrossAttention, self).__init__()
        self.self = RobertaCoAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class RobertaCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(RobertaCrossAttentionLayer, self).__init__()
        self.attention = RobertaCrossAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class UMT(RobertaPreTrainedModel):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels_=2, auxnum_labels=2):
        super(UMT, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        self.self_attention = RobertaSelfEncoder(config)
        self.self_attention_v2 = RobertaSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(2048, config.hidden_size)
        self.txt2img_attention = RobertaCrossEncoder(config, layer_num1)
        self.img2txt_attention = RobertaCrossEncoder(config, layer_num2)
        self.txt2txt_attention = RobertaCrossEncoder(config, layer_num3)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        ### self.self_attention = BertLastSelfAttention(config)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels_)
        self.aux_classifier = nn.Linear(config.hidden_size, auxnum_labels)

        self.crf = CRF(num_labels_, batch_first=True)
        self.aux_crf = CRF(auxnum_labels, batch_first=True)

        # trans_matrix used to be a forward() argument (set below), but a plain tensor argument
        # is not what nn.DataParallel expects: DataParallel.scatter() splits every positional
        # tensor arg along dim 0 as if it were the batch, which silently corrupts this constant
        # (auxnum_labels, num_labels) matrix across multi-GPU replicas (e.g. shape (7, 17) -> a
        # wrong (4, 17) slice on one GPU) -- "RuntimeError: mat1 and mat2 shapes cannot be
        # multiplied". A registered buffer is replicated correctly per device instead.
        # persistent=False: this is rebuilt deterministically from label_list/auxlabel_list by
        # the training script (see build_trans_matrix() in train_umt_newsmner.py) and reassigned
        # right after construction, so it does not need to be saved in/loaded from checkpoints.
        self.register_buffer(
            "trans_matrix", torch.zeros(auxnum_labels, num_labels_, dtype=torch.float), persistent=False)

        self.init_weights()

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_att,
                labels=None, auxlabels=None):
        # Get the emission scores from the BiLSTM
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)  # batch_size * seq_len * hidden_size
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)

        extended_txt_mask = input_mask.unsqueeze(1).unsqueeze(2)
        # sequence_output.dtype, not next(self.parameters()).dtype: under nn.DataParallel,
        # calling self.parameters() inside forward() on a replica raises
        # "StopIteration: Caught StopIteration in replica N on device N" with current
        # torch/transformers versions. sequence_output is a real tensor already in scope with
        # the same dtype the model is actually computing in (also correct under autocast/fp16,
        # unlike a hardcoded torch.float32).
        extended_txt_mask = extended_txt_mask.to(dtype=sequence_output.dtype)
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0
        aux_addon_sequence_encoder = self.self_attention(sequence_output, extended_txt_mask)

        aux_addon_sequence_output = aux_addon_sequence_encoder[-1]
        aux_addon_sequence_output = aux_addon_sequence_output[0]
        aux_bert_feats = self.aux_classifier(aux_addon_sequence_output)
        #######aux_bert_feats = self.aux_classifier(sequence_output)
        trans_bert_feats = torch.matmul(aux_bert_feats, self.trans_matrix.float())

        main_addon_sequence_encoder = self.self_attention_v2(sequence_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]
        main_addon_sequence_output = main_addon_sequence_output[0]
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:,:49]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=sequence_output.dtype)  # see note above
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # apply img2txt attention mechanism to obtain multimodal-based text representations
        converted_vis_embed_map_v2 = self.vismap2text_v2(vis_embed_map)  # self.batch_size, 49, hidden_dim

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * 49 * hidden_dim
        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        ##cross_final_txt_encoder = self.txt2txt_attention(aux_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim
        #cross_final_txt_layer = torch.add(cross_final_txt_layer, sequence_output)

        # visual gate
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)
        # reverse_gate_value = torch.neg(gate_value).add(1)
        # gated_converted_att_vis_embed = torch.add(torch.mul(reverse_gate_value, cross_final_txt_layer),
                                        # torch.mul(gate_value, cross_output_layer))

        # direct concatenation
        # gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
        ###### final_output = self.dropout(final_output)
        #middle_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
        #final_output = torch.cat((sequence_output, middle_output), dim=-1)

        ###### addon_sequence_output = self.self_attention(final_output, extended_txt_mask)
        bert_feats = self.classifier(final_output)

        alpha = 0.5
        final_bert_feats = torch.add(torch.mul(bert_feats, alpha),torch.mul(trans_bert_feats, 1-alpha))

        # suggested by Hongjie
        #bert_feats = F.log_softmax(bert_feats, dim=-1)

        if labels is not None:
            beta = 0.5  # 73.87(73.50) 85.37(85.00) 0.5 5e-5 #73.45 85.05 1.0 1 1 1 4e-5 # 73.63 0.1 1 1 1 5e-5 # old 0.1 2 1 1 85.23 0.2 1 1 85.04
            ##beta = 0.6
            aux_loss = - self.aux_crf(aux_bert_feats, auxlabels, mask=input_mask.byte(), reduction='mean')
            main_loss = - self.crf(final_bert_feats, labels, mask=input_mask.byte(), reduction='mean')
            loss = main_loss + beta*aux_loss
            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats, mask=input_mask.byte())
            return pred_tags

if __name__ == "__main__": 
    model = UMT.from_pretrained('vinai/phobert-base-v2',cache_dir='cache')
    print(model)