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
"""UMT (Yu et al. 2020, "Improving Multimodal Named Entity Recognition via Entity
Span Detection with Unified Multimodal Transformer") -- the ORIGINAL cross-modal
co-attention architecture (txt2img / img2txt / txt2txt attention + visual gate +
auxiliary entity-span CRF), unchanged from UMT.py, except the two backbones are
swapped for the ones used elsewhere in this project:
  - text encoder: any RobertaModel-compatible checkpoint (originally BERT) -- pass
    xlm-roberta-large as --bert_model to train_umt_xlmr_vit.py, no code change needed
    here since RobertaModel/RobertaPreTrainedModel are already backbone-agnostic.
  - visual backbone: ViT instead of ResNet-152 -- the only architectural
    consequence is that the visual branch is no longer a fixed 2048-dim / 7x7=49
    grid, so vismap2text's input dim and the number of visual "tokens" used for
    txt2img/img2txt attention are constructor parameters (vis_hidden_size,
    num_img_tokens) instead of the hardcoded 2048/49 in UMT.py. Pass
    vis_hidden_size=1024, num_img_tokens=196 for ViT-Large/16 @ 224px (14x14
    patches, no CLS token) via modules/vit/vit_utils.py's myViT, which already
    returns visual_embeds_att in the same (batch, hidden, num_patches) layout
    ResNet's spatial feature map has, so the .view(...).permute(...) call below
    is unchanged in shape logic, just parametrized.
No PixelCNN / contrastive-loss additions here -- this is the plain paper
architecture, unlike UMT_PixelCNN_ViT.py.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math

from torchcrf import CRF

import torch
from torch import nn

logger = logging.getLogger(__name__)


def gelu(x):
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

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + s2_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
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


class UMT_XLMR_ViT(RobertaPreTrainedModel):
    """Original UMT co-attention architecture, text/vision backbones parametrized
    (see module docstring): vis_hidden_size/num_img_tokens replace UMT.py's
    hardcoded 2048/49 so the visual branch can be a ViT's patch tokens instead of a
    ResNet's 7x7 spatial feature map."""

    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1, num_labels_=2, auxnum_labels=2,
                 vis_hidden_size=1024, num_img_tokens=196):
        super(UMT_XLMR_ViT, self).__init__(config)
        self.num_labels = num_labels_
        self.vis_hidden_size = vis_hidden_size
        self.num_img_tokens = num_img_tokens
        self.roberta = RobertaModel(config)
        self.self_attention = RobertaSelfEncoder(config)
        self.self_attention_v2 = RobertaSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(vis_hidden_size, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(vis_hidden_size, config.hidden_size)
        self.txt2img_attention = RobertaCrossEncoder(config, layer_num1)
        self.img2txt_attention = RobertaCrossEncoder(config, layer_num2)
        self.txt2txt_attention = RobertaCrossEncoder(config, layer_num3)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels_)
        self.aux_classifier = nn.Linear(config.hidden_size, auxnum_labels)

        self.crf = CRF(num_labels_, batch_first=True)
        self.aux_crf = CRF(auxnum_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_att, trans_matrix,
                labels=None, auxlabels=None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)

        extended_txt_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype)
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0
        aux_addon_sequence_encoder = self.self_attention(sequence_output, extended_txt_mask)

        aux_addon_sequence_output = aux_addon_sequence_encoder[-1]
        aux_addon_sequence_output = aux_addon_sequence_output[0]
        aux_bert_feats = self.aux_classifier(aux_addon_sequence_output)
        trans_bert_feats = torch.matmul(aux_bert_feats, trans_matrix.float())

        main_addon_sequence_encoder = self.self_attention_v2(sequence_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]
        main_addon_sequence_output = main_addon_sequence_output[0]
        vis_embed_map = visual_embeds_att.view(-1, self.vis_hidden_size, self.num_img_tokens).permute(0, 2, 1)
        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # batch, num_img_tokens, hidden_dim

        img_mask = added_attention_mask[:, :self.num_img_tokens]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]

        converted_vis_embed_map_v2 = self.vismap2text_v2(vis_embed_map)

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]
        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]

        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)

        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
        bert_feats = self.classifier(final_output)

        alpha = 0.5
        final_bert_feats = torch.add(torch.mul(bert_feats, alpha), torch.mul(trans_bert_feats, 1 - alpha))

        if labels is not None:
            beta = 0.5
            aux_loss = - self.aux_crf(aux_bert_feats, auxlabels, mask=input_mask.byte(), reduction='mean')
            main_loss = - self.crf(final_bert_feats, labels, mask=input_mask.byte(), reduction='mean')
            loss = main_loss + beta * aux_loss
            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats, mask=input_mask.byte())
            return pred_tags
