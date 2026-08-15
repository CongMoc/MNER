import torch.nn.functional as F
import random
import time
from transformers import RobertaConfig
from modules.model_architecture.common import RobertaPreTrainedModel, RobertaModel, ImageDecoder, RobertaSelfEncoder,RobertaCrossEncoder, LOSS_TI
import numpy as np
import torch
from torch import nn
from modules.model_architecture.torchcrf import CRF


class UMT_PixelCNN_ViT(RobertaPreTrainedModel):
    """Same architecture as UMT_PixelCNN, generalized so the visual backbone's
    hidden size and patch-token count are not hardcoded to ResNet-152's
    2048-dim / 7x7=49 grid — lets the vision branch be a ViT instead
    (e.g. vis_hidden_size=1024, num_img_tokens=196 for ViT-Large/16 @ 224px),
    and the text backbone be any Roberta-architecture model (e.g. XLM-R large)
    since config.hidden_size is already read dynamically.
    """
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1, num_labels_=2, auxnum_labels=2,
                 vis_hidden_size=2048, num_img_tokens=49):
        super(UMT_PixelCNN_ViT, self).__init__(config)
        self.num_labels = num_labels_
        self.auxnum_labels = auxnum_labels
        self.vis_hidden_size = vis_hidden_size
        self.num_img_tokens = num_img_tokens
        self.roberta = RobertaModel(config)
        self.image_decoder = ImageDecoder(nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                                 resnet_nonlinearity='concat_elu', input_channels=3, h_dim=config.hidden_size)
        self.self_attention = RobertaSelfEncoder(config)
        self.self_attention_v2 = RobertaSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.vismap2text = nn.Linear(vis_hidden_size, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(vis_hidden_size, config.hidden_size)
        self.txt2img_attention = RobertaCrossEncoder(config, layer_num1)
        self.text_dense_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.text_ouput_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.image_dense_cl = nn.Linear(vis_hidden_size, config.hidden_size)
        self.image_output_cl = nn.Linear(config.hidden_size, config.hidden_size)

        self.img2txt_attention = RobertaCrossEncoder(config, layer_num2)
        self.txt2txt_attention = RobertaCrossEncoder(config, layer_num3)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.aux_classifier = nn.Linear(config.hidden_size, self.auxnum_labels)

        self.crf = CRF(self.num_labels, batch_first=True)
        self.aux_crf = CRF(self.auxnum_labels, batch_first=True)


    def text_toimage_loss(self,text_h1, image_h1, temp):
        batch_size = text_h1.shape[0]
        loss = 0
        for i in range(batch_size):
            up = torch.exp(
                (torch.matmul(text_h1[i], image_h1[i]) / (torch.norm(text_h1[i]) * torch.norm(image_h1[i]))) / temp
            )

            down = torch.sum(
                torch.exp((torch.sum(text_h1[i] * image_h1, dim=-1) / (
                            torch.norm(text_h1[i]) * torch.norm(image_h1, dim=1))) / temp), dim=-1)

            loss += -torch.log(up / down)

        return loss

    def image_totext_loss(self,text_h1, image_h1, temp):
        batch_size = text_h1.shape[0]
        loss = 0
        for i in range(batch_size):
            up = torch.exp(
                (
                        torch.matmul(image_h1[i], text_h1[i]) / (torch.norm(image_h1[i]) * torch.norm(text_h1[i]))
                ) / temp
            )

            down = torch.sum(
                torch.exp((torch.sum(image_h1[i] * text_h1, dim=-1) / (
                            torch.norm(image_h1[i]) * torch.norm(text_h1, dim=1))) / temp), dim=-1)

            loss += -torch.log(up / down)

        return loss

    def total_loss(self,text_h1, image_h1, temp, temp_lamb):
        lamb = temp_lamb
        batch_size = text_h1.shape[0]
        loss = (1 / batch_size) * (
                    lamb * self.text_toimage_loss(text_h1, image_h1, temp) + (1 - lamb) * self.image_totext_loss(text_h1, image_h1, temp))
        return loss

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_mean, visual_embeds_att, trans_matrix,
                image_decode=None, alpha=None, beta=None, theta=None, sigma=None, temp=None, temp_lamb=None, labels=None, auxlabels=None):

        # Get the emission scores from the BiLSTM
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)  # batch_size * seq_len * hidden_size
        sequence_output = features["last_hidden_state"]
        pooler_output = features["pooler_output"]
        sequence_output = self.dropout(sequence_output)
        pooler_output = self.linear(pooler_output)


        extended_txt_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0
        aux_addon_sequence_encoder = self.self_attention(sequence_output, extended_txt_mask)

        aux_addon_sequence_output = aux_addon_sequence_encoder[-1]
        aux_addon_sequence_output = aux_addon_sequence_output[0]
        aux_bert_feats = self.aux_classifier(aux_addon_sequence_output)
        trans_bert_feats = torch.matmul(aux_bert_feats, trans_matrix.float())

        main_addon_sequence_encoder = self.self_attention_v2(sequence_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]
        main_addon_sequence_output = main_addon_sequence_output[0]
        vis_embed_map = visual_embeds_att.view(-1, self.vis_hidden_size, self.num_img_tokens).permute(0, 2, 1)  # self.batch_size, num_img_tokens, 2048
        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, num_img_tokens, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :self.num_img_tokens]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # apply img2txt attention mechanism to obtain multimodal-based text representations
        converted_vis_embed_map_v2 = self.vismap2text_v2(vis_embed_map)  # self.batch_size, num_img_tokens, hidden_dim

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * num_img_tokens * hidden_dim
        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # visual gate
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)

        # direct concatenation
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        bert_feats = self.classifier(final_output)


        final_bert_feats = torch.add(torch.mul(bert_feats, alpha),torch.mul(trans_bert_feats, 1-alpha))

        if labels is not None:

            # Loss 1
            aux_loss = - self.aux_crf(aux_bert_feats, auxlabels, mask=input_mask.byte(), reduction='mean')
            # Loss 2
            main_loss = - self.crf(final_bert_feats, labels, mask=input_mask.byte(), reduction='mean')
            # Loss 3
            image_generate = self.image_decoder(x=image_decode, h=pooler_output)
            assert torch.isfinite(image_generate).all(), "image_generate has nan"
            loss_ti = LOSS_TI(image_decode, image_generate)
            # Loss 4
            text_output_cl = self.text_ouput_cl(self.relu(self.text_dense_cl(pooler_output)))
            image_ouput_cl = self.image_output_cl(self.relu(self.image_dense_cl(visual_embeds_mean)))
            cl_loss = self.total_loss(text_output_cl, image_ouput_cl, temp, temp_lamb)
            loss = main_loss + theta * cl_loss + beta*aux_loss + loss_ti*sigma
            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats, mask=input_mask.byte())
            return pred_tags
