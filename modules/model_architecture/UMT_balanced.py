"""Balanced-fusion variant of UMT (modules/model_architecture/UMT.py).

The original UMT.forward() is structurally biased toward the text branch
regardless of what the data says:
  1. final_output = cat(cross_final_txt_layer, gate * cross_output_layer) --
     the sigmoid gate can drive the image branch to ~0 but has no symmetric
     way to drive the text branch down; text is always passed through in full.
  2. final_bert_feats = alpha*bert_feats + (1-alpha)*trans_bert_feats with
     alpha hardcoded to 0.5 -- half of the final prediction always comes from
     trans_bert_feats, a purely text-only auxiliary CRF branch that never
     sees the image at all, regardless of whether the image actually helped.

UMTBalanced turns exactly those two numbers into *learned* quantities so
training itself decides the text/image split, instead of the architecture
assuming a fixed (effectively text-favored) split up front:
  1. gate_value is now interpreted as the image branch's own mixing weight:
         final_output = cat((1 - gate_value) * cross_final_txt_layer, gate_value * cross_output_layer)
     gate_value -> 0 reproduces the original "pure text" behavior; gate_value -> 1
     is "pure image-informed text" -- both directions are now reachable, so a
     gate that converges near 0 is actual evidence the image doesn't help,
     not an artifact of the architecture only being able to suppress image.
  2. alpha = sigmoid(self.alpha_logit) is a learned scalar (init 0 -> sigmoid=0.5,
     same starting point as the original hardcoded value) instead of a constant.

Everything else (encoders, CRFs, aux loss structure) is unchanged. beta (the
aux_loss weight) is kept as a constructor/CLI hyperparameter rather than
hardcoded, so it can be ablated without another model file.

self.last_gate_mean / self.last_alpha are populated on every forward() call
(single-GPU only -- no DataParallel gather handling) so the training loop can
log, epoch over epoch, how much the model is actually listening to the image.
"""
import torch
from torch import nn

from modules.model_architecture.UMT import UMT


class UMTBalanced(UMT):
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1,
                 num_labels_=2, auxnum_labels=2, aux_loss_weight=0.5):
        super().__init__(config, layer_num1, layer_num2, layer_num3, num_labels_, auxnum_labels)
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5, same init as original hardcoded alpha
        self.aux_loss_weight = aux_loss_weight
        self.last_gate_mean = None
        self.last_alpha = None

    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_att,
                labels=None, auxlabels=None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)

        # sequence_output.dtype, not next(self.parameters()).dtype: see UMT.forward() for why
        # (StopIteration under nn.DataParallel). self.trans_matrix (registered buffer in
        # UMT.__init__, inherited here), not a trans_matrix forward() argument: see UMT.__init__
        # for why (DataParallel.scatter() corrupts a non-batched constant tensor argument).
        extended_txt_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=sequence_output.dtype)
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0
        aux_addon_sequence_encoder = self.self_attention(sequence_output, extended_txt_mask)
        aux_addon_sequence_output = aux_addon_sequence_encoder[-1][0]
        aux_bert_feats = self.aux_classifier(aux_addon_sequence_output)
        trans_bert_feats = torch.matmul(aux_bert_feats, self.trans_matrix.float())

        main_addon_sequence_encoder = self.self_attention_v2(sequence_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1][0]
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        converted_vis_embed_map = self.vismap2text(vis_embed_map)

        img_mask = added_attention_mask[:, :49]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=sequence_output.dtype)
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]

        converted_vis_embed_map_v2 = self.vismap2text_v2(vis_embed_map)
        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]
        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]

        # symmetric visual gate: gate_value now mixes BOTH branches instead of only attenuating image
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch, text_len, hidden_dim
        gated_txt = torch.mul(1.0 - gate_value, cross_final_txt_layer)
        gated_vis = torch.mul(gate_value, cross_output_layer)
        final_output = torch.cat((gated_txt, gated_vis), dim=-1)

        bert_feats = self.classifier(final_output)

        alpha = torch.sigmoid(self.alpha_logit)  # learned main/aux mix, init 0.5
        final_bert_feats = torch.add(torch.mul(bert_feats, alpha), torch.mul(trans_bert_feats, 1 - alpha))

        self.last_gate_mean = gate_value.detach().mean().item()
        self.last_alpha = alpha.detach().item()

        if labels is not None:
            beta = self.aux_loss_weight
            aux_loss = - self.aux_crf(aux_bert_feats, auxlabels, mask=input_mask.byte(), reduction='mean')
            main_loss = - self.crf(final_bert_feats, labels, mask=input_mask.byte(), reduction='mean')
            loss = main_loss + beta * aux_loss
            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats, mask=input_mask.byte())
            return pred_tags
