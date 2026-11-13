"""CLIP-ViT-B/32 visual-backbone variant of UMT, built on top of UMTBalanced
(modules/model_architecture/UMT_balanced.py: symmetric gate + learned main/aux mix).

Only the visual backbone changes, from ResNet-152 (ImageNet-pretrained, 2048-dim
conv5 feature map, 7x7=49 regions) to CLIP ViT-B/32 (image-text-aligned
pretraining, 768-dim patch embeddings, also 49 regions for a 224x224 input at
patch_size=32 -> (224/32)^2 = 49). The cross-attention/gate/aux-loss machinery
downstream is untouched -- only vismap2text/vismap2text_v2 change input dim
(2048 -> 768) to match, and the ResNet-specific
`visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)` reshape is replaced by
a direct CLIP forward pass.

Unlike the ResNet pipeline (feature extraction done by a separate `encoder`
object in the training script, wrapped in torch.no_grad()), CLIP is owned by
this model and frozen via requires_grad=False on every CLIP parameter --
since neither CLIP's weights nor the raw pixel input require grad, autograd
never retains CLIP's intermediate activations for backward, so no explicit
no_grad() context is needed for the same memory savings. The training script
only has to pass raw (CLIP-normalized) pixel values where it used to pass
precomputed ResNet features.

CLIP's own normalization stats (0.4815/0.4578/0.4082 mean, 0.2686/0.2613/0.2758
std) are NOT ImageNet's (0.485/0.456/0.406 mean, 0.229/0.224/0.225 std) --
the training script must pass CLIP_IMAGE_MEAN/CLIP_IMAGE_STD (below) into
convert_mm_examples_to_features, and use a distinct dataloader cache suffix,
or it will silently train on ImageNet-normalized pixels CLIP was never
pretrained on.
"""
import torch
from torch import nn
from transformers import CLIPVisionModel

from modules.model_architecture.UMT_balanced import UMTBalanced

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class UMTClip(UMTBalanced):
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1,
                 num_labels_=2, auxnum_labels=2, aux_loss_weight=0.5,
                 clip_model_name=CLIP_MODEL_NAME):
        super().__init__(config, layer_num1, layer_num2, layer_num3,
                          num_labels_, auxnum_labels, aux_loss_weight)

        # CLIP ViT-B/32 patch embeddings are 768-dim (vs ResNet-152's 2048-dim
        # conv5 map) -- re-create the two projection layers UMT.__init__ built
        # for 2048 so their input dim actually matches what forward() feeds them.
        clip_hidden = 768
        self.vismap2text = nn.Linear(clip_hidden, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(clip_hidden, config.hidden_size)

        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

    def extract_visual_embeds(self, pixel_values):
        """pixel_values: [B,3,224,224], CLIP-normalized (see CLIP_IMAGE_MEAN/STD).
        Returns [B,49,768] patch embeddings with the CLS token dropped."""
        out = self.clip(pixel_values=pixel_values)
        return out.last_hidden_state[:, 1:, :]  # drop CLS -> 49 patches, 768-dim

    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask,
                visual_embeds_att, labels=None, auxlabels=None):
        # visual_embeds_att here is the RAW pixel tensor [B,3,224,224] -- unlike
        # UMT/UMTBalanced, where it's an already-extracted ResNet feature map,
        # CLIP extraction happens inside this forward() call (frozen, see class
        # docstring), so the training script no longer needs a separate
        # encoder() call before model(...).
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)

        # sequence_output.dtype, not next(self.parameters()).dtype, and self.trans_matrix (a
        # registered buffer, not a forward() arg): see UMT.forward()/UMT.__init__ for why --
        # both avoid nn.DataParallel bugs (StopIteration / mis-scattered constant tensor) under
        # multi-GPU training.
        extended_txt_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=sequence_output.dtype)
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0
        aux_addon_sequence_encoder = self.self_attention(sequence_output, extended_txt_mask)
        aux_addon_sequence_output = aux_addon_sequence_encoder[-1][0]
        aux_bert_feats = self.aux_classifier(aux_addon_sequence_output)
        trans_bert_feats = torch.matmul(aux_bert_feats, self.trans_matrix.float())

        main_addon_sequence_encoder = self.self_attention_v2(sequence_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1][0]

        # ── CLIP replaces the ResNet `.view(-1, 2048, 49).permute(0, 2, 1)` reshape:
        # patch embeddings already come out of CLIP as [B, 49, 768], no reshape needed.
        vis_embed_map = self.extract_visual_embeds(visual_embeds_att)  # [B, 49, 768]
        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # [B, 49, hidden_dim]

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

        # symmetric visual gate (from UMTBalanced): gate_value mixes BOTH branches
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))
        gated_txt = torch.mul(1.0 - gate_value, cross_final_txt_layer)
        gated_vis = torch.mul(gate_value, cross_output_layer)
        final_output = torch.cat((gated_txt, gated_vis), dim=-1)

        bert_feats = self.classifier(final_output)

        alpha = torch.sigmoid(self.alpha_logit)
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


# Re-exported so `from modules.model_architecture.UMT_clip import UMT` (a smoke-test
# convenience, matching the UMT_balanced.py precedent) still resolves to something --
# the original UMT class, NOT UMTClip. Import UMTClip explicitly to use this variant.
from modules.model_architecture.UMT import UMT  # noqa: E402,F401
