import torch.nn as nn
from transformers import ViTModel


class myViT(nn.Module):
    """Vision Transformer feature extractor, drop-in replacement for myResnet.

    Mirrors myResnet's (x, fc, att) return signature so the rest of the
    multimodal pipeline (vismap2text, image_dense_cl, cross-attention) does
    not need to know whether the backbone is a CNN or a ViT:
      - x:   global image representation (batch, hidden_size) -> used as
             visual_embeds_mean (contrastive loss branch)
      - fc:  mean-pooled patch representation (batch, hidden_size), kept for
             parity with myResnet but unused downstream
      - att: patch token grid (batch, hidden_size, num_patches) -> used as
             visual_embeds_att for the txt2img / img2txt cross-attention
    """

    def __init__(self, vit_model_name, if_fine_tune, device, cache_dir=None):
        super(myViT, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name, cache_dir=cache_dir, add_pooling_layer=False)
        self.if_fine_tune = if_fine_tune
        self.device = device
        self.hidden_size = self.vit.config.hidden_size
        self.num_patches = (self.vit.config.image_size // self.vit.config.patch_size) ** 2

    def forward(self, x):
        # x shape: batch_size * channels * image_size * image_size
        outputs = self.vit(pixel_values=x)
        sequence_output = outputs.last_hidden_state  # batch, 1 + num_patches, hidden_size

        cls_token = sequence_output[:, 0, :]  # batch, hidden_size
        patch_tokens = sequence_output[:, 1:, :]  # batch, num_patches, hidden_size

        x_global = cls_token
        fc = patch_tokens.mean(dim=1)
        att = patch_tokens.transpose(1, 2)  # batch, hidden_size, num_patches

        if not self.if_fine_tune:
            x_global = x_global.detach()
            fc = fc.detach()
            att = att.detach()

        return x_global, fc, att
