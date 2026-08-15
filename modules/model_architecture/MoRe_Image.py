import torch
from torch import nn
from modules.model_architecture.common import RobertaPreTrainedModel, RobertaModel
from modules.model_architecture.torchcrf import CRF


class MoRe_Image(RobertaPreTrainedModel):
    """MoRe-Image: same text encoder/input as MoRe-Text (XLM-R-large over the
    sentence + retrieved external context, single view, no dual-branch/CL/
    PixelCNN machinery), plus a real image (ViT CLS token) fused into every
    token's representation before the CRF classifier. Image features are
    encoded outside this module (see myViT in modules/vit/vit_utils.py) and
    passed in as img_global -- keeps this class free of vision-backbone deps,
    mirroring how UMT_PixelCNN_ViT consumes precomputed ViT output.
    """

    def __init__(self, config, num_labels_, vis_hidden_size=1024):
        super(MoRe_Image, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        self.fusion_proj = nn.Linear(config.hidden_size + vis_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)

    def forward(self, input_ids, segment_ids, input_mask, img_global, labels=None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]  # batch, seq, hidden

        seq_len = sequence_output.size(1)
        img_expanded = img_global.unsqueeze(1).expand(-1, seq_len, -1)  # batch, seq, vis_hidden
        fused = torch.cat([sequence_output, img_expanded], dim=-1)  # batch, seq, hidden+vis_hidden
        fused = self.dropout(torch.relu(self.fusion_proj(fused)))

        logits = self.classifier(fused)

        if labels is not None:
            loss = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(logits, mask=input_mask.byte())
