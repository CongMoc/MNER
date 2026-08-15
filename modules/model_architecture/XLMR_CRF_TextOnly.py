import torch
from torch import nn
from modules.model_architecture.common import RobertaPreTrainedModel, RobertaModel
from modules.model_architecture.torchcrf import CRF


class XLMR_CRF_TextOnly(RobertaPreTrainedModel):
    """Plain text-only NER: encoder + linear classifier + CRF, no visual branch."""

    def __init__(self, config, num_labels_):
        super(XLMR_CRF_TextOnly, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = self.dropout(features["last_hidden_state"])
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(logits, mask=input_mask.byte())
