import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from modules.model_architecture.torchcrf import CRF


class BERT_CRF_TextOnly(BertPreTrainedModel):
    """Plain text-only NER: BERT-family encoder + linear classifier + CRF, no visual branch.

    Mirrors XLMR_CRF_TextOnly but wraps transformers.BertModel instead of RobertaModel --
    loading a BERT checkpoint (e.g. bert-base-multilingual-cased) into a RobertaModel would
    silently fail to match state_dict keys ('bert.*' vs 'roberta.*'), so this needs its own
    class rather than passing --bert_model through the XLM-R/RoBERTa trainer.
    """

    def __init__(self, config, num_labels_):
        super(BERT_CRF_TextOnly, self).__init__(config)
        self.num_labels = num_labels_
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        features = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = self.dropout(features["last_hidden_state"])
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(logits, mask=input_mask.byte())
