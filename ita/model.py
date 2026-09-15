"""ITA model: XLM-RoBERTa-large encoder (shared between the T and I+T views) +
first-subtoken gathering + a linear classifier + linear-chain CRF.

No image features anywhere in this file -- both views are plain text, differing
only in which tokens are appended after the sentence (see dataset.py). The
encoder and classifier are the SAME weights for both views (this is what makes
L_T and L_I+T comparable/consistent enough for the CVA term to make sense): one
forward pass per view, not two separate sub-models.
"""
import torch
import torch.nn as nn
from transformers import XLMRobertaModel

from crf import LinearChainCRF


class ITAModel(nn.Module):
    def __init__(self, encoder_name, num_tags, dropout=0.1, cache_dir=None):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(encoder_name, cache_dir=cache_dir)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.crf = LinearChainCRF(num_tags)
        self.num_tags = num_tags

    def _emissions(self, input_ids, attention_mask, first_subtok_positions):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)  # (batch, seq_len, hidden)
        hidden = seq.size(-1)
        idx = first_subtok_positions.unsqueeze(-1).expand(-1, -1, hidden)
        gathered = torch.gather(seq, 1, idx)  # (batch, max_words, hidden)
        return self.classifier(gathered)  # (batch, max_words, num_tags)

    def forward(self, batch, compute_loss=True):
        emissions_t = self._emissions(batch["t_input_ids"], batch["t_attention_mask"],
                                       batch["t_first_subtok_positions"])
        emissions_it = self._emissions(batch["it_input_ids"], batch["it_attention_mask"],
                                        batch["it_first_subtok_positions"])
        word_mask = batch["word_mask"]
        result = {"emissions_T": emissions_t, "emissions_IT": emissions_it}
        if not compute_loss:
            return result

        labels = batch["labels"]
        l_t = self.crf.neg_log_likelihood(emissions_t, labels, word_mask, reduction="mean")
        l_it = self.crf.neg_log_likelihood(emissions_it, labels, word_mask, reduction="mean")

        # CVA (Consistency Voting/Alignment): cross-entropy from the I+T view's
        # marginal distribution onto the T view's, i.e. L_CVA = -sum_i p(y_i|I+T) *
        # log p(y_i|T). NOTE the negative sign, which the literal spec formula
        # omitted -- see the corrected-formula note where this model is used.
        # p(y|I+T) is detached (backprop ONLY through log p(y|T), per the paper).
        with torch.no_grad():
            p_it = self.crf.marginals(emissions_it, word_mask)  # (batch, words, tags)
        log_p_t = self.crf.log_marginals(emissions_t, word_mask)  # (batch, words, tags)
        cross_entropy_per_pos = -(p_it * log_p_t).sum(dim=-1)  # (batch, words)
        mask_f = word_mask.float()
        l_cva = (cross_entropy_per_pos * mask_f).sum() / mask_f.sum().clamp(min=1)

        loss = l_t + l_it + l_cva
        result.update({"loss": loss, "L_T": l_t, "L_IT": l_it, "L_CVA": l_cva})
        return result

    def decode(self, emissions, word_mask):
        return self.crf.decode(emissions, word_mask)
