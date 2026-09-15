"""Linear-chain CRF with forward-backward marginals.

torchcrf/allennlp's CRF classes expose the joint negative log-likelihood (for L_T,
L_I+T) and Viterbi decoding, but neither cleanly exposes the per-token marginal
distribution p(y_i = tag | x) that the paper's Consistency Voting/Alignment (CVA)
loss needs:

    L_CVA = sum_i  p(y_i | I+T) * log p(y_i | T)

computed via "forward-backward algorithm on CRF marginals" (see the ITA paper,
Sec 3.3): this requires beta (backward) scores in addition to the alpha (forward)
scores libraries like torchcrf compute internally but don't return. Implementing
the CRF here directly -- rather than bolting a marginals method onto a third-party
class -- keeps forward/backward/marginals/decode consistent with each other and
avoids depending on internals of a library not designed to expose them.

Convention: batch_first, emissions (batch, seq_len, num_tags), mask (batch, seq_len)
as a bool/byte tensor (1 = real token, 0 = padding). Follows torchcrf's API shape
so this is a drop-in replacement for it in model.py.
"""
import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _log_alpha(self, emissions, mask):
        """Forward scores. Returns alpha: (seq_len, batch, num_tags), alpha[t] =
        log-sum over all label paths ending in each tag at position t (for the
        positions where mask==1; padded positions just carry the last real
        alpha forward unchanged, so alpha[-1] is always usable via seq_ends)."""
        seq_len, batch_size, _ = emissions.shape
        alpha = [self.start_transitions + emissions[0]]  # (batch, num_tags)
        for t in range(1, seq_len):
            # (batch, num_tags_prev, 1) + (num_tags_prev, num_tags_cur) -> (batch, prev, cur)
            broadcast = alpha[-1].unsqueeze(2) + self.transitions.unsqueeze(0)
            new_alpha = torch.logsumexp(broadcast, dim=1) + emissions[t]  # (batch, num_tags)
            m = mask[t].unsqueeze(1)
            alpha.append(torch.where(m, new_alpha, alpha[-1]))
        return torch.stack(alpha, dim=0)  # (seq_len, batch, num_tags)

    def _log_beta(self, emissions, mask):
        """Backward scores. beta[t] = log-sum over all label paths from position
        t+1 to the end, for each possible tag at t. beta[T-1] (last real position
        per sequence) = end_transitions."""
        seq_len, batch_size, num_tags = emissions.shape
        seq_ends = mask.long().sum(dim=0) - 1  # (batch,) index of last real token
        beta = [None] * seq_len
        beta[seq_len - 1] = self.end_transitions.expand(batch_size, num_tags).clone()
        for t in range(seq_len - 2, -1, -1):
            # score of: being in tag i at t, moving to tag j at t+1, emitting j, then beta[t+1][j]
            broadcast = (self.transitions.unsqueeze(0)
                         + (emissions[t + 1] + beta[t + 1]).unsqueeze(1))  # (batch, i, j)
            new_beta = torch.logsumexp(broadcast, dim=2)  # (batch, num_tags) -- summed over j
            m = mask[t + 1].unsqueeze(1)
            # position t is "before the last real token" only if t+1 is still real;
            # otherwise t itself is the last real token, whose beta is end_transitions.
            is_last = (seq_ends == t).unsqueeze(1)
            beta[t] = torch.where(is_last, self.end_transitions.expand(batch_size, num_tags),
                                   torch.where(m, new_beta, beta[t + 1]))
        return torch.stack(beta, dim=0)  # (seq_len, batch, num_tags)

    def _partition(self, alpha, mask):
        seq_ends = mask.long().sum(dim=0) - 1  # (batch,)
        seq_len, batch_size, num_tags = alpha.shape
        last_alpha = alpha[seq_ends, torch.arange(batch_size, device=alpha.device)]
        return torch.logsumexp(last_alpha + self.end_transitions, dim=1)  # (batch,)

    def _score(self, emissions, tags, mask):
        seq_len, batch_size, _ = emissions.shape
        score = self.start_transitions[tags[0]] + emissions[0].gather(1, tags[0].unsqueeze(1)).squeeze(1)
        for t in range(1, seq_len):
            emit = emissions[t].gather(1, tags[t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[t - 1], tags[t]]
            score = score + (trans + emit) * mask[t].float()
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size, device=tags.device)]
        score = score + self.end_transitions[last_tags]
        return score

    def neg_log_likelihood(self, emissions, tags, mask, reduction="mean"):
        """emissions/tags/mask: batch_first (batch, seq_len[, num_tags]) -- matches
        torchcrf's call convention used elsewhere in this repo."""
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1).bool()
        alpha = self._log_alpha(emissions, mask)
        log_z = self._partition(alpha, mask)
        gold_score = self._score(emissions, tags, mask)
        nll = log_z - gold_score
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        return nll

    def marginals(self, emissions, mask):
        """Returns p(y_t = tag | x) for every position: (batch, seq_len, num_tags)."""
        emissions_t = emissions.transpose(0, 1)
        mask_t = mask.transpose(0, 1).bool()
        alpha = self._log_alpha(emissions_t, mask_t)
        beta = self._log_beta(emissions_t, mask_t)
        log_z = self._partition(alpha, mask_t)  # (batch,)
        log_marginal = alpha + beta - log_z.unsqueeze(0).unsqueeze(2)
        return log_marginal.exp().transpose(0, 1)  # (batch, seq_len, num_tags)

    def log_marginals(self, emissions, mask):
        """log p(y_t = tag | x), used directly as the log p(y|T) term in L_CVA to
        avoid a redundant log(exp(.)) round trip."""
        emissions_t = emissions.transpose(0, 1)
        mask_t = mask.transpose(0, 1).bool()
        alpha = self._log_alpha(emissions_t, mask_t)
        beta = self._log_beta(emissions_t, mask_t)
        log_z = self._partition(alpha, mask_t)
        log_marginal = alpha + beta - log_z.unsqueeze(0).unsqueeze(2)
        return log_marginal.transpose(0, 1)  # (batch, seq_len, num_tags)

    def decode(self, emissions, mask):
        """Viterbi decode. Returns a list (len=batch) of tag-index lists (one per
        real token)."""
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1).bool()
        seq_len, batch_size, num_tags = emissions.shape
        history = []
        score = self.start_transitions + emissions[0]
        for t in range(1, seq_len):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)  # (batch, prev, cur)
            best_score, best_prev = broadcast.max(dim=1)  # (batch, cur)
            new_score = best_score + emissions[t]
            m = mask[t].unsqueeze(1)
            score = torch.where(m, new_score, score)
            history.append(best_prev)

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        for b in range(batch_size):
            _, best_last_tag = (score[b] + self.end_transitions).max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[b]]):
                best_last_tag = hist[b][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list
