"""NewsMNER data loading for ITA, with the paper's first-subtoken labeling scheme
(Sec A.5): every word is tokenized into 1+ XLM-R subtokens, but only the FIRST
subtoken's hidden state is fed to the CRF -- there is no "X" label and no padding
label glued onto the continuation subtokens (that per-subtoken-label approach is
what UMT/train_umt_newsmner.py does, and what the paper calls out as the wrong way
to do it). Concretely: input_ids carries every subtoken (so XLM-R still sees the
whole word), while a separate `first_subtok_positions` index selects, after
encoding, exactly one hidden state per word for the CRF to run over -- so the CRF's
sequence length is "number of words", not "number of subtokens".

Two views share the same tokenized sentence prefix:
  - T view:   <s> word_1 ... word_n </s>
  - I+T view: <s> word_1 ... word_n </s> LA </s> GA </s> OCA </s>
Labels/first_subtok_positions only ever cover the sentence words -- the appended
visual-context tokens (LA/GA/OCA) are never given labels; they exist purely so the
encoder can condition the sentence's representations on them (same "context is
extra input text, not something the CRF predicts over" idea as MoRe's setup).
"""
import json
import os

import torch
from torch.utils.data import Dataset


def read_base_conll(path):
    """Same base format as data_export/newsmner_local_eval/*.txt: one optional
    IMGID line, then tab-separated token/label lines, blank line = sentence break."""
    sentences = []
    img_id = None
    tokens, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("IMGID:") or line.startswith("IMID:"):
                key = "IMGID:" if line.startswith("IMGID:") else "IMID:"
                img_id = line.strip().split(key, 1)[1].strip()
                continue
            if line.strip() == "":
                if tokens:
                    sentences.append({"img_id": img_id, "tokens": tokens, "labels": labels})
                tokens, labels = [], []
                img_id = None
                continue
            stripped = line.rstrip("\n").rstrip("\r")
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            if label in ("B-OTHER", "B-ORTHER"):
                label = "B-MISC"
            elif label in ("I-OTHER", "I-ORTHER"):
                label = "I-MISC"
            tokens.append(token)
            labels.append(label)
    if tokens:
        sentences.append({"img_id": img_id, "tokens": tokens, "labels": labels})
    return sentences


def collect_label_list(paths):
    """Derive the label set from the data itself rather than hardcoding it --
    NewsMNER has 6 entity types (DATE/LOC/MISC/NUM/ORG/PER), not just the
    PER/ORG/LOC the paper's own example schema shows; hardcoding a fixed schema
    here would silently mis-handle labels it doesn't know about (this is exactly
    the bug class train_umt_newsmner.py's trans_matrix had to be fixed for)."""
    labels = set()
    for p in paths:
        for sent in read_base_conll(p):
            labels.update(sent["labels"])
    return sorted(labels)


class ITAExample:
    __slots__ = ("tokens", "labels", "img_id")

    def __init__(self, tokens, labels, img_id):
        self.tokens = tokens
        self.labels = labels
        self.img_id = img_id


def load_examples(path):
    return [ITAExample(s["tokens"], s["labels"], s["img_id"]) for s in read_base_conll(path)]


def load_visual_contexts(path):
    """visual_contexts.json: {img_id: {"la": ..., "ga": ..., "oca": ...}}"""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


class ITADataset(Dataset):
    """Produces both the T-view and I+T-view encodings for every example so a
    single batch can compute L_T, L_I+T, and L_CVA together."""

    def __init__(self, examples, tokenizer, label_to_id, max_seq_length=512,
                 visual_contexts=None, max_words=None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_seq_length = max_seq_length
        self.visual_contexts = visual_contexts or {}
        # Cap on words/example across the whole dataset, for fixed-size label tensors.
        self.max_words = max_words or max((len(e.tokens) for e in examples), default=1)

    def __len__(self):
        return len(self.examples)

    def _encode_words(self, words):
        """Tokenize a list of words, return (subtoken_ids, first_subtok_offsets)
        where offsets[i] is the position (within this word-only subtoken run, no
        specials yet) of word i's first subtoken."""
        subtoken_ids, offsets = [], []
        for w in words:
            offsets.append(len(subtoken_ids))
            pieces = self.tokenizer.tokenize(w) or [self.tokenizer.unk_token]
            subtoken_ids.extend(self.tokenizer.convert_tokens_to_ids(pieces))
        return subtoken_ids, offsets

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tok = self.tokenizer
        cls_id, sep_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

        sent_subtok_ids, sent_offsets = self._encode_words(ex.tokens)
        # +2 for <s>/</s>; first_subtok_positions shift by 1 for the leading <s>.
        n_words = len(ex.tokens)
        label_ids = [self.label_to_id[l] for l in ex.labels]

        def build_view(extra_subtok_ids):
            ids = [cls_id] + sent_subtok_ids + [sep_id] + extra_subtok_ids
            if not ids or ids[-1] != sep_id:
                ids = ids + [sep_id]
            ids = ids[: self.max_seq_length]
            positions = [1 + off for off in sent_offsets if 1 + off < len(ids)]
            attn = [1] * len(ids)
            pad_len = self.max_seq_length - len(ids)
            ids = ids + [pad_id] * pad_len
            attn = attn + [0] * pad_len
            return ids, attn, positions

        t_ids, t_attn, t_pos = build_view([])

        vc = self.visual_contexts.get(ex.img_id, {}) if ex.img_id else {}
        extra_parts = [vc.get("la", ""), vc.get("ga", ""), vc.get("oca", "")]
        extra_ids = []
        for i, part in enumerate(extra_parts):
            if part:
                extra_ids.extend(tok.convert_tokens_to_ids(tok.tokenize(part)))
            if i < len(extra_parts) - 1:
                extra_ids.append(sep_id)
        it_ids, it_attn, it_pos = build_view(extra_ids)

        # Word-level tensors (labels + which positions in each view are "real"
        # words, since truncation can drop trailing words if the view got long).
        n_kept = min(len(t_pos), len(it_pos), n_words)
        word_mask = [1] * n_kept + [0] * (self.max_words - n_kept)
        labels_padded = label_ids[:n_kept] + [0] * (self.max_words - n_kept)
        t_pos_padded = t_pos[:n_kept] + [0] * (self.max_words - n_kept)
        it_pos_padded = it_pos[:n_kept] + [0] * (self.max_words - n_kept)

        return {
            "t_input_ids": torch.tensor(t_ids, dtype=torch.long),
            "t_attention_mask": torch.tensor(t_attn, dtype=torch.long),
            "t_first_subtok_positions": torch.tensor(t_pos_padded, dtype=torch.long),
            "it_input_ids": torch.tensor(it_ids, dtype=torch.long),
            "it_attention_mask": torch.tensor(it_attn, dtype=torch.long),
            "it_first_subtok_positions": torch.tensor(it_pos_padded, dtype=torch.long),
            "word_mask": torch.tensor(word_mask, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "img_id": ex.img_id or "",
            "tokens": ex.tokens,
            "gold_labels": ex.labels,
        }


def collate_ita(batch):
    out = {}
    tensor_keys = ["t_input_ids", "t_attention_mask", "t_first_subtok_positions",
                   "it_input_ids", "it_attention_mask", "it_first_subtok_positions",
                   "word_mask", "labels"]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["img_id"] = [b["img_id"] for b in batch]
    out["tokens"] = [b["tokens"] for b in batch]
    out["gold_labels"] = [b["gold_labels"] for b in batch]
    return out
