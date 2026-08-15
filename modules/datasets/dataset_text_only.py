class TextOnlyFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_text_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    features = []

    for example in examples:
        textlist = example.text_a.split(' ')
        labellist = example.label
        assert len(textlist) == len(labellist), (
            f"token/label count mismatch ({len(textlist)} vs {len(labellist)}) for guid={example.guid}: "
            "a token likely contains an embedded space, desyncing text_a.split(' ') from the label list"
        )
        tokens, labels = [], []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                labels.append(labellist[i] if m == 0 else "X")

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        ntokens = ["<s>"] + tokens + ["</s>"]
        segment_ids = [0] * len(ntokens)
        label_ids = [label_map["<s>"]] + [label_map[l] for l in labels] + [label_map["</s>"]]
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(TextOnlyFeatures(input_ids=input_ids, input_mask=input_mask,
                                          segment_ids=segment_ids, label_id=label_ids))
    return features
