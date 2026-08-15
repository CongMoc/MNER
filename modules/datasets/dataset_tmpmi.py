import os
import random

import torch
from torchvision import transforms

from modules.datasets.dataset_roberta_main import image_process


class TMPMIFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, img_feats, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.img_feats = img_feats
        self.label_id = label_id


def convert_tmpmi_examples_to_features(examples, label_list, max_seq_length, tokenizer, crop_size, path_img,
                                        num_images=4, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
                                        random_seed=37):
    """Each example gets num_images image slots: slot 0 is its own IMGID-matched
    image (background.jpg if missing), the rest are other real images picked at
    random (fixed once via random_seed, same convention as the random-image
    ablation elsewhere in this repo).
    """
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    image_pool = sorted(f for f in os.listdir(path_img) if f != 'background.jpg')
    rng = random.Random(random_seed)

    transform = transforms.Compose([
        transforms.Resize([crop_size, crop_size]),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])

    features = []
    for example in examples:
        textlist = example.text_a.split(' ')
        labellist = example.label
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

        own_path = os.path.join(path_img, example.img_id)
        if not os.path.exists(own_path):
            own_path = os.path.join(path_img, 'background.jpg')
        img_paths = [own_path] + [os.path.join(path_img, rng.choice(image_pool)) for _ in range(num_images - 1)]

        img_feats = []
        for p in img_paths:
            try:
                img_feats.append(image_process(p, transform))
            except Exception:
                img_feats.append(image_process(os.path.join(path_img, 'background.jpg'), transform))
        img_feats = torch.stack(img_feats)

        features.append(TMPMIFeatures(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
            img_feats=img_feats, label_id=label_ids,
        ))
    return features
