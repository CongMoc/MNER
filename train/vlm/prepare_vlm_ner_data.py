import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llm_text_only')))
from modules.datasets.dataset_roberta_main import MNERProcessor
from prepare_llm_ner_data import bio_to_entities


def convert_split(examples, out_path):
    n_entities = 0
    with open(out_path, 'w', encoding='utf8') as f:
        for ex in examples:
            tokens = ex.text_a.split(' ')
            entities = bio_to_entities(tokens, ex.label)
            n_entities += len(entities)
            f.write(json.dumps({
                'sentence': ex.text_a,
                'entities': entities,
                'image': ex.img_id,
            }, ensure_ascii=False) + '\n')
    print(f'{out_path}: {len(examples)} sentences, {n_entities} entities')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='dir with train.txt/dev.txt/test.txt in IMGID+BIO format')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--labels', required=True, help='comma-separated base label list (no X/<s>/</s>)')
    args = ap.parse_args()

    os.environ['LABELS'] = args.labels
    processor = MNERProcessor()
    os.makedirs(args.out_dir, exist_ok=True)

    convert_split(processor.get_train_examples(args.data_dir), os.path.join(args.out_dir, 'train.jsonl'))
    convert_split(processor.get_dev_examples(args.data_dir), os.path.join(args.out_dir, 'dev.jsonl'))
    convert_split(processor.get_test_examples(args.data_dir), os.path.join(args.out_dir, 'test.jsonl'))
