import os
import sys
import json
import argparse

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from vlm_ner_common import load_jsonl, build_system_prompt, build_messages, parse_entities


def load_image(image_dir, image_name):
    path = os.path.join(image_dir, image_name)
    if not os.path.exists(path):
        path = os.path.join(image_dir, 'background.jpg')
    return Image.open(path).convert('RGB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--adapter_dir', default=None, help='optional LoRA adapter dir; omit for pure zero-shot')
    ap.add_argument('--data_dir', required=True, help='dir with test.jsonl from prepare_vlm_ner_data.py')
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    entity_types = args.labels.split(',')
    system_prompt = build_system_prompt(entity_types)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    test_rows = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))

    tp, fp, fn = 0, 0, 0
    pred_lines = []
    for row in tqdm(test_rows, desc='Evaluating'):
        gold = set(tuple(e) for e in row['entities'])
        image = load_image(args.image_dir, row['image'])
        messages = build_messages(system_prompt, row['sentence'], image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen_ids = out[0][inputs['input_ids'].shape[1]:]
        gen_text = processor.decode(gen_ids, skip_special_tokens=True)
        pred = set(parse_entities(gen_text, entity_types))

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        pred_lines.append(json.dumps({
            'sentence': row['sentence'], 'image': row['image'], 'gold': sorted(gold), 'pred': sorted(pred),
        }, ensure_ascii=False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'w', encoding='utf8') as f:
        f.write(f"Overall: {precision} {recall} {f1}\n")
        f.write(f"tp={tp} fp={fp} fn={fn}\n")
    with open(os.path.join(args.output_dir, 'vlm_pred.jsonl'), 'w', encoding='utf8') as f:
        f.write('\n'.join(pred_lines) + '\n')

    print(f"Overall: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
