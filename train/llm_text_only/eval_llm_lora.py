import os
import sys
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm_ner_common import load_jsonl, build_system_prompt, build_messages, parse_entities


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--adapter_dir', required=True)
    ap.add_argument('--data_dir', required=True, help='dir with test.jsonl from prepare_llm_ner_data.py')
    ap.add_argument('--labels', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    entity_types = args.labels.split(',')
    system_prompt = build_system_prompt(entity_types)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    test_rows = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))

    tp, fp, fn = 0, 0, 0
    pred_lines = []
    for row in tqdm(test_rows, desc='Evaluating'):
        gold = set(tuple(e) for e in row['entities'])
        messages = build_messages(system_prompt, row['sentence'])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        gen_text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = set(parse_entities(gen_text, entity_types))

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        pred_lines.append(json.dumps({
            'sentence': row['sentence'], 'gold': sorted(gold), 'pred': sorted(pred),
        }, ensure_ascii=False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'w', encoding='utf8') as f:
        f.write(f"Overall: {precision} {recall} {f1}\n")
        f.write(f"tp={tp} fp={fp} fn={fn}\n")
    with open(os.path.join(args.output_dir, 'llm_pred.jsonl'), 'w', encoding='utf8') as f:
        f.write('\n'.join(pred_lines) + '\n')

    print(f"Overall: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
