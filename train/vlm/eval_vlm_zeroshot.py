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
    use_bf16 = torch.cuda.is_bf16_supported()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map="auto")
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    test_rows = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))

    tp, fp, fn = 0, 0, 0
    # Per-type counts, so eval_results.txt can report a breakdown (precision/recall/f1/support
    # per entity type) like every other model's eval_results.txt in this repo -- the aggregate-only
    # "Overall" line this used to write couldn't show e.g. "PER is fine but MISC is weak".
    per_type_tp = {t: 0 for t in entity_types}
    per_type_fp = {t: 0 for t in entity_types}
    per_type_fn = {t: 0 for t in entity_types}
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
        for etype in entity_types:
            gold_t = {e for e in gold if e[0] == etype}
            pred_t = {e for e in pred if e[0] == etype}
            per_type_tp[etype] += len(gold_t & pred_t)
            per_type_fp[etype] += len(pred_t - gold_t)
            per_type_fn[etype] += len(gold_t - pred_t)
        pred_lines.append(json.dumps({
            'sentence': row['sentence'], 'image': row['image'], 'gold': sorted(gold), 'pred': sorted(pred),
        }, ensure_ascii=False))

    def prf(t, p, f):
        prec = t / (t + p) if (t + p) > 0 else 0.0
        rec = t / (t + f) if (t + f) > 0 else 0.0
        f1_ = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1_

    precision, recall, f1 = prf(tp, fp, fn)

    per_type_scores = {}
    for etype in entity_types:
        p_t, r_t, f_t = prf(per_type_tp[etype], per_type_fp[etype], per_type_fn[etype])
        support = per_type_tp[etype] + per_type_fn[etype]
        per_type_scores[etype] = (p_t, r_t, f_t, support)

    total_support = sum(s[3] for s in per_type_scores.values())
    macro_p = sum(s[0] for s in per_type_scores.values()) / len(entity_types) if entity_types else 0.0
    macro_r = sum(s[1] for s in per_type_scores.values()) / len(entity_types) if entity_types else 0.0
    macro_f1 = sum(s[2] for s in per_type_scores.values()) / len(entity_types) if entity_types else 0.0
    weighted_p = sum(s[0] * s[3] for s in per_type_scores.values()) / total_support if total_support else 0.0
    weighted_r = sum(s[1] * s[3] for s in per_type_scores.values()) / total_support if total_support else 0.0
    weighted_f1 = sum(s[2] * s[3] for s in per_type_scores.values()) / total_support if total_support else 0.0

    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'w', encoding='utf8') as f:
        f.write(f"{'':>12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n\n")
        for etype in entity_types:
            p_t, r_t, f_t, support = per_type_scores[etype]
            f.write(f"{etype:>12}{p_t:>10.4f}{r_t:>10.4f}{f_t:>10.4f}{support:>10d}\n")
        f.write("\n")
        f.write(f"{'micro avg':>12}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{total_support:>10d}\n")
        f.write(f"{'macro avg':>12}{macro_p:>10.4f}{macro_r:>10.4f}{macro_f1:>10.4f}{total_support:>10d}\n")
        f.write(f"{'weighted avg':>12}{weighted_p:>10.4f}{weighted_r:>10.4f}{weighted_f1:>10.4f}{total_support:>10d}\n\n")
        f.write(f"Overall: {precision} {recall} {f1}\n")
        f.write(f"tp={tp} fp={fp} fn={fn}\n")
    with open(os.path.join(args.output_dir, 'vlm_pred.jsonl'), 'w', encoding='utf8') as f:
        f.write('\n'.join(pred_lines) + '\n')

    print(f"Overall: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    for etype in entity_types:
        p_t, r_t, f_t, support = per_type_scores[etype]
        print(f"  {etype}: precision={p_t:.4f} recall={r_t:.4f} f1={f_t:.4f} support={support}")
