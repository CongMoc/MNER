import os
import sys
import json
import argparse
from collections import defaultdict

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

    use_bf16 = torch.cuda.is_bf16_supported()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    test_rows = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))

    tp, fp, fn = 0, 0, 0
    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)
    type_support = defaultdict(int)
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
        for etype in entity_types:
            g_t = {e for e in gold if e[0] == etype}
            p_t = {e for e in pred if e[0] == etype}
            type_tp[etype] += len(g_t & p_t)
            type_fp[etype] += len(p_t - g_t)
            type_fn[etype] += len(g_t - p_t)
            type_support[etype] += len(g_t)
        pred_lines.append(json.dumps({
            'sentence': row['sentence'], 'gold': sorted(gold), 'pred': sorted(pred),
        }, ensure_ascii=False))

    def prf(tp_, fp_, fn_):
        p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        return p_, r_, f_

    precision, recall, f1 = prf(tp, fp, fn)
    total_support = sum(type_support.values())

    # per-type exact-match (type, text) precision/recall/F1 -- same methodology as the
    # overall tp/fp/fn above, just broken down by entity type, so it reports the same
    # per-label breakdown shape as the seqeval-based CRF models' eval_results.txt.
    lines = []
    lines.append(f"{'':>12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    lines.append("")
    macro_p, macro_r, macro_f1 = [], [], []
    for etype in sorted(entity_types):
        p_t, r_t, f1_t = prf(type_tp[etype], type_fp[etype], type_fn[etype])
        sup_t = type_support[etype]
        lines.append(f"{etype:>12}{p_t:>10.4f}{r_t:>10.4f}{f1_t:>10.4f}{sup_t:>10}")
        macro_p.append(p_t); macro_r.append(r_t); macro_f1.append(f1_t)
    lines.append("")
    n_types = len(entity_types) or 1
    lines.append(f"{'micro avg':>12}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{total_support:>10}")
    lines.append(f"{'macro avg':>12}{sum(macro_p)/n_types:>10.4f}{sum(macro_r)/n_types:>10.4f}"
                 f"{sum(macro_f1)/n_types:>10.4f}{total_support:>10}")
    if total_support > 0:
        w_p = sum(prf(type_tp[t], type_fp[t], type_fn[t])[0] * type_support[t] for t in entity_types) / total_support
        w_r = sum(prf(type_tp[t], type_fp[t], type_fn[t])[1] * type_support[t] for t in entity_types) / total_support
        w_f1 = sum(prf(type_tp[t], type_fp[t], type_fn[t])[2] * type_support[t] for t in entity_types) / total_support
    else:
        w_p, w_r, w_f1 = 0.0, 0.0, 0.0
    lines.append(f"{'weighted avg':>12}{w_p:>10.4f}{w_r:>10.4f}{w_f1:>10.4f}{total_support:>10}")
    report = "\n".join(lines) + "\n"

    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'w', encoding='utf8') as f:
        f.write(report)
        f.write(f"Overall: {precision} {recall} {f1}\n")
        f.write(f"tp={tp} fp={fp} fn={fn}\n")
    with open(os.path.join(args.output_dir, 'llm_pred.jsonl'), 'w', encoding='utf8') as f:
        f.write('\n'.join(pred_lines) + '\n')

    print(report)
    print(f"Overall: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
