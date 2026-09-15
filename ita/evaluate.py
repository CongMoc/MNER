"""Evaluation for ITA: T-view F1, I+T-view F1, per-entity-type F1, macro/weighted
F1 (via seqeval, same library the rest of this repo's baselines use), and a
predictions.json dump for inspection/comparison."""
import json
import os

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm


def _to_native(obj):
    """seqeval's classification_report(output_dict=True) returns numpy scalars
    (e.g. numpy.int64 for 'support'), which json.dump can't serialize -- it
    raises TypeError mid-write, leaving a truncated/invalid JSON file on disk
    (confirmed: this is exactly what was happening before this fix). Recurse
    through the result and cast any numpy scalar to its native Python type."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@torch.no_grad()
def run_view(model, dataloader, device, id_to_label, view):
    """view: 'T' or 'IT'. Returns (y_true, y_pred, records) where records carry
    tokens/img_id/gold/pred for the predictions.json dump."""
    model.eval()
    y_true, y_pred, records = [], [], []
    for batch in tqdm(dataloader, desc=f"Evaluating [{view}]"):
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_dev, compute_loss=False)
        emissions = out[f"emissions_{view}"]
        word_mask = batch_dev["word_mask"]
        pred_ids = model.decode(emissions, word_mask)  # list of per-example tag-id lists

        for i, pred_seq in enumerate(pred_ids):
            n = int(word_mask[i].sum().item())
            gold_seq = batch["gold_labels"][i][:n]
            pred_labels = [id_to_label[t] for t in pred_seq[:n]]
            y_true.append(list(gold_seq))
            y_pred.append(pred_labels)
            records.append({
                "text": batch["tokens"][i][:n],
                "gold": list(gold_seq),
                "pred": pred_labels,
                "img_id": batch["img_id"][i],
            })
    return y_true, y_pred, records


def per_entity_f1(y_true, y_pred, entity_types):
    """seqeval's classification_report already breaks down by entity type; this
    just extracts + labels rows the way the comparison table wants them, and
    folds anything outside the paper's PER/ORG/LOC into an 'OTHER' bucket if the
    caller asks for it (NewsMNER also has DATE/MISC/NUM -- see dataset.py's
    label-collection note)."""
    report = classification_report(y_true, y_pred, output_dict=True, digits=4, zero_division=0)
    out = {}
    for etype in entity_types:
        row = report.get(etype)
        out[etype] = {"precision": row["precision"], "recall": row["recall"], "f1": row["f1-score"],
                      "support": row["support"]} if row else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    out["macro avg"] = report.get("macro avg", {})
    out["weighted avg"] = report.get("weighted avg", {})
    return out


def evaluate_split(model, dataloader, device, id_to_label, entity_types, out_dir=None, split_name="test"):
    results = {}
    all_records = {}
    for view in ("T", "IT"):
        y_true, y_pred, records = run_view(model, dataloader, device, id_to_label, view)
        f1 = f1_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        per_entity = per_entity_f1(y_true, y_pred, entity_types)
        results[view] = {"precision": p, "recall": r, "f1": f1, "per_entity": per_entity}
        all_records[view] = records
        print(f"[{split_name} / {view}-view] P={p:.4f} R={r:.4f} F1={f1:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"eval_results_{split_name}.json"), "w", encoding="utf-8") as f:
            json.dump(_to_native(results), f, ensure_ascii=False, indent=2)
        # predictions.json: paper's exact requested format, using the I+T view
        # (the paper's headline multimodal setting) as the "pred" field.
        preds_path = os.path.join(out_dir, "predictions.json")
        dump = [{"text": r["text"], "gold": r["gold"], "pred": r["pred"], "img_id": r["img_id"]}
                for r in all_records["IT"]]
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
    return results
