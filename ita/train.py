"""ITA training loop: L = L_T + L_I+T + L_CVA (see model.py for the CVA sign
correction note). Supports a single seeded run or averaging over --num_seeds
runs (the paper reports mean F1 over 5 seeds).

Usage:
  python3 train.py --config configs/default.yaml [--frac 0.1] [--num_seeds 1]
"""
import argparse
import copy
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ITADataset, collate_ita, collect_label_list, load_examples, load_visual_contexts
from evaluate import evaluate_split
from model import ITAModel

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subsample_train(examples, frac, seed):
    if frac >= 1.0:
        return examples
    rng = random.Random(seed)
    with_entity = [e for e in examples if any(l != "O" for l in e.labels)]
    without_entity = [e for e in examples if all(l == "O" for l in e.labels)]
    n_e = max(1, round(len(with_entity) * frac)) if with_entity else 0
    n_ne = max(1, round(len(without_entity) * frac)) if without_entity else 0
    subset = rng.sample(with_entity, min(n_e, len(with_entity))) + \
             rng.sample(without_entity, min(n_ne, len(without_entity)))
    rng.shuffle(subset)
    return subset


def build_dataloaders(cfg, tokenizer, label_to_id, frac, seed):
    data_dir = cfg["data"]["data_dir"]
    vc = load_visual_contexts(cfg["data"]["visual_contexts_path"])
    train_ex = subsample_train(load_examples(os.path.join(data_dir, "train.txt")), frac, seed)
    dev_ex = load_examples(os.path.join(data_dir, "dev.txt"))
    test_ex = load_examples(os.path.join(data_dir, "test.txt"))
    max_words = max(len(e.tokens) for e in train_ex + dev_ex + test_ex)

    max_seq_length = cfg["data"]["max_seq_length"]
    train_ds = ITADataset(train_ex, tokenizer, label_to_id, max_seq_length, vc, max_words)
    dev_ds = ITADataset(dev_ex, tokenizer, label_to_id, max_seq_length, vc, max_words)
    test_ds = ITADataset(test_ex, tokenizer, label_to_id, max_seq_length, vc, max_words)

    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["train_batch_size"],
                          shuffle=True, collate_fn=collate_ita)
    dev_dl = DataLoader(dev_ds, batch_size=cfg["training"]["eval_batch_size"],
                        shuffle=False, collate_fn=collate_ita)
    test_dl = DataLoader(test_ds, batch_size=cfg["training"]["eval_batch_size"],
                         shuffle=False, collate_fn=collate_ita)
    return train_dl, dev_dl, test_dl


def run_one_seed(cfg, seed, frac, out_dir, device, entity_types, label_to_id, id_to_label):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"], cache_dir=cfg["model"]["cache_dir"])
    train_dl, dev_dl, test_dl = build_dataloaders(cfg, tokenizer, label_to_id, frac, seed)

    model = ITAModel(cfg["model"]["encoder_name"], len(label_to_id) , cfg["model"]["dropout"],
                      cache_dir=cfg["model"]["cache_dir"]).to(device)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    encoder_params = list(model.encoder.named_parameters()) + list(model.classifier.named_parameters())
    crf_params = list(model.crf.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         "lr": cfg["training"]["lr_encoder"], "weight_decay": cfg["training"]["weight_decay"]},
        {"params": [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         "lr": cfg["training"]["lr_encoder"], "weight_decay": 0.0},
        {"params": [p for _, p in crf_params], "lr": cfg["training"]["lr_crf"], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    num_epochs = cfg["training"]["num_epochs"]
    steps_per_epoch = -(-len(train_dl.dataset) // cfg["training"]["train_batch_size"])  # ceil
    total_steps = int(steps_per_epoch / grad_accum * num_epochs)
    warmup_steps = int(total_steps * cfg["training"]["warmup_proportion"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_dev_f1, best_state = -1.0, None
    for epoch in range(int(num_epochs)):
        model.train()
        optimizer.zero_grad()
        running = {"loss": 0.0, "L_T": 0.0, "L_IT": 0.0, "L_CVA": 0.0}
        for step, batch in enumerate(tqdm(train_dl, desc=f"seed={seed} epoch={epoch}")):
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch_dev, compute_loss=True)
            loss = out["loss"] / grad_accum
            loss.backward()
            for k in running:
                running[k] += out[k].item() if k != "loss" else loss.item() * grad_accum
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % 20 == 0:
                logger.info("  step=%d loss=%.4f L_T=%.4f L_IT=%.4f L_CVA=%.4f",
                            step + 1, out["loss"].item(), out["L_T"].item(),
                            out["L_IT"].item(), out["L_CVA"].item())
        n_batches = len(train_dl)
        logger.info("epoch=%d loss=%.4f L_T=%.4f L_IT=%.4f L_CVA=%.4f",
                    epoch, running["loss"] / n_batches, running["L_T"] / n_batches,
                    running["L_IT"] / n_batches, running["L_CVA"] / n_batches)

        dev_results = evaluate_split(model, dev_dl, device, id_to_label, entity_types)
        dev_f1 = dev_results["IT"]["f1"]  # paper's headline I+T-view F1 selects the checkpoint
        if dev_f1 >= best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    seed_out_dir = os.path.join(out_dir, f"seed{seed}")
    test_results = evaluate_split(model, test_dl, device, id_to_label, entity_types,
                                  out_dir=seed_out_dir, split_name="test")
    return test_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--num_seeds", type=int, default=None, help="override config's num_seeds")
    ap.add_argument("--seed", type=int, default=None, help="run exactly this one seed only")
    ap.add_argument("--num_epochs", type=float, default=None, help="override config's num_epochs (e.g. for a quick sanity check)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = cfg["data"]["data_dir"]
    label_list = collect_label_list([os.path.join(data_dir, s) for s in ("train.txt", "dev.txt", "test.txt")])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    entity_types = sorted({l[2:] for l in label_list if l.startswith(("B-", "I-"))})
    logger.info("labels (%d): %s", len(label_list), label_list)
    logger.info("entity types: %s", entity_types)

    out_dir = cfg["output"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.seed is not None:
        seeds = [args.seed]
    else:
        base_seed = cfg["training"]["seed"]
        n_seeds = args.num_seeds if args.num_seeds is not None else cfg["training"]["num_seeds"]
        seeds = [base_seed + i for i in range(n_seeds)]

    all_results = []
    for seed in seeds:
        logger.info("=== seed=%d ===", seed)
        res = run_one_seed(cfg, seed, args.frac, out_dir, device, entity_types, label_to_id, id_to_label)
        all_results.append(res)

    if len(all_results) > 1:
        for view in ("T", "IT"):
            f1s = [r[view]["f1"] for r in all_results]
            logger.info("[%d-seed average, %s-view] F1 mean=%.4f std=%.4f (%s)",
                        len(f1s), view, float(np.mean(f1s)), float(np.std(f1s)), f1s)


if __name__ == "__main__":
    main()
