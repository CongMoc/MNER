import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from modules.model_architecture.TMP_MI import TMP_MI
from modules.datasets.dataset_roberta_main import MNERProcessor
from modules.datasets.dataset_tmpmi import convert_tmpmi_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.optimization import BertAdam
from seqeval.metrics import classification_report
from ner_evaluate import evaluate
from tqdm import tqdm, trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# TMP-MI (Temporal Prompt Model for Multi-Image), adapted from the official
# TPM-MI (JinFish/MNER-MI, LREC-COLING 2024): text encoder swapped BERT ->
# XLM-R-large, image encoder swapped ViT-base -> ViT-large. Each example uses
# 4 image slots: its own crawled image (or background.jpg if missing) plus 3
# other real images picked at random (fixed once via --seed).

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--path_image", required=True, type=str)
parser.add_argument("--bert_model", default='xlm-roberta-large', type=str)
parser.add_argument("--vit_model", default='google/vit-large-patch16-224-in21k', type=str)
parser.add_argument("--num_images", type=int, default=4)
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--task_name", default='sonba', type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--cache_dir", default="cache", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=2.2e-5, type=float)
parser.add_argument("--num_train_epochs", default=10.0, type=float)
parser.add_argument("--warmup_proportion", default=0.4, type=float)
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: %s n_gpu: %d", device, n_gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")
if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

processor = MNERProcessor()
label_list = processor.get_labels()
num_labels = len(label_list) + 1
label_map = {i: label for i, label in enumerate(label_list, 1)}
label_map[0] = "<pad>"
reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}

tokenizer = AutoTokenizer.from_pretrained(args.bert_model, cache_dir=args.cache_dir)

model = TMP_MI(args.bert_model, args.vit_model, num_labels_=num_labels,
               num_images=args.num_images, cache_dir=args.cache_dir)
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')


def build_dataset(examples, cache_tag):
    cache_file = os.path.join(args.data_dir, f"{cache_tag}_dataset_tmpmi.pth")
    if os.path.exists(cache_file):
        return torch.load(cache_file)
    features = convert_tmpmi_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image,
        num_images=args.num_images, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
        random_seed=args.seed)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feats for f in features])
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_img_feats, all_label_ids)
    torch.save(data, cache_file)
    return data


def run_eval(dataloader, data_dir, split_name):
    model.eval()
    y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, img_feats, label_ids = batch
        with torch.no_grad():
            pred_tags = model(input_ids, input_mask, segment_ids, img_feats, labels=None)
        label_ids_np = label_ids.to('cpu').numpy()
        input_mask_np = input_mask.to('cpu').numpy()
        for i, mask in enumerate(input_mask_np):
            t1, t2, i1, i2 = [], [], [], []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids_np[i][j]] not in ("X", "</s>"):
                        t1.append(label_map[label_ids_np[i][j]]); i1.append(label_ids_np[i][j])
                        t2.append(label_map[pred_tags[i][j]]); i2.append(pred_tags[i][j])
                else:
                    break
            y_true.append(t1); y_pred.append(t2); y_true_idx.append(i1); y_pred_idx.append(i2)

    report = classification_report(y_true, y_pred, digits=4)
    data_raw, _, _ = processor._read_sbtsv(os.path.join(data_dir, f"{split_name}.txt"))
    sentence_list = [data_raw[i][0] for i in range(len(y_pred))]
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
    logger.info("\n%s", report)
    return report, p, r, f1


if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if 'crf' not in n and not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if 'crf' not in n and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if 'crf' in n], 'lr': 1.0e-2, 'weight_decay': 0.00005},
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                          warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

    train_data = build_dataset(train_examples, "train")
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.train_batch_size)

    dev_examples = processor.get_dev_examples(args.data_dir)
    dev_data = build_dataset(dev_examples, "dev")
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.eval_batch_size)

    max_dev_f1, best_dev_epoch = 0.0, 0
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss, nb_tr_steps = 0, 0
        for batch in tqdm(train_dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, img_feats, label_ids = batch
            loss = model(input_ids, input_mask, segment_ids, img_feats, label_ids)
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f"===============Main loss: {tr_loss/nb_tr_steps}===============")

        report, p, r, f1 = run_eval(dev_dataloader, args.data_dir, "dev")
        print("Overall (dev): ", p, r, f1)

        if f1 >= max_dev_f1:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)
            max_dev_f1, best_dev_epoch = f1, train_idx
            logger.info("******************SAVE NEW MODEL WEIGHT*********************")

    logger.info("Best epoch: %s, best dev F1: %s", best_dev_epoch, max_dev_f1)

if args.do_eval:
    model.load_state_dict(torch.load(output_model_file, map_location=device))
    model.to(device)

    eval_examples = processor.get_test_examples(args.data_dir)
    eval_data = build_dataset(eval_examples, "test")
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=args.eval_batch_size)

    report, p, r, f1 = run_eval(eval_dataloader, args.data_dir, "test")
    print("Overall (test): ", p, r, f1)

    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
