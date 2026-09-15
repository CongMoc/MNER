"""Paper-faithful MoRe (Wang et al., "Named Entity and Relation Extraction with
Multi-Modal Retrieval", Findings of EMNLP 2022; official code: modelscope/AdaSeq
examples/MoRe).

In the original MoRe framework BOTH the MoRe-Text and MoRe-Image task models are
the *same* architecture: an XLM-RoBERTa-large encoder over

        [ input sentence ]  ++  [X]  ++  [ retrieved knowledge text z ]

followed by a linear-chain CRF. The two variants differ only in *where z came
from*: the text retriever (z_T, BM25 over ~200M Wikipedia paragraphs) or the
image-based retriever (z_I, CLIP ViT-B/32 + Faiss k-NN over ~4M Wikipedia image
entries -> title + intro of the article). The raw input image is NEVER encoded by
a vision backbone in the task model -- "Image" in "MoRe-Image" only means the
retrieval query was the image; CLIP-ViT lives only in the retriever and the final
MoE gate. Paper "Context Processing": z = {[X], t_1, ..., t_k}, top-k = 10.

Config (AdaSeq twitter-17-img.yaml + paper Sec. 3.1):
  encoder      xlm-roberta-large,  linear-chain CRF,  word/hidden dropout 0.1
  optimizer    AdamW,  lr 5e-6,  CRF param group lr 5e-2
  scheduler    linear decay to 0, NO warmup
  batch        4 (train) / 32 (eval),  max_seq_length 512
  epochs       10 (paper text; repo config says 15)
  results averaged over 3 random seeds

This script implements exactly that. It reads the same merged data format as
train_more_image.py (IMGID header, sentence tokens, `<EOS>` marker == the [X]
mark, retrieved-context tokens tagged `E`); `E`/`X`/`<s>`/`</s>` positions are
fed to the encoder but excluded from the seqeval metrics.

NOTE ON RETRIEVAL: reproducing the exact CLIP+Faiss retrieval over 4M Wikipedia
image entries is out of scope -- this run consumes whatever retrieved/generated
context the data_dir already contains.

(train_more_image.py in this repo is a *different*, non-paper model: same text
input but additionally fuses a real ViT-large image CLS embedding into every
token before the CRF.)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import logging
import random

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, RobertaConfig, get_linear_schedule_with_warmup
from modules.model_architecture.common import RobertaModel
from modules.model_architecture.XLMR_CRF_TextOnly import XLMR_CRF_TextOnly
from modules.datasets.dataset_roberta_main import MNERProcessor
from modules.datasets.dataset_text_only import convert_text_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from seqeval.metrics import classification_report
from ner_evaluate import evaluate
from tqdm import tqdm, trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=str,
                    help="dir with train/dev/test.txt, sentence + retrieved-context merged; LABELS must include E")
parser.add_argument("--bert_model", default='xlm-roberta-large', type=str)
parser.add_argument("--task_name", default='more_retrieval', type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--cache_dir", default="cache", type=str)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=5e-6, type=float)
parser.add_argument("--crf_learning_rate", default=5e-2, type=float)
parser.add_argument("--num_train_epochs", default=10.0, type=float)
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="paper/AdaSeq use NO warmup")
parser.add_argument("--weight_decay", default=0.0, type=float)
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
EXCLUDED_LABELS = ("X", "<s>", "</s>", "E")  # subword continuation, specials, retrieved-context filler

tokenizer = AutoTokenizer.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
config = RobertaConfig.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
roberta_pretrained = RobertaModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
model = XLMR_CRF_TextOnly(config, num_labels_=num_labels)
model.roberta.load_state_dict(roberta_pretrained.state_dict())
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')


def make_dataset(examples):
    feats = convert_text_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    return TensorDataset(
        torch.tensor([f.input_ids for f in feats], dtype=torch.long),
        torch.tensor([f.input_mask for f in feats], dtype=torch.long),
        torch.tensor([f.segment_ids for f in feats], dtype=torch.long),
        torch.tensor([f.label_id for f in feats], dtype=torch.long),
    )


def run_eval(dataloader, data_dir, split_name):
    model.eval()
    y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Evaluating"):
        input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)
        with torch.no_grad():
            pred_tags = model(input_ids, segment_ids, input_mask, labels=None)
        label_ids_np = label_ids.numpy()
        input_mask_np = input_mask.to('cpu').numpy()
        for i, mask in enumerate(input_mask_np):
            t1, t2, i1, i2 = [], [], [], []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids_np[i][j]] not in EXCLUDED_LABELS:
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
    steps_per_epoch = max(1, len(train_examples) // args.train_batch_size // args.gradient_accumulation_steps)
    total_steps = int(steps_per_epoch * args.num_train_epochs)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    grouped = [
        {'params': [p for n, p in param_optimizer if 'crf' not in n and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if 'crf' not in n and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if 'crf' in n],
         'weight_decay': 0.0, 'lr': args.crf_learning_rate},
    ]
    optimizer = AdamW(grouped, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)

    train_data = make_dataset(train_examples)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.train_batch_size)
    dev_data = make_dataset(processor.get_dev_examples(args.data_dir))
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.eval_batch_size)

    max_dev_f1, best_dev_epoch = 0.0, 0
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss, nb_tr_steps = 0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            input_ids, input_mask, segment_ids, label_ids = tuple(t.to(device) for t in batch)
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
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
    eval_data = make_dataset(processor.get_test_examples(args.data_dir))
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=args.eval_batch_size)
    report, p, r, f1 = run_eval(eval_dataloader, args.data_dir, "test")
    print("Overall (test): ", p, r, f1)
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
