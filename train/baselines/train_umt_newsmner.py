"""UMT (Yu et al. 2020) baseline on NewsMNER -- text + image only, no external
context. Same architecture/training loop as train_umt.py (ResNet-152 vision
backbone, unchanged), with two changes:

  1. --bert_model defaults to xlm-roberta-large instead of bert-base-cased, since
     UMT.py's text branch is just a RobertaModel-family encoder (already
     demonstrated with vinai/phobert-base-v2 in UMT.py's own __main__ block) --
     no model code change needed, XLM-R-large is a drop-in swap.

  2. trans_matrix (maps the 6-way auxiliary entity-SPAN prediction O/B/I/X/<s>/</s>
     onto the real per-type NER labels) is now built generically from
     label_list/auxlabel_list instead of hardcoded. train_umt.py's original matrix
     (and every other trans_matrix block elsewhere in this repo except the
     70+-label VLSP2021 branch in train_pixelcnn_cl_xlmr_vit.py) hardcodes indices
     that only work for a 4-entity-type set (PER/ORG/LOC/MISC) -- NewsMNER's label
     set additionally has DATE and NUM (6 entity types), so every B-<type> is a
     different column index than the hardcoded 2/4/6/8 and would silently
     mis-route the auxiliary loss onto the wrong labels (or index out of range) if
     reused as-is. build_trans_matrix() spreads the aux "B" (resp. "I") prediction
     evenly across however many B-<type> (resp. I-<type>) labels label_list
     actually contains, reproducing the exact original matrix when there are 4
     types and generalizing correctly for NewsMNER's 6.

Usage mirrors train_umt.py (--data_dir, --path_image, --output_dir, --do_train,
--do_eval, ...); --task_name defaults to "newsmner" and LABELS must be set via the
LABELS env var (see scripts/server_ops/run_*.sh for the NewsMNER label list) since
MNERProcessor.get_labels() reads it from there.
"""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import logging
import random
import numpy as np
import torch
from transformers import AutoTokenizer, RobertaConfig
from modules.model_architecture.UMT import UMT
from modules.resnet import resnet as resnet
from modules.resnet.resnet_utils import myResnet
from modules.datasets.dataset_roberta_main import convert_mm_examples_to_features, MNERProcessor
from modules.model_architecture.common import RobertaModel
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam
from ner_evaluate import evaluate
from seqeval.metrics import classification_report
from tqdm import tqdm, trange
import json

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def build_trans_matrix(label_list, auxlabel_list):
    """Generic replacement for the hardcoded-per-dataset trans_matrix blocks
    elsewhere in this repo -- see module docstring."""
    num_labels = len(label_list) + 1
    auxnum_labels = len(auxlabel_list) + 1
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    aux_map = {label: i for i, label in enumerate(auxlabel_list, 1)}
    trans_matrix = np.zeros((auxnum_labels, num_labels), dtype=float)
    trans_matrix[0, 0] = 1  # pad -> pad
    if "O" in label_map and "O" in aux_map:
        trans_matrix[aux_map["O"], label_map["O"]] = 1
    b_labels = [l for l in label_list if l.startswith("B-")]
    i_labels = [l for l in label_list if l.startswith("I-")]
    if "B" in aux_map and b_labels:
        w = 1.0 / len(b_labels)
        for l in b_labels:
            trans_matrix[aux_map["B"], label_map[l]] = w
    if "I" in aux_map and i_labels:
        w = 1.0 / len(i_labels)
        for l in i_labels:
            trans_matrix[aux_map["I"], label_map[l]] = w
    for special in ("X", "<s>", "</s>"):
        if special in aux_map and special in label_map:
            trans_matrix[aux_map[special], label_map[special]] = 1
    return trans_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data_export/newsmner_local_eval', type=str,
                    help="dir with train/dev/test.txt (IMGID line + tokens, no external context)")
parser.add_argument("--bert_model", default='xlm-roberta-large', type=str)
parser.add_argument("--task_name", default='newsmner', type=str)
parser.add_argument("--output_dir", default='./output_result', type=str)
parser.add_argument("--cache_dir", default="", type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--num_train_epochs", default=12.0, type=float)
parser.add_argument("--warmup_proportion", default=0.1, type=float)
parser.add_argument("--no_cuda", action='store_true')
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--loss_scale', type=float, default=0)
parser.add_argument('--layer_num1', type=int, default=1, help='number of txt2img layers')
parser.add_argument('--layer_num2', type=int, default=1, help='number of img2txt layers')
parser.add_argument('--layer_num3', type=int, default=1, help='number of txt2txt layers')
parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained ResNet if True')
parser.add_argument('--resnet_root', default='./out_res', help='dir with resnet152.pth')
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--path_image', required=True, help='path to NewsMNER images ({IMGID}.jpg)')
args = parser.parse_args()

processors = {"newsmner": MNERProcessor}

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}".format(args.gradient_accumulation_steps))
args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

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

task_name = args.task_name.lower()
if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
label_list = processor.get_labels()
auxlabel_list = processor.get_auxlabels()
num_labels = len(label_list) + 1
auxnum_labels = len(auxlabel_list) + 1
logger.info("label_list (%d): %s", len(label_list), label_list)
logger.info("auxlabel_list (%d): %s", len(auxlabel_list), auxlabel_list)

trans_matrix = build_trans_matrix(label_list, auxlabel_list)

tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    # ceil(), matching the actual number of batches a DataLoader with drop_last=False
    # produces per epoch -- the original train_umt.py used int() (floor) here, which
    # under-counts steps_per_epoch whenever len(train_examples) isn't an exact
    # multiple of the batch size. BertAdam's LR schedule decays to 0 by t_total
    # steps, so an under-counted t_total makes the LR schedule finish (and LR go to
    # 0) partway through *actual* training. The relative error from floor-vs-ceil is
    # tiny for a large dataset (128 vs 129 steps/epoch at 100%) but enormous for a
    # small one (1 vs 2 steps/epoch at the 1% fraction, where the schedule ends at
    # step 12 of 24 real steps) -- exactly why the 0.01/0.05/0.10 fraction runs got
    # stuck predicting all-O (LR=0 for the back half of training) while 0.25+ and
    # 100% trained normally.
    steps_per_epoch = math.ceil(len(train_examples) / args.train_batch_size)
    num_train_optimization_steps = int(
        steps_per_epoch / args.gradient_accumulation_steps * args.num_train_epochs)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
roberta_pretrained = RobertaModel.from_pretrained(args.bert_model, cache_dir='cache')
model = UMT(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
            num_labels_=num_labels, auxnum_labels=auxnum_labels)
model.roberta.load_state_dict(roberta_pretrained.state_dict())

net = getattr(resnet, 'resnet152')()
net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth'), weights_only=False))
encoder = myResnet(net, args.fine_tune_cnn, device)

if args.fp16:
    model.half()
    encoder.half()
model.to(device)
encoder.to(device)
if args.local_rank != -1:
    from apex.parallel import DistributedDataParallel as DDP
    model = DDP(model)
    encoder = DDP(encoder)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)
    encoder = torch.nn.DataParallel(encoder)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'crf' not in n],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'crf' not in n],
     'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if 'crf' in n], 'lr': 1.0e-2, 'weight_decay': 0.0},
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                     warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

global_step = 0
output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

if args.do_train:
    train_dataloader_save_path = args.data_dir + "/train_dataloader_dataset.pth"
    dev_dataloader_save_path = args.data_dir + "/dev_dataloader_dataset.pth"
    if not os.path.exists(train_dataloader_save_path):
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask,
                                    all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids)
        torch.save(train_data, train_dataloader_save_path)
    else:
        logger.info("Loading cached train TensorDataset")
        train_data = torch.load(train_dataloader_save_path, weights_only=False)
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    dev_eval_examples = processor.get_dev_examples(args.data_dir)
    if not os.path.exists(dev_dataloader_save_path):
        dev_eval_features = convert_mm_examples_to_features(
            dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in dev_eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in dev_eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in dev_eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in dev_eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in dev_eval_features])
        all_label_ids = torch.tensor([f.label_id for f in dev_eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in dev_eval_features], dtype=torch.long)
        dev_eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                        all_img_feats, all_label_ids, all_auxlabel_ids)
        torch.save(dev_eval_data, dev_dataloader_save_path)
    else:
        logger.info("Loading cached dev TensorDataset")
        dev_eval_data = torch.load(dev_dataloader_save_path, weights_only=False)
    dev_eval_sampler = SequentialSampler(dev_eval_data)
    dev_eval_dataloader = DataLoader(dev_eval_data, sampler=dev_eval_sampler, batch_size=args.eval_batch_size)

    max_dev_f1 = 0.0
    best_dev_epoch = 0
    logger.info("***** Running training *****")
    trans_matrix_t = torch.tensor(trans_matrix).to(device)
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info("********** Epoch: %d **********", train_idx)
        model.train()
        encoder.train()
        encoder.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids = batch
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)

            neg_log_likelihood = model(input_ids, segment_ids, input_mask, added_input_mask,
                                        img_att, trans_matrix_t, label_ids, auxlabel_ids)
            if n_gpu > 1:
                neg_log_likelihood = neg_log_likelihood.mean()
            if args.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps
            neg_log_likelihood.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        model.eval()
        encoder.eval()
        logger.info("***** Running Dev evaluation *****")
        y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "<pad>"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
                dev_eval_dataloader, desc="Evaluating"):
            input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids = (
                input_ids.to(device), input_mask.to(device), added_input_mask.to(device),
                segment_ids.to(device), img_feats.to(device), label_ids.to(device), auxlabel_ids.to(device))
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask,
                                                img_att, trans_matrix_t)
            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1, temp_2, tmp1_idx, tmp2_idx = [], [], [], []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] not in ("X", "</s>"):
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)

        report = classification_report(y_true, y_pred, digits=4)
        dev_data, imgs, _ = processor._read_sbtsv(os.path.join(args.data_dir, "dev.txt"))
        sentence_list = [dev_data[i][0] for i in range(len(y_pred))]
        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        logger.info("***** Dev Eval results *****\n%s", report)
        print("Overall: ", p, r, f1)

        if f1 >= max_dev_f1:
            model_to_save = model.module if hasattr(model, 'module') else model
            encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(encoder_to_save.state_dict(), output_encoder_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                            "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                            "label_map": {i: label for i, label in enumerate(label_list, 1)}}
            json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
            max_dev_f1 = f1
            best_dev_epoch = train_idx

    logger.info("Best epoch on dev: %s, best dev F1: %s", best_dev_epoch, max_dev_f1)

if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
    model = UMT(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
                num_labels_=num_labels, auxnum_labels=auxnum_labels)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder.load_state_dict(torch.load(output_encoder_file))
    encoder.to(device)
    trans_matrix_t = torch.tensor(trans_matrix).to(device)

    eval_examples = processor.get_test_examples(args.data_dir)
    test_dataloader_save_path = args.data_dir + "/test_dataloader_dataset.pth"
    if not os.path.exists(test_dataloader_save_path):
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                    all_img_feats, all_label_ids, all_auxlabel_ids)
        torch.save(eval_data, test_dataloader_save_path)
    else:
        eval_data = torch.load(test_dataloader_save_path, weights_only=False)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    encoder.eval()
    y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "<pad>"
    for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
            eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids = (
            input_ids.to(device), input_mask.to(device), added_input_mask.to(device),
            segment_ids.to(device), img_feats.to(device), label_ids.to(device), auxlabel_ids.to(device))
        with torch.no_grad():
            imgs_f, img_mean, img_att = encoder(img_feats)
            predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask,
                                            img_att, trans_matrix_t)
        logits = predicted_label_seq_ids
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        for i, mask in enumerate(input_mask):
            temp_1, temp_2, tmp1_idx, tmp2_idx = [], [], [], []
            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids[i][j]] not in ("X", "</s>"):
                        temp_1.append(label_map[label_ids[i][j]])
                        tmp1_idx.append(label_ids[i][j])
                        temp_2.append(label_map[logits[i][j]])
                        tmp2_idx.append(logits[i][j])
                else:
                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
            y_true_idx.append(tmp1_idx)
            y_pred_idx.append(tmp2_idx)

    report = classification_report(y_true, y_pred, digits=4)
    test_data, imgs, _ = processor._read_sbtsv(os.path.join(args.data_dir, "test.txt"))
    sentence_list = [test_data[i][0] for i in range(len(y_pred))]
    reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
    print("Overall: ", p, r, f1)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test Eval results *****\n%s", report)
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
