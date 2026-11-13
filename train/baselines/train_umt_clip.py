"""UMT with CLIP ViT-B/32 visual backbone (modules/model_architecture/UMT_clip.py),
replacing ResNet-152. Copied from train_umt_newsmner.py and simplified: CLIP is
owned + frozen inside the model itself (see UMT_clip.py docstring), so there is
no separate `encoder` object, no torch.no_grad() wrapper around a visual-feature
extraction call, and no --fine_tune_cnn/--resnet_* flags -- img_feats goes
straight into model(...) as CLIP-normalized pixel values.

Everything else (trans_matrix construction, --bert_model default, training loop
shape, eval loop) is unchanged from train_umt_newsmner.py.
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
from modules.model_architecture.UMT_clip import UMTClip, CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
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
    elsewhere in this repo -- see train_umt_newsmner.py's module docstring."""
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
parser.add_argument("--clip_model", default='openai/clip-vit-base-patch32', type=str)
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
parser.add_argument('--aux_loss_weight', default=0.5, type=float,
                    help="beta: weight of the text-only auxiliary CRF loss (UMTBalanced mix).")
parser.add_argument('--crop_size', type=int, default=224,
                    help="CLIP ViT-B/32: 224/32=7 -> 7x7=49 patches, matching UMT's 49 image regions.")
parser.add_argument('--path_image', required=True, help='path to NewsMNER images ({IMGID}.jpg)')
parser.add_argument('--image_source', default='crawled', choices=['crawled', 'random', 'blank', 'generated'],
                    help="'crawled' uses each example's correct IMGID-matched image; "
                         "'random' (ablation) fixes each example to a randomly-chosen real image instead; "
                         "'blank' (ablation) uses the same neutral background.jpg for every example; "
                         "'generated' (ablation) uses a pre-generated (e.g. text-to-image) image per example, "
                         "looked up as path_image/{guid}.jpg")
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
    steps_per_epoch = math.ceil(len(train_examples) / args.train_batch_size)
    num_train_optimization_steps = int(
        steps_per_epoch / args.gradient_accumulation_steps * args.num_train_epochs)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
roberta_pretrained = RobertaModel.from_pretrained(args.bert_model, cache_dir='cache')
model = UMTClip(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
                num_labels_=num_labels, auxnum_labels=auxnum_labels, aux_loss_weight=args.aux_loss_weight,
                clip_model_name=args.clip_model)
model.roberta.load_state_dict(roberta_pretrained.state_dict())
# Set before model.to(device)/DataParallel wrap below so both carry it along correctly (see
# the trans_matrix buffer registered in UMT.__init__ for why this isn't a forward() argument).
model.trans_matrix = torch.tensor(trans_matrix, dtype=torch.float)

if args.fp16:
    model.half()
model.to(device)
if args.local_rank != -1:
    from apex.parallel import DistributedDataParallel as DDP
    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Only trainable params go to the optimizer -- CLIP's weights have requires_grad=False
# (frozen in UMTClip.__init__), so filter them out instead of passing dead param groups.
param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
# alpha_logit (UMTBalanced's learned main/aux mix, a single scalar init to 0 -> sigmoid=0.5)
# was falling into the weight_decay=0.01 group below like any ordinary weight matrix. Decay
# continuously pulls a scalar toward 0 regardless of its gradient, and for a single parameter
# with an already-weak/noisy gradient signal that's enough to pin it at init the whole run
# (observed: sigmoid(alpha_logit) stuck at ~0.4999 for 12 epochs in news_results/umt_clip's
# gate_stats.jsonl) -- excluding it lets real gradient signal move it.
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'alpha_logit']
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

# "_clip" suffix: CLIP normalization (mean/std) differs from ResNet's ImageNet stats,
# so cached pixel tensors are NOT interchangeable with the ResNet runs' cache files
# even for the same image_source -- a shared cache_suffix would silently feed
# ImageNet-normalized pixels into CLIP.
cache_suffix = ("" if args.image_source == "crawled" else f"_{args.image_source}") + "_clip"
use_random_image = args.image_source == "random"
use_blank_image = args.image_source == "blank"
use_generated_image = args.image_source == "generated"

if args.do_train:
    train_dataloader_save_path = args.data_dir + f"/train_dataloader_dataset{cache_suffix}.pth"
    dev_dataloader_save_path = args.data_dir + f"/dev_dataloader_dataset{cache_suffix}.pth"
    if not os.path.exists(train_dataloader_save_path):
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image,
            image_mean=CLIP_IMAGE_MEAN, image_std=CLIP_IMAGE_STD,
            random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
            generated_image_source=use_generated_image)
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
            dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image,
            image_mean=CLIP_IMAGE_MEAN, image_std=CLIP_IMAGE_STD,
            random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
            generated_image_source=use_generated_image)
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
    gate_stats_file = os.path.join(args.output_dir, "gate_stats.jsonl")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info("********** Epoch: %d **********", train_idx)
        model.train()
        gate_sum, gate_n, last_alpha = 0.0, 0, None
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids = batch
            # img_feats here is CLIP-normalized raw pixels -- CLIP runs INSIDE model(...)
            # (frozen, see UMT_clip.py), no separate encoder() call / no_grad() needed.
            neg_log_likelihood = model(input_ids, segment_ids, input_mask, added_input_mask,
                                        img_feats, label_ids, auxlabel_ids)
            if n_gpu > 1:
                neg_log_likelihood = neg_log_likelihood.mean()
            if args.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps
            neg_log_likelihood.backward()

            m = model.module if hasattr(model, 'module') else model
            if m.last_gate_mean is not None:
                gate_sum += m.last_gate_mean
                gate_n += 1
                last_alpha = m.last_alpha

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if gate_n > 0:
            gate_mean_epoch = gate_sum / gate_n
            logger.info("***** Epoch %d image-gate stats: gate_mean=%.4f (0=pure text, 1=pure image) "
                        "alpha=%.4f (main-branch weight; 1-alpha=aux text-only weight) *****",
                        train_idx, gate_mean_epoch, last_alpha)
            with open(gate_stats_file, "a") as gf:
                gf.write(json.dumps({"epoch": train_idx, "gate_mean": gate_mean_epoch, "alpha": last_alpha}) + "\n")

        model.eval()
        # UMT's main-task head decodes via CRF (List[List[int]]), which
        # torch.nn.DataParallel.gather() cannot reduce across multiple GPUs (crashes with
        # "TypeError: 'int' object is not iterable"). Keep training on all GPUs via
        # DataParallel, but unwrap to the underlying module for this eval-only forward pass,
        # same fix already applied to train_umt_newsmner.py / train_xlmr_crf.py.
        eval_model = model.module if hasattr(model, 'module') else model
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
                predicted_label_seq_ids = eval_model(input_ids, segment_ids, input_mask, added_input_mask,
                                                img_feats)
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
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            model_config = {"bert_model": args.bert_model, "clip_model": args.clip_model,
                            "do_lower": args.do_lower_case,
                            "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                            "label_map": {i: label for i, label in enumerate(label_list, 1)}}
            json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
            max_dev_f1 = f1
            best_dev_epoch = train_idx

    logger.info("Best epoch on dev: %s, best dev F1: %s", best_dev_epoch, max_dev_f1)

if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
    model = UMTClip(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
                    num_labels_=num_labels, auxnum_labels=auxnum_labels, aux_loss_weight=args.aux_loss_weight,
                    clip_model_name=args.clip_model)
    # trans_matrix is a non-persistent buffer (see UMT.__init__), so it is never in the
    # checkpoint and load_state_dict below does not need/expect it -- reassign explicitly.
    model.trans_matrix = torch.tensor(trans_matrix, dtype=torch.float)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    eval_examples = processor.get_test_examples(args.data_dir)
    test_dataloader_save_path = args.data_dir + f"/test_dataloader_dataset{cache_suffix}.pth"
    if not os.path.exists(test_dataloader_save_path):
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image,
            image_mean=CLIP_IMAGE_MEAN, image_std=CLIP_IMAGE_STD,
            random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
            generated_image_source=use_generated_image)
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
    y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "<pad>"
    for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
            eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids = (
            input_ids.to(device), input_mask.to(device), added_input_mask.to(device),
            segment_ids.to(device), img_feats.to(device), label_ids.to(device), auxlabel_ids.to(device))
        with torch.no_grad():
            predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask,
                                            img_feats)
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
