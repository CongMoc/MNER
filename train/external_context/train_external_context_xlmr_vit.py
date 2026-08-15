import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse

import logging
import random
import numpy as np
import torch
from transformers import AutoTokenizer, RobertaConfig
from modules.model_architecture.common import RobertaModel
from modules.model_architecture.UMT_PixelCNN_external_context_ViT import UMT_PixelCNN_ExternalContext_ViT
from modules.vit.vit_utils import myViT
from modules.datasets.dataset_externalcontext import convert_mm_examples_to_features, MNERProcessor
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
parser = argparse.ArgumentParser()

parser.add_argument("--alpha", default=0.5, type=float, help="parameter for Conversion Matrix")
parser.add_argument("--temp", type=float, default=0.179, help="parameter for CL training")
parser.add_argument("--temp_lamb", type=float, default=0.7, help="parameter for CL training")
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--bert_model", default='xlm-roberta-large', type=str)
parser.add_argument("--task_name", default='sonba', type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--cache_dir", default="cache", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=2.2e-5, type=float)
parser.add_argument("--num_train_epochs", default=10.0, type=float)
parser.add_argument("--warmup_proportion", default=0.4, type=float)
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--layer_num1', type=int, default=1)
parser.add_argument('--layer_num2', type=int, default=1)
parser.add_argument('--layer_num3', type=int, default=1)
parser.add_argument('--fine_tune_cnn', action='store_true')
parser.add_argument('--vit_model', default='google/vit-large-patch16-224-in21k')
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--path_image', required=True)
parser.add_argument('--image_source', default='crawled', choices=['crawled', 'random', 'blank', 'generated'])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: %s n_gpu: %d", device, n_gpu)

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

processor = MNERProcessor()
label_list = processor.get_labels()
auxlabel_list = processor.get_auxlabels()
num_labels = len(label_list) + 1
auxnum_labels = len(auxlabel_list) + 1

trans_matrix = np.zeros((auxnum_labels, num_labels), dtype=float)
if num_labels > 70:
    trans_matrix[0, 0] = 1
    trans_matrix[1, 1] = 1
    for k in range(2, 86, 2):
        trans_matrix[2, k] = 0.25
    for k in range(3, 86, 2):
        trans_matrix[3, k] = 0.25
    trans_matrix[4, 86] = 1
    trans_matrix[5, 87] = 1
    trans_matrix[6, 88] = 1
else:
    trans_matrix[0, 0] = 1
    trans_matrix[1, 1] = 1
    trans_matrix[2, 2] = 0.25
    trans_matrix[2, 4] = 0.25
    trans_matrix[2, 6] = 0.25
    trans_matrix[2, 8] = 0.25
    trans_matrix[3, 3] = 0.25
    trans_matrix[3, 5] = 0.25
    trans_matrix[3, 7] = 0.25
    trans_matrix[3, 9] = 0.25
    trans_matrix[4, 10] = 1
    trans_matrix[5, 11] = 1
    trans_matrix[6, 12] = 1

tokenizer = AutoTokenizer.from_pretrained(args.bert_model, cache_dir=args.cache_dir)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

config = RobertaConfig.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
roberta_pretrained = RobertaModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir)

encoder = myViT(args.vit_model, args.fine_tune_cnn, device, cache_dir=args.cache_dir)
vis_hidden_size = encoder.hidden_size
num_img_tokens = encoder.num_patches

model = UMT_PixelCNN_ExternalContext_ViT(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2,
                                          layer_num3=args.layer_num3, num_labels_=num_labels, auxnum_labels=auxnum_labels,
                                          vis_hidden_size=vis_hidden_size, num_img_tokens=num_img_tokens)
model.roberta.load_state_dict(roberta_pretrained.state_dict())
model.to(device)
encoder.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    encoder = torch.nn.DataParallel(encoder)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
weight_decay_crf = 0.00005
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if 'crf' not in n and not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if 'crf' not in n and any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if 'crf' in n], 'lr': 1.0e-2, 'weight_decay': weight_decay_crf},
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                      warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

alpha = args.alpha
temp = args.temp
temp_lamb = args.temp_lamb

use_random_image = args.image_source == "random"
use_blank_image = args.image_source == "blank"
use_generated_image = args.image_source == "generated"
cache_suffix = "_extctx_vit" if args.image_source == "crawled" else f"_extctx_vit_{args.image_source}"

IMAGE_KW = dict(num_image_tokens=num_img_tokens, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
                random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
                generated_image_source=use_generated_image)


def build_tensor_dataset(features):
    return TensorDataset(
        torch.tensor([f.input_ids_external for f in features], dtype=torch.long),
        torch.tensor([f.input_mask_external for f in features], dtype=torch.long),
        torch.tensor([f.added_input_mask_external for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids_external for f in features], dtype=torch.long),
        torch.tensor([f.input_ids_origin for f in features], dtype=torch.long),
        torch.tensor([f.input_mask_origin for f in features], dtype=torch.long),
        torch.tensor([f.added_input_mask_origin for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids_origin for f in features], dtype=torch.long),
        torch.stack([f.img_feat for f in features]),
        torch.stack([f.img_ti_feat for f in features]),
        torch.tensor([f.label_id_external for f in features], dtype=torch.long),
        torch.tensor([f.label_id_origin for f in features], dtype=torch.long),
        torch.tensor([f.auxlabel_id_external for f in features], dtype=torch.long),
        torch.tensor([f.auxlabel_id_origin for f in features], dtype=torch.long),
    )


def run_eval(dataloader, examples, split_file):
    model.eval()
    encoder.eval()
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "<pad>"
    y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        (input_ids_external, input_mask_external, added_input_mask_external, segment_ids_external,
         input_ids_origin, input_mask_origin, added_input_mask_origin, segment_ids_origin,
         img_feats, image_ti_feat, label_ids_external, label_ids_origin,
         auxlabel_ids_external, auxlabel_ids_origin) = batch
        with torch.no_grad():
            imgs_f, img_mean, img_att = encoder(img_feats)
            trans_matrix_t = torch.tensor(trans_matrix).to(device)
            pred_tags = model(
                input_ids_external=input_ids_external, segment_ids_external=segment_ids_external,
                input_mask_external=input_mask_external, added_attention_mask_external=added_input_mask_external,
                visual_embeds_mean=imgs_f, visual_embeds_att=img_att, trans_matrix=trans_matrix_t,
                added_attention_mask_origin=added_input_mask_origin,
                input_ids_origin=input_ids_origin, segment_ids_origin=segment_ids_origin,
                input_mask_origin=input_mask_origin, image_decode=None, alpha=alpha, temp=temp, temp_lamb=temp_lamb,
                labels_external=None, auxlabels_external=None, labels_origin=None, auxlabels_origin=None,
            )
        label_ids_np = label_ids_external.to('cpu').numpy()
        input_mask_np = input_mask_external.to('cpu').numpy()
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
    data_raw, _, _ = processor._read_sbtsv(os.path.join(args.data_dir, split_file))
    sentence_list = [data_raw[i][0] for i in range(len(y_pred))]
    reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
    return report, p, r, f1


if args.do_train:
    train_dataloader_save_path = args.data_dir + f"/train_dataloader_dataset{cache_suffix}.pth"
    dev_dataloader_save_path = args.data_dir + f"/dev_dataloader_dataset{cache_suffix}.pth"

    if not os.path.exists(train_dataloader_save_path):
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size,
            args.path_image, **IMAGE_KW)
        train_data = build_tensor_dataset(train_features)
        torch.save(train_data, train_dataloader_save_path)
    else:
        train_data = torch.load(train_dataloader_save_path, weights_only=False)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.train_batch_size)

    dev_eval_examples = processor.get_dev_examples(args.data_dir)
    if not os.path.exists(dev_dataloader_save_path):
        dev_eval_features = convert_mm_examples_to_features(
            dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size,
            args.path_image, **IMAGE_KW)
        dev_eval_data = build_tensor_dataset(dev_eval_features)
        torch.save(dev_eval_data, dev_dataloader_save_path)
    else:
        dev_eval_data = torch.load(dev_dataloader_save_path, weights_only=False)
    dev_eval_dataloader = DataLoader(dev_eval_data, sampler=SequentialSampler(dev_eval_data),
                                      batch_size=args.eval_batch_size)

    max_dev_f1, best_dev_epoch = 0.0, 0
    logger.info("***** Running training *****")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        encoder.train()
        tr_loss, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            (input_ids_external, input_mask_external, added_input_mask_external, segment_ids_external,
             input_ids_origin, input_mask_origin, added_input_mask_origin, segment_ids_origin,
             img_feats, image_ti_feat, label_ids_external, label_ids_origin,
             auxlabel_ids_external, auxlabel_ids_origin) = batch
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
            trans_matrix_t = torch.tensor(trans_matrix).to(device)
            loss = model(
                input_ids_external=input_ids_external, segment_ids_external=segment_ids_external,
                input_mask_external=input_mask_external, added_attention_mask_external=added_input_mask_external,
                visual_embeds_mean=imgs_f, visual_embeds_att=img_att, trans_matrix=trans_matrix_t,
                added_attention_mask_origin=added_input_mask_origin,
                input_ids_origin=input_ids_origin, segment_ids_origin=segment_ids_origin,
                input_mask_origin=input_mask_origin, image_decode=image_ti_feat, alpha=alpha, temp=temp,
                temp_lamb=temp_lamb, labels_external=label_ids_external, auxlabels_external=auxlabel_ids_external,
                labels_origin=label_ids_origin, auxlabels_origin=auxlabel_ids_origin,
            )
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        logger.info(f"===============Main loss: {tr_loss/nb_tr_steps}===============")
        print(f"===============Main loss: {tr_loss/nb_tr_steps}===============")

        report, p, r, f1 = run_eval(dev_eval_dataloader, dev_eval_examples, "dev.txt")
        logger.info("***** Dev Eval results *****")
        logger.info("\n%s", report)
        print("Overall (dev): ", p, r, f1)

        if f1 >= max_dev_f1:
            model_to_save = model.module if hasattr(model, 'module') else model
            encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(encoder_to_save.state_dict(), output_encoder_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            max_dev_f1, best_dev_epoch = f1, train_idx
            logger.info("******************SAVE NEW MODEL WEIGHT*********************")

    logger.info("Best epoch: %s, best dev F1: %s", best_dev_epoch, max_dev_f1)

if args.do_eval:
    model.load_state_dict(torch.load(output_model_file, map_location=device))
    model.to(device)
    encoder.load_state_dict(torch.load(output_encoder_file, map_location=device))
    encoder.to(device)

    eval_examples = processor.get_test_examples(args.data_dir)
    test_dataloader_save_path = args.data_dir + f"/test_dataloader_dataset{cache_suffix}.pth"
    if not os.path.exists(test_dataloader_save_path):
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size,
            args.path_image, **IMAGE_KW)
        eval_data = build_tensor_dataset(eval_features)
        torch.save(eval_data, test_dataloader_save_path)
    else:
        eval_data = torch.load(test_dataloader_save_path, weights_only=False)
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=args.eval_batch_size)

    report, p, r, f1 = run_eval(eval_dataloader, eval_examples, "test.txt")
    print("Overall (test): ", p, r, f1)

    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
