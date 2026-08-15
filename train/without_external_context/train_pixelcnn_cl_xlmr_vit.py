import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, RobertaConfig
from modules.model_architecture.common import RobertaModel
from modules.model_architecture.UMT_PixelCNN_ViT import UMT_PixelCNN_ViT
from modules.vit.vit_utils import myViT
from modules.datasets.dataset_roberta_main import convert_mm_examples_to_features,MNERProcessor
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam,warmup_linear
from ner_evaluate import evaluate_each_class
from seqeval.metrics import classification_report
from ner_evaluate import evaluate
from tqdm import tqdm, trange
import json
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--alpha",
                    default=0.5,
                    type=float,
                    help="parameter for Conversion Matrix")

parser.add_argument("--beta",
                    default=0.5,
                    type=float,
                    help="parameter for aux loss")


parser.add_argument("--sigma",
                    default=0.005,
                    type=float,
                    help="parameter for PixelCNN loss")


parser.add_argument("--theta",
                    default=0.05,
                    type=float,
                    help="parameter for CL loss")


parser.add_argument("--weight_decay_pixelcnn",
                    default=0.0,
                    type=float,
                    help="weight_decay for PixelCNN++")

parser.add_argument("--lr_pixelcnn",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for PixelCNN++")

parser.add_argument('--lamb',
                    default=0.62,
                    type=float)

parser.add_argument('--temp',
                    type=float,
                    default=0.179,
                    help="parameter for CL training")

parser.add_argument('--temp_lamb',
                    type=float,
                    default=0.7,
                    help="parameter for CL training")

parser.add_argument("--data_dir",
                    default='./data/twitter2017',
                    type=str,

                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default='xlm-roberta-large', type=str)
parser.add_argument("--task_name",
                    default='twitter2017',
                    type=str,

                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./output_result',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--cache_dir",
                    default="",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")

parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--train_batch_size",
                    default=64,
                    type=int,
                    help="Total batch size for training.")

parser.add_argument("--eval_batch_size",
                    default=64,
                    type=int,
                    help="Total batch size for eval.")

parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--num_train_epochs",
                    default=12.0,
                    type=float,
                    help="Total number of training epochs to perform.")

parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")

parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")

parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")

parser.add_argument('--seed',
                    type=int,
                    default=37,
                    help="random seed for initialization")

parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")

parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                            "0 (default value): dynamic loss scaling.\n"
                            "Positive power of 2: static loss scaling value.\n")

parser.add_argument('--mm_model', default='MTCCMBert', help='model name')  # 'MTCCMBert', 'NMMTCCMBert'
parser.add_argument('--layer_num1', type=int, default=1, help='number of txt2img layer')
parser.add_argument('--layer_num2', type=int, default=1, help='number of img2txt layer')
parser.add_argument('--layer_num3', type=int, default=1, help='number of txt2txt layer')
parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained ViT if True')
parser.add_argument('--vit_model', default='google/vit-large-patch16-224-in21k', help='HF model id/path of the ViT visual backbone')
parser.add_argument('--crop_size', type=int, default=224, help='crop size of image (must match --vit_model image_size)')
parser.add_argument('--path_image', default='./IJCAI2019_data/twitter2017_images/', help='path to images')
parser.add_argument('--image_source', default='crawled', choices=['crawled', 'random', 'blank', 'generated'],
                    help="'crawled' uses each example's correct IMGID-matched image; "
                         "'random' (ablation) fixes each example to a randomly-chosen real image instead; "
                         "'blank' (ablation) uses the same neutral background.jpg for every example; "
                         "'generated' (ablation) uses a pre-generated (e.g. text-to-image) image per example, "
                         "looked up as path_image/{guid}.jpg")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
args = parser.parse_args()

if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

processors = {
    "twitter2015": MNERProcessor,
    "twitter2017": MNERProcessor,
    "sonba": MNERProcessor
}

if args.local_rank == -1 or args.no_cuda:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1 or args.train_batch_size < args.gradient_accumulation_steps:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))

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
num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1
auxnum_labels = len(auxlabel_list)+1 # label 0 corresponds to padding, label in label_list starts from 1

start_label_id = processor.get_start_label_id()
stop_label_id = processor.get_stop_label_id()

trans_matrix = np.zeros((auxnum_labels,num_labels), dtype=float)
if num_labels > 70:
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1 # O to O
    trans_matrix[2,2]=0.25
    trans_matrix[2,4]=0.25
    trans_matrix[2,6]=0.25
    trans_matrix[2,8]=0.25
    trans_matrix[2,10]=0.25
    trans_matrix[2,12]=0.25
    trans_matrix[2,14]=0.25
    trans_matrix[2,16]=0.25
    trans_matrix[2,18]=0.25
    trans_matrix[2,20]=0.25
    trans_matrix[2,22]=0.25
    trans_matrix[2,24]=0.25
    trans_matrix[2,26]=0.25
    trans_matrix[2,28]=0.25
    trans_matrix[2,30]=0.25
    trans_matrix[2,32]=0.25
    trans_matrix[2,34]=0.25
    trans_matrix[2,36]=0.25
    trans_matrix[2,38]=0.25
    trans_matrix[2,40]=0.25
    trans_matrix[2,42]=0.25
    trans_matrix[2,44]=0.25
    trans_matrix[2,46]=0.25
    trans_matrix[2,48]=0.25
    trans_matrix[2,50]=0.25
    trans_matrix[2,52]=0.25
    trans_matrix[2,54]=0.25
    trans_matrix[2,56]=0.25
    trans_matrix[2,58]=0.25
    trans_matrix[2,60]=0.25
    trans_matrix[2,62]=0.25
    trans_matrix[2,64]=0.25
    trans_matrix[2,66]=0.25
    trans_matrix[2,68]=0.25
    trans_matrix[2,70]=0.25
    trans_matrix[2,72]=0.25
    trans_matrix[2,74]=0.25
    trans_matrix[2,76]=0.25
    trans_matrix[2,78]=0.25
    trans_matrix[2,80]=0.25
    trans_matrix[2,82]=0.25
    trans_matrix[2,84]=0.25
    trans_matrix[3,3]=0.25
    trans_matrix[3,5]=0.25
    trans_matrix[3,7]=0.25
    trans_matrix[3,9]=0.25
    trans_matrix[3,11]=0.25
    trans_matrix[3,13]=0.25
    trans_matrix[3,15]=0.25
    trans_matrix[3,17]=0.25
    trans_matrix[3,19]=0.25
    trans_matrix[3,21]=0.25
    trans_matrix[3,23]=0.25
    trans_matrix[3,25]=0.25
    trans_matrix[3,27]=0.25
    trans_matrix[3,29]=0.25
    trans_matrix[3,31]=0.25
    trans_matrix[3,33]=0.25
    trans_matrix[3,35]=0.25
    trans_matrix[3,37]=0.25
    trans_matrix[3,39]=0.25
    trans_matrix[3,41]=0.25
    trans_matrix[3,43]=0.25
    trans_matrix[3,45]=0.25
    trans_matrix[3,47]=0.25
    trans_matrix[3,49]=0.25
    trans_matrix[3,51]=0.25
    trans_matrix[3,53]=0.25
    trans_matrix[3,55]=0.25
    trans_matrix[3,57]=0.25
    trans_matrix[3,59]=0.25
    trans_matrix[3,61]=0.25
    trans_matrix[3,63]=0.25
    trans_matrix[3,65]=0.25
    trans_matrix[3,67]=0.25
    trans_matrix[3,69]=0.25
    trans_matrix[3,71]=0.25
    trans_matrix[3,73]=0.25
    trans_matrix[3,75]=0.25
    trans_matrix[3,77]=0.25
    trans_matrix[3,79]=0.25
    trans_matrix[3,81]=0.25
    trans_matrix[3,83]=0.25
    trans_matrix[3,85]=0.25
    trans_matrix[4,86]=1   # X to X
    trans_matrix[5,87]=1   # [CLS] to [CLS]
    trans_matrix[6,88]=1   # [SEP] to [SEP]
else:
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1 # O to O
    trans_matrix[2,2]=0.25 # B to B-MISC
    trans_matrix[2,4]=0.25 # B to B-PER
    trans_matrix[2,6]=0.25 # B to B-ORG
    trans_matrix[2,8]=0.25 # B to B-LOC
    trans_matrix[3,3]=0.25 # I to I-MISC
    trans_matrix[3,5]=0.25 # I to I-PER
    trans_matrix[3,7]=0.25 # I to I-ORG
    trans_matrix[3,9]=0.25 # I to I-LOC
    trans_matrix[4,10]=1   # X to X
    trans_matrix[5,11]=1   # [CLS] to [CLS]
    trans_matrix[6,12]=1   # [SEP] to [SEP]
'''
trans_matrix = np.zeros((num_labels, auxnum_labels), dtype=float)
trans_matrix[0,0]=1 # pad to pad
trans_matrix[1,1]=1 # O to O
trans_matrix[2,2]=0.25 # B to B-MISC
trans_matrix[2,4]=0.25 # B to B-PER
trans_matrix[2,6]=0.25 # B to B-ORG
trans_matrix[2,8]=0.25 # B to B-LOC
trans_matrix[3,3]=0.25 # I to I-MISC
trans_matrix[3,5]=0.25 # I to I-PER
trans_matrix[3,7]=0.25 # I to I-ORG
trans_matrix[3,9]=0.25 # I to I-LOC
trans_matrix[4,10]=1   # X to X
trans_matrix[5,11]=1   # [CLS] to [CLS]
trans_matrix[6,12]=1   # [SEP] to [SEP]
'''

tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


encoder = myViT(args.vit_model, args.fine_tune_cnn, device, cache_dir='cache')
vis_hidden_size = encoder.hidden_size
num_img_tokens = encoder.num_patches

if args.mm_model == 'MTCCMBert':
    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
    roberta_pretrained = RobertaModel.from_pretrained(args.bert_model, cache_dir='cache')
    model = UMT_PixelCNN_ViT(config, layer_num1=args.layer_num1,
                                layer_num2=args.layer_num2,
                                layer_num3=args.layer_num3,
                                num_labels_=num_labels, auxnum_labels = auxnum_labels,
                                vis_hidden_size=vis_hidden_size, num_img_tokens=num_img_tokens)
    model.roberta.load_state_dict(roberta_pretrained.state_dict())
else:
    print('please define your MNER Model')

# if args.fp16:
#     model.half()
#     encoder.half()
model.to(device)


encoder.to(device)
if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
    encoder = DDP(encoder)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)
    encoder = torch.nn.DataParallel(encoder)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

weight_decay_pixelcnn = args.weight_decay_pixelcnn
weight_decay_crf = 0.00005       # adjust as needed (could be same as pixelcnn or different)

optimizer_grouped_parameters = [
    # Group 1: All parameters except image_decoder and CRF, without no_decay terms → weight_decay=0.01
    {
        'params': [p for n, p in param_optimizer 
                   if not any(nd in n for nd in ['image_decoder', 'crf']) 
                   and not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    
    # Group 2: All parameters except image_decoder and CRF, with no_decay terms → weight_decay=0.0
    {
        'params': [p for n, p in param_optimizer 
                   if not any(nd in n for nd in ['image_decoder', 'crf']) 
                   and any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    },
    
    # Group 3: image_decoder parameters (PixelCNN-like) with custom lr and weight_decay
    {
        'lr': 0.001,
        'params': [p for n, p in param_optimizer 
                   if any(nd in n for nd in ['image_decoder'])],
        'weight_decay': weight_decay_pixelcnn
    },
    
    # Group 4: CRF parameters (e.g., transition matrix, bias) with custom lr and weight_decay
    {
        'lr': 1.0e-2,                 # adjust if needed (often CRF uses smaller LR)
        'params': [p for n, p in param_optimizer 
                   if any(nd in n for nd in ['crf', 'crf_layer', 'transition'])],
        'weight_decay': weight_decay_crf
    }
]

if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")


temp = args.temp
temp_lamb = args.temp_lamb
lamb = args.lamb
alpha = args.alpha
beta = args.beta
theta = args.theta
sigma = args.sigma


if args.do_train:
    # separate cache filenames per image_source so the "crawled" and "random" ablation
    # runs never silently reuse each other's cached features (they share --data_dir)
    cache_suffix = "_vit" if args.image_source == "crawled" else f"_vit_{args.image_source}"
    train_dataloader_save_path = args.data_dir + f"/train_dataloader_dataset{cache_suffix}.pth"
    dev_dataloader_save_path = args.data_dir + f"/dev_dataloader_dataset{cache_suffix}.pth"
    use_random_image = args.image_source == "random"
    use_blank_image = args.image_source == "blank"
    use_generated_image = args.image_source == "generated"
    if not os.path.exists(train_dataloader_save_path):
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image,
            num_image_tokens=num_img_tokens, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
            random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
            generated_image_source=use_generated_image)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_image_ti_feat = torch.stack([f.img_ti_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, \
                                    all_segment_ids, all_img_feats, all_image_ti_feat, all_label_ids,all_auxlabel_ids)
        torch.save(train_data, train_dataloader_save_path)
    else:
        print("Loading the train_data (TensorDataset)")
        train_data = torch.load(train_dataloader_save_path, weights_only=False)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    dev_eval_examples = processor.get_dev_examples(args.data_dir)
    if not os.path.exists(dev_dataloader_save_path):
        dev_eval_features = convert_mm_examples_to_features(
            dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size,
            args.path_image, num_image_tokens=num_img_tokens, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
            random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
            generated_image_source=use_generated_image)
        all_input_ids = torch.tensor([f.input_ids for f in dev_eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in dev_eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in dev_eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in dev_eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in dev_eval_features])
        all_image_ti_feat = torch.stack([f.img_ti_feat for f in dev_eval_features])
        all_label_ids = torch.tensor([f.label_id for f in dev_eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in dev_eval_features], dtype=torch.long)
        dev_eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                        all_img_feats, all_image_ti_feat, all_label_ids, all_auxlabel_ids)
        torch.save(dev_eval_data, dev_dataloader_save_path)
    else:
        print("Loading the dev_dataloader_save_path (TensorDataset)")
        dev_eval_data = torch.load(dev_dataloader_save_path, weights_only=False)
    # Run prediction for full data
    dev_eval_sampler = SequentialSampler(dev_eval_data)
    dev_eval_dataloader = DataLoader(dev_eval_data, sampler=dev_eval_sampler, batch_size=args.eval_batch_size)

    max_dev_f1 = 0.0
    best_dev_epoch = 0
    logger.info("***** Running training *****")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info("********** Epoch: " + str(train_idx) + " **********")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        model.train()
        encoder.train()
        encoder.zero_grad()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, added_input_mask, segment_ids, img_feats, image_ti_feat, label_ids, auxlabel_ids = batch
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)

            trans_matrix = torch.tensor(trans_matrix).to(device)
            neg_log_likelihood = model(input_ids, segment_ids, input_mask, added_input_mask,
                                        imgs_f, img_att, trans_matrix, image_ti_feat, alpha, beta, theta, sigma, temp, temp_lamb, label_ids, auxlabel_ids)

            if n_gpu > 1:
                neg_log_likelihood = neg_log_likelihood.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(neg_log_likelihood)
            else:
                neg_log_likelihood.backward()
            tr_loss += neg_log_likelihood.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                        args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


        logger.info(f"===============Main loss: {tr_loss/nb_tr_steps}===============")
        print(f"===============Main loss: {tr_loss/nb_tr_steps}===============")


        model.eval()
        encoder.eval()

        logger.info("***** Running Dev evaluation *****")
        logger.info("  Num examples = %d", len(dev_eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "<pad>"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, image_ti_feat, label_ids, auxlabel_ids in tqdm(
                dev_eval_dataloader,
                desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)
            auxlabel_ids = auxlabel_ids.to(device)
            image_decode = None
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, imgs_f, img_att, trans_matrix, image_decode, alpha, beta, theta, sigma, temp = temp, temp_lamb = temp_lamb, labels=None, auxlabels=None)

            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>":
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
        sentence_list = []
        dev_data, imgs, _ = processor._read_sbtsv(os.path.join(args.data_dir, "dev.txt"))
        for i in range(len(y_pred)):
            sentence = dev_data[i][0]
            sentence_list.append(sentence)

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)

        logger.info("***** Dev Eval results *****")
        logger.info("\n%s", report)
        print("Overall: ", p, r, f1)
        F_score_dev = f1

        if F_score_dev >= max_dev_f1:
            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            encoder_to_save = encoder.module if hasattr(encoder,
                                                        'module') else encoder  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(encoder_to_save.state_dict(), output_encoder_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            model_config = {"bert_model": args.bert_model,"best_epoch": best_dev_epoch,"max_f1": max_dev_f1, "do_lower": args.do_lower_case,
                            "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                            "label_map": label_map}
            
            logger.info("******************SAVE NEW MODEL WEIGHT*********************")
            json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
            max_dev_f1 = F_score_dev
            best_dev_epoch = train_idx

    logger.info("**************************************************")
    logger.info("The best epoch on the dev set: %s", best_dev_epoch)
    logger.info("The best Overall-F1 score on the dev set: %s", max_dev_f1)
    logger.info('\n')

# loadmodel
if args.mm_model == 'MTCCMBert':
    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
    model = UMT_PixelCNN_ViT(config, layer_num1=args.layer_num1,
                                layer_num2=args.layer_num2,
                                layer_num3=args.layer_num3,
                                num_labels_=num_labels, auxnum_labels = auxnum_labels,
                                vis_hidden_size=vis_hidden_size, num_img_tokens=num_img_tokens)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)
else:
    print('please define your MNER Model')

if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    cache_suffix = "_vit" if args.image_source == "crawled" else f"_vit_{args.image_source}"
    use_random_image = args.image_source == "random"
    use_blank_image = args.image_source == "blank"
    use_generated_image = args.image_source == "generated"
    eval_examples = processor.get_test_examples(args.data_dir)
    test_dataloader_save_path = args.data_dir + f"/test_dataloader_dataset{cache_suffix}.pth"
    if not os.path.exists(test_dataloader_save_path):
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image,
            num_image_tokens=num_img_tokens, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
            random_image_source=use_random_image, random_seed=args.seed, blank_image_source=use_blank_image,
            generated_image_source=use_generated_image)
        logger.info("***** Running Test Evaluation with the Best Model on the Dev Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_image_ti_feat = torch.stack([f.img_ti_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats, all_image_ti_feat,
                                    all_label_ids,all_auxlabel_ids)
        torch.save(eval_data, test_dataloader_save_path)
    else:
        print("Loading the test_dataloader_save_path (TensorDataset)")
        eval_data = torch.load(test_dataloader_save_path, weights_only=False)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    encoder.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    y_true_idx = []
    y_pred_idx = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "<pad>"
    for input_ids, input_mask, added_input_mask, segment_ids, img_feats, image_ti_feat, label_ids, auxlabel_ids in tqdm(
            eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        added_input_mask = added_input_mask.to(device)
        segment_ids = segment_ids.to(device)
        img_feats = img_feats.to(device)
        label_ids = label_ids.to(device)
        auxlabel_ids = auxlabel_ids.to(device)

        image_decode = None
        trans_matrix = torch.tensor(trans_matrix).to(device)
        with torch.no_grad():
            imgs_f, img_mean, img_att = encoder(img_feats)
            predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, imgs_f, img_att, trans_matrix, image_decode, alpha, beta, theta, sigma, temp = temp, temp_lamb = temp_lamb, labels=None, auxlabels=None)

        logits = predicted_label_seq_ids
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        for i, mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []
            tmp1_idx = []
            tmp2_idx = []

            for j, m in enumerate(mask):
                if j == 0:
                    continue
                if m:
                    if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>":
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

    sentence_list = []
    test_data, imgs, _ = processor._read_sbtsv(os.path.join(args.data_dir, "test.txt"))
    output_pred_file = os.path.join(args.output_dir, "mtmner_pred.txt")
    fout = open(output_pred_file, 'w')
    for i in range(len(y_pred)):
        sentence = test_data[i][0]
        sentence_list.append(sentence)
        img = imgs[i]
        samp_pred_label = y_pred[i]
        samp_true_label = y_true[i]
        fout.write(img + '\n')
        fout.write(' '.join(sentence) + '\n')
        fout.write(' '.join(samp_pred_label) + '\n')
        fout.write(' '.join(samp_true_label) + '\n' + '\n')
    fout.close()

    reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
    print("Overall: ", p, r, f1)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')