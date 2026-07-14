import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from modules.model_architecture.common import RobertaModel
from transformers import AutoTokenizer,BertConfig, RobertaConfig
from modules.model_architecture.EXCT import EXTCModel
from modules.resnet import resnet as resnet
from modules.resnet.resnet_utils import myResnet
from modules.datasets.dataset_externalcontext import convert_mm_examples_to_features,MNERProcessor
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
parser.add_argument("--bert_model", default='bert-base-cased', type=str)
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
parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
parser.add_argument('--resnet_root', default='./out_res', help='path the pre-trained cnn models')
parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
parser.add_argument('--path_image', default='./IJCAI2019_data/twitter2017_images/', help='path to images')
# parser.add_argument('--mm_model', default='TomBert', help='model name') #
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
args = parser.parse_args()

if args.task_name == "twitter2017":
    args.path_image = "./IJCAI2019_data/twitter2017_images/"
elif args.task_name == "twitter2015":
    args.path_image = "./IJCAI2019_data/twitter2015_images/"

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

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

# '''
#
if args.task_name == "twitter2015":
    args.num_train_epochs = 24.0
if args.task_name == "twitter2017":
    args.num_train_epochs = 23.0
# '''

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
tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

if args.mm_model == 'MTCCMBert':
    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
    roberta_pretrained = RobertaModel.from_pretrained(args.bert_model, cache_dir='cache')
    model = EXTCModel(config, layer_num1=args.layer_num1,
                                layer_num2=args.layer_num2,
                                layer_num3=args.layer_num3,
                                num_labels_=num_labels, auxnum_labels = auxnum_labels)
else:
    print('please define your MNER Model')

net = getattr(resnet, 'resnet152')()
net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth'), weights_only=False))
encoder = myResnet(net, args.fine_tune_cnn, device)

if args.fp16:
    model.half()
    encoder.half()
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
    train_dataloader_save_path = os.path.join(args.data_dir, "train_dataloader_dataset.pth")
    dev_dataloader_save_path = os.path.join(args.data_dir, "dev_dataloader_dataset.pth")

    if not os.path.exists(train_dataloader_save_path):
        print("Creating train_data TensorDataset...")
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)

        # Extract tensors for ALL fields from SBInputFeatures
        all_input_ids_external = torch.tensor([f.input_ids_external for f in train_features], dtype=torch.long)
        all_input_mask_external = torch.tensor([f.input_mask_external for f in train_features], dtype=torch.long)
        all_added_input_mask_external = torch.tensor([f.added_input_mask_external for f in train_features], dtype=torch.long)
        all_segment_ids_external = torch.tensor([f.segment_ids_external for f in train_features], dtype=torch.long)

        all_input_ids_origin = torch.tensor([f.input_ids_origin for f in train_features], dtype=torch.long)
        all_input_mask_origin = torch.tensor([f.input_mask_origin for f in train_features], dtype=torch.long)
        all_added_input_mask_origin = torch.tensor([f.added_input_mask_origin for f in train_features], dtype=torch.long)
        all_segment_ids_origin = torch.tensor([f.segment_ids_origin for f in train_features], dtype=torch.long)

        all_img_feats = torch.stack([f.img_feat for f in train_features]) # Assuming img_feat is a tensor
        all_image_ti_feat = torch.stack([f.img_ti_feat for f in train_features]) # Assuming img_ti_feat is a tensor

        all_label_ids_external = torch.tensor([f.label_ids_external for f in train_features], dtype=torch.long)
        all_label_ids_origin = torch.tensor([f.label_ids_origin for f in train_features], dtype=torch.long)
        all_auxlabel_ids_external = torch.tensor([f.auxlabel_ids_external for f in train_features], dtype=torch.long)
        all_auxlabel_ids_origin = torch.tensor([f.auxlabel_ids_origin for f in train_features], dtype=torch.long)

        # Create TensorDataset with ALL tensors in the correct order
        # Ensure this order matches what your model's forward function expects
        train_data = TensorDataset(
            all_input_ids_external, all_input_mask_external, all_added_input_mask_external, all_segment_ids_external,
            all_input_ids_origin, all_input_mask_origin, all_added_input_mask_origin, all_segment_ids_origin,
            all_img_feats, all_image_ti_feat,
            all_label_ids_external, all_label_ids_origin,
            all_auxlabel_ids_external, all_auxlabel_ids_origin
        )
        torch.save(train_data, train_dataloader_save_path)
        print(f"Saved train_data TensorDataset to {train_dataloader_save_path}")
    else:
        print("Loading the train_data (TensorDataset)...")
        # Important: Set weights_only=False if loading pickled objects like Tensors/Datasets
        # Be cautious with weights_only=False if the source is untrusted.
        train_data = torch.load(train_dataloader_save_path, weights_only=False)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    # Make sure args.train_batch_size is defined
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # --- Dev/Eval Data Loading ---
    dev_eval_examples = processor.get_dev_examples(args.data_dir) # Assuming processor is defined

    if not os.path.exists(dev_dataloader_save_path):
        print("Creating dev_eval_data TensorDataset...")
        dev_eval_features = convert_mm_examples_to_features(
            dev_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)

        # Extract tensors for ALL fields from SBInputFeatures for dev set
        all_input_ids_external = torch.tensor([f.input_ids_external for f in dev_eval_features], dtype=torch.long)
        all_input_mask_external = torch.tensor([f.input_mask_external for f in dev_eval_features], dtype=torch.long)
        all_added_input_mask_external = torch.tensor([f.added_input_mask_external for f in dev_eval_features], dtype=torch.long)
        all_segment_ids_external = torch.tensor([f.segment_ids_external for f in dev_eval_features], dtype=torch.long)

        all_input_ids_origin = torch.tensor([f.input_ids_origin for f in dev_eval_features], dtype=torch.long)
        all_input_mask_origin = torch.tensor([f.input_mask_origin for f in dev_eval_features], dtype=torch.long)
        all_added_input_mask_origin = torch.tensor([f.added_input_mask_origin for f in dev_eval_features], dtype=torch.long)
        all_segment_ids_origin = torch.tensor([f.segment_ids_origin for f in dev_eval_features], dtype=torch.long)

        all_img_feats = torch.stack([f.img_feat for f in dev_eval_features])
        all_image_ti_feat = torch.stack([f.img_ti_feat for f in dev_eval_features])

        all_label_ids_external = torch.tensor([f.label_ids_external for f in dev_eval_features], dtype=torch.long)
        all_label_ids_origin = torch.tensor([f.label_ids_origin for f in dev_eval_features], dtype=torch.long)
        all_auxlabel_ids_external = torch.tensor([f.auxlabel_ids_external for f in dev_eval_features], dtype=torch.long)
        all_auxlabel_ids_origin = torch.tensor([f.auxlabel_ids_origin for f in dev_eval_features], dtype=torch.long)

        # Create TensorDataset for dev set with ALL tensors
        dev_eval_data = TensorDataset(
            all_input_ids_external, all_input_mask_external, all_added_input_mask_external, all_segment_ids_external,
            all_input_ids_origin, all_input_mask_origin, all_added_input_mask_origin, all_segment_ids_origin,
            all_img_feats, all_image_ti_feat,
            all_label_ids_external, all_label_ids_origin,
            all_auxlabel_ids_external, all_auxlabel_ids_origin
        )
        torch.save(dev_eval_data, dev_dataloader_save_path)
        print(f"Saved dev_eval_data TensorDataset to {dev_dataloader_save_path}")
    else:
        print("Loading the dev_eval_data (TensorDataset)...")
        # Important: Set weights_only=False if loading pickled objects like Tensors/Datasets
        dev_eval_data = torch.load(dev_dataloader_save_path, weights_only=False)

    # Run prediction for full data
    dev_eval_sampler = SequentialSampler(dev_eval_data)
    # Make sure args.eval_batch_size is defined
    dev_eval_dataloader = DataLoader(dev_eval_data, sampler=dev_eval_sampler, batch_size=args.eval_batch_size)

    max_dev_f1 = 0.0
    best_dev_epoch = 0
    logger.info("***** Running training *****")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info("********** Epoch: " + str(train_idx) + " **********")
        logger.info("  Num examples = %d", len(train_examples)) # Make sure train_examples is accessible
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps) # Make sure this is calculated
        model.train()
        encoder.train() # Assuming encoder is your image encoder (e.g., ViT)
        # encoder.zero_grad() # Usually optimizer.zero_grad() is sufficient, but if encoder has separate params...
        optimizer.zero_grad() # Zero gradients before the batch loop
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)

            input_ids_external, segment_ids_external, input_mask_external, added_input_mask_external, \
            img_feats, image_ti_feat, \
            added_input_mask_origin, \
            input_ids_origin, segment_ids_origin, input_mask_origin, \
            label_ids_external, auxlabel_ids_external, \
            label_ids_origin, auxlabel_ids_origin = batch

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
            neg_log_likelihood = model(
                input_ids_external=input_ids_external,
                segment_ids_external=segment_ids_external,
                input_mask_external=input_mask_external,
                added_attention_mask_external=added_input_mask_external, # Assuming this maps
                visual_embeds_mean=img_mean, # From encoder
                visual_embeds_att=img_att,   # From encoder
                added_attention_mask_origin=added_input_mask_origin, # From data loader
                input_ids_origin=input_ids_origin, # From data loader
                segment_ids_origin=segment_ids_origin, # From data loader
                input_mask_origin=input_mask_origin, # From data loader
                image_decode=image_ti_feat, # Assuming this is image_ti_feat, clarify if not
                alpha=alpha,
                temp=temp, temp_lamb=temp_lamb, # Hyperparameters/calculated
                labels_external=label_ids_external, # From data loader
                auxlabels_external=auxlabel_ids_external, # From data loader
                labels_origin=label_ids_origin, # From data loader
                auxlabels_origin=auxlabel_ids_origin # From data loader
            )

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

        # --- Iterate through the dev dataloader ---
        # The dataloader now yields batches based on the updated TensorDataset order:
        # (input_ids_external, segment_ids_external, input_mask_external, added_input_mask_external,
        #  img_feats, image_ti_feat,
        #  added_input_mask_origin,
        #  input_ids_origin, segment_ids_origin, input_mask_origin,
        #  label_ids_external, auxlabel_ids_external,
        #  label_ids_origin, auxlabel_ids_origin)
        for batch in tqdm(dev_eval_dataloader, desc="Evaluating"):
            # --- Unpack the batch ---
            # Note: image_ti_feat is loaded but not explicitly used in this eval loop snippet.
            # If your model's 'image_decode' requires it, assign it accordingly.
            (input_ids_external, segment_ids_external, input_mask_external, added_input_mask_external,
             img_feats, _, # image_ti_feat loaded but not used here directly
             added_input_mask_origin,
             input_ids_origin, segment_ids_origin, input_mask_origin,
             label_ids_external, auxlabel_ids_external,
             label_ids_origin, auxlabel_ids_origin) = tuple(t.to(device) for t in batch)
            # Move only the necessary tensors to device for the model call
            # input_ids_external, segment_ids_external, input_mask_external, added_input_mask_external,
            # img_feats, added_input_mask_origin, input_ids_origin, segment_ids_origin, input_mask_origin,
            # label_ids_external, auxlabel_ids_external, label_ids_origin, auxlabel_ids_origin
            # are already on 'device' due to the tuple comprehension above.

            with torch.no_grad():
                # --- Encode image features ---
                imgs_f, img_mean, img_att = encoder(img_feats) # Encoder processes img_feats
                image_decode = None # Clarify if this should be image_ti_feat or something else
                # We pass None for labels/auxlabels during evaluation.
                predicted_label_seq_ids = model(
                    input_ids_external=input_ids_external,
                    segment_ids_external=segment_ids_external,
                    input_mask_external=input_mask_external,
                    added_attention_mask_external=added_input_mask_external, # Map name
                    visual_embeds_mean=img_mean, # From encoder
                    visual_embeds_att=img_att,   # From encoder
                    added_attention_mask_origin=added_input_mask_origin, # From data loader
                    input_ids_origin=input_ids_origin, # From data loader
                    segment_ids_origin=segment_ids_origin, # From data loader
                    input_mask_origin=input_mask_origin, # From data loader
                    image_decode=image_decode, # Often image_ti_feat or None during eval
                    alpha=alpha,
                    temp=temp, temp_lamb=temp_lamb, # Hyperparameters/calculated
                    labels_external=None, auxlabels_external=None, # None during evaluation
                    labels_origin=None, auxlabels_origin=None   # None during evaluation
                )
                # predicted_label_seq_ids is expected to hold the predictions,
                # likely matching the shape of input_ids_external or label_ids_external.

            # --- Process predictions ---
            logits = predicted_label_seq_ids # Alias for clarity based on usage

            # Move relevant tensors to CPU for processing
            # Use 'label_ids_external' for comparison as per original logic (assuming labels match external input)
            label_ids_external_cpu = label_ids_external.to('cpu').numpy()
            input_mask_external_cpu = input_mask_external.to('cpu').numpy()
            # logits is assumed to correspond to the external sequence predictions
            logits_cpu = logits.detach().cpu().numpy() # Detach just in case, though no_grad is used

            # --- Collect true/predicted labels for metrics ---
            # Iterate through the batch
            for i, mask in enumerate(input_mask_external_cpu): # Use external mask
                temp_1 = [] # True labels (text)
                temp_2 = [] # Predicted labels (text)
                tmp1_idx = [] # True label indices
                tmp2_idx = [] # Predicted label indices
                # Iterate through tokens in the sequence
                for j, m in enumerate(mask):
                    # Skip the first token (<s>)
                    if j == 0:
                        continue
                    # If mask is active (valid token)
                    if m:
                        true_label_idx = label_ids_external_cpu[i][j]
                        pred_label_idx = np.argmax(logits_cpu[i][j]) # Assuming logits are class scores
                        true_label_text = label_map.get(true_label_idx, "<unk>")
                        pred_label_text = label_map.get(pred_label_idx, "<unk>")

                        # Filter out special tokens "X" and "</s>" as in original code
                        if true_label_text != "X" and true_label_text != "</s>":
                            temp_1.append(true_label_text)
                            tmp1_idx.append(true_label_idx)
                            temp_2.append(pred_label_text)
                            tmp2_idx.append(pred_label_idx)
                    else:
                        # Stop at the first masked token (end of sequence)
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)

        # --- Calculate Metrics ---
        report = classification_report(y_true, y_pred, digits=4)
        sentence_list = []
        # Assuming processor._read_sbtsv exists and works as before
        dev_data, imgs, _ = processor._read_sbtsv(os.path.join(args.data_dir, "dev.txt"))
        for i in range(len(y_pred)): # Use y_pred or y_true, they should be the same length
            sentence = dev_data[i][0] # Assuming sentence is the first element
            sentence_list.append(sentence)

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        # Assuming evaluate function exists and takes these arguments correctly
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)

        logger.info("***** Dev Eval results *****")
        logger.info("\n%s", report)
        print("Overall: ", p, r, f1)
        F_score_dev = f1 # Assuming F1 score is the metric to track

        # --- Save Best Model ---
        if F_score_dev >= max_dev_f1:
            # Save a trained model and the associated configuration
            # Handle potential DataParallel modules
            model_to_save = model.module if hasattr(model, 'module') else model
            encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
            # Define output paths (ensure they are defined in your script context)
            # output_model_file, output_encoder_file, output_config_file = ...
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(encoder_to_save.state_dict(), output_encoder_file)
            # Save model config (ensure model_to_save.config is accessible)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

            # Save model configuration details
            model_config = {
                "bert_model": args.bert_model,
                "best_epoch": best_dev_epoch,
                "max_f1": max_dev_f1,
                "do_lower": args.do_lower_case,
                "max_seq_length": args.max_seq_length,
                "num_labels": len(label_list) + 1,
                "label_map": label_map # Save the updated label map
            }
            json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))

            # Update best metrics
            max_dev_f1 = F_score_dev
            best_dev_epoch = train_idx # Assuming train_idx is the current epoch variable

    # --- Log Final Best Results ---
    logger.info("**************************************************")
    logger.info("The best epoch on the dev set: %s", best_dev_epoch)
    logger.info("The best Overall-F1 score on the dev set: %s", max_dev_f1)
    logger.info('\n')

# loadmodel
if args.mm_model == 'MTCCMBert':
    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir='cache')
    model = EXTCModel(config, layer_num1=args.layer_num1,
                                layer_num2=args.layer_num2,
                                layer_num3=args.layer_num3,
                                num_labels_=num_labels, auxnum_labels = auxnum_labels)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)     
    pass               
else:
    print('please define your MNER Model')

if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    eval_examples = processor.get_test_examples(args.data_dir)
    test_dataloader_save_path = os.path.join(args.data_dir, "test_dataloader_dataset.pth") # Use os.path.join for consistency

    if not os.path.exists(test_dataloader_save_path):
        print("Creating eval_data TensorDataset...")
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)

        logger.info("***** Running Test Evaluation with the Best Model on the Dev Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        # --- Extract tensors for ALL fields from SBInputFeatures ---
        all_input_ids_external = torch.tensor([f.input_ids_external for f in eval_features], dtype=torch.long)
        all_input_mask_external = torch.tensor([f.input_mask_external for f in eval_features], dtype=torch.long)
        all_added_input_mask_external = torch.tensor([f.added_input_mask_external for f in eval_features], dtype=torch.long)
        all_segment_ids_external = torch.tensor([f.segment_ids_external for f in eval_features], dtype=torch.long)

        all_input_ids_origin = torch.tensor([f.input_ids_origin for f in eval_features], dtype=torch.long)
        all_input_mask_origin = torch.tensor([f.input_mask_origin for f in eval_features], dtype=torch.long)
        all_added_input_mask_origin = torch.tensor([f.added_input_mask_origin for f in eval_features], dtype=torch.long)
        all_segment_ids_origin = torch.tensor([f.segment_ids_origin for f in eval_features], dtype=torch.long)

        all_img_feats = torch.stack([f.img_feat for f in eval_features]) # Assuming img_feat is a tensor from image_process
        all_image_ti_feat = torch.stack([f.img_ti_feat for f in eval_features]) # Assuming img_ti_feat is a tensor

        all_label_ids_external = torch.tensor([f.label_ids_external for f in eval_features], dtype=torch.long)
        all_label_ids_origin = torch.tensor([f.label_ids_origin for f in eval_features], dtype=torch.long)
        all_auxlabel_ids_external = torch.tensor([f.auxlabel_ids_external for f in eval_features], dtype=torch.long)
        all_auxlabel_ids_origin = torch.tensor([f.auxlabel_ids_origin for f in eval_features], dtype=torch.long)

        # --- Create TensorDataset with ALL tensors ---
        # Order MUST match the evaluation loop's unpacking and potentially the model's forward method signature.
        # Based on the training loop and model signature, a likely order is:
        eval_data = TensorDataset(
            all_input_ids_external, all_segment_ids_external, all_input_mask_external, all_added_input_mask_external,
            all_img_feats, all_image_ti_feat, # img_feats for encoder, image_ti_feat
            all_added_input_mask_origin, # Added for model's added_attention_mask_origin if needed
            all_input_ids_origin, all_segment_ids_origin, all_input_mask_origin, # Origin inputs if needed
            all_label_ids_external, all_auxlabel_ids_external, # External labels if needed (though not used in eval call)
            all_label_ids_origin, all_auxlabel_ids_origin # Origin labels if needed (though not used in eval call)
            # image_decode (often image_ti_feat) is handled during model call
        )
        torch.save(eval_data, test_dataloader_save_path)
        print(f"Saved eval_data TensorDataset to {test_dataloader_save_path}")
    else:
        print("Loading the eval_data (TensorDataset)...")
        eval_data = torch.load(test_dataloader_save_path, weights_only=False) # Set appropriately

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    encoder.eval()
    # eval_loss, eval_accuracy = 0, 0 # These are calculated in the original loop but commented out logic removed
    # nb_eval_steps, nb_eval_examples = 0, 0 # Not used in the provided snippet
    y_true = []
    y_pred = []
    y_true_idx = []
    y_pred_idx = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "<pad>"

    # --- Iterate through the eval dataloader ---
    # The dataloader now yields batches based on the updated TensorDataset order:
    print("Starting evaluation loop...")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # --- Unpack the batch according to TensorDataset order ---
        # TensorDataset Order (based on creation above):
        # 0:input_ids_external, 1:segment_ids_external, 2:input_mask_external, 3:added_input_mask_external,
        # 4:img_feats, 5:image_ti_feat,
        # 6:added_input_mask_origin,
        # 7:input_ids_origin, 8:segment_ids_origin, 9:input_mask_origin,
        # 10:label_ids_external, 11:auxlabel_ids_external,
        # 12:label_ids_origin, 13:auxlabel_ids_origin

        # Unpack and move tensors to device
        (input_ids_external, segment_ids_external, input_mask_external, added_input_mask_external,
         img_feats, image_ti_feat, # image_ti_feat loaded, might be used for image_decode
         added_input_mask_origin,
         input_ids_origin, segment_ids_origin, input_mask_origin,
         label_ids_external, auxlabel_ids_external,
         label_ids_origin, auxlabel_ids_origin) = tuple(t.to(device) for t in batch)

        # image_decode: Decide based on your model's needs. Often it's image_ti_feat or derived from it.
        # If not used or set to None in model.forward during eval, you can keep it as None or pass image_ti_feat.
        # Following the pattern from dev eval and your model signature, passing image_ti_feat is common.
        # Let's assume image_decode = image_ti_feat for now. Adjust if your model expects None or something else.
        image_decode = image_ti_feat # Or image_decode = None if that's what the model expects during eval

        with torch.no_grad():
            # --- Encode image features ---
            imgs_f, img_mean, img_att = encoder(img_feats) # Encoder processes img_feats

            predicted_label_seq_ids = model(
                input_ids_external=input_ids_external,
                segment_ids_external=segment_ids_external,
                input_mask_external=input_mask_external,
                added_attention_mask_external=added_input_mask_external, # Map name
                visual_embeds_mean=img_mean, # From encoder
                visual_embeds_att=img_att,   # From encoder
                added_attention_mask_origin=added_input_mask_origin, # From data loader
                input_ids_origin=input_ids_origin, # From data loader (if used by model)
                segment_ids_origin=segment_ids_origin, # From data loader (if used by model)
                input_mask_origin=input_mask_origin, # From data loader (if used by model)
                image_decode=image_decode, # Often image_ti_feat during eval
                alpha=alpha,
                temp=temp, temp_lamb=temp_lamb, # Hyperparameters/calculated (ensure defined)
                labels_external=None, auxlabels_external=None, # None during evaluation
                labels_origin=None, auxlabels_origin=None   # None during evaluation
            )
            # predicted_label_seq_ids is expected to hold the predictions,
            # likely matching the shape of input_ids_external or label_ids_external.

        # --- Process predictions ---
        logits = predicted_label_seq_ids # Alias for clarity based on usage

        # Move relevant tensors to CPU for processing
        # Use 'label_ids_external' for comparison as it corresponds to 'input_ids_external'
        label_ids_external_cpu = label_ids_external.to('cpu').numpy()
        input_mask_external_cpu = input_mask_external.to('cpu').numpy()
        # logits is assumed to correspond to the external sequence predictions
        logits_cpu = logits.detach().cpu().numpy() # Detach just in case, though no_grad is used

        # --- Collect true/predicted labels for metrics ---
        # Iterate through the batch
        for i, mask in enumerate(input_mask_external_cpu): # Use external mask
            temp_1 = [] # True labels (text)
            temp_2 = [] # Predicted labels (text)
            tmp1_idx = [] # True label indices
            tmp2_idx = [] # Predicted label indices
            # Iterate through tokens in the sequence
            for j, m in enumerate(mask):
                # Skip the first token (<s>)
                if j == 0:
                    continue
                # If mask is active (valid token)
                if m:
                    true_label_idx = label_ids_external_cpu[i][j]
                    # Assuming logits contain class scores, get the predicted class index
                    pred_label_idx = np.argmax(logits_cpu[i][j])
                    true_label_text = label_map.get(true_label_idx, "<unk>")
                    pred_label_text = label_map.get(pred_label_idx, "<unk>")

                    # Filter out special tokens "X" and "</s>" as in original code
                    if true_label_text != "X" and true_label_text != "</s>":
                        temp_1.append(true_label_text)
                        tmp1_idx.append(true_label_idx)
                        temp_2.append(pred_label_text)
                        tmp2_idx.append(pred_label_idx)
                else:
                    # Stop at the first masked token (end of sequence)
                    break
            y_true.append(temp_1)
            y_pred.append(temp_2)
            y_true_idx.append(tmp1_idx)
            y_pred_idx.append(tmp2_idx)

    # --- Calculate Metrics and Save Results ---
    report = classification_report(y_true, y_pred, digits=4)

    sentence_list = []
    # Assuming processor._read_sbtsv exists and works as before
    test_data, imgs, _ = processor._read_sbtsv(os.path.join(args.data_dir, "test.txt"))
    output_pred_file = os.path.join(args.output_dir, "mtmner_pred.txt")
    with open(output_pred_file, 'w') as fout: # Use 'with' for safe file handling
        for i in range(len(y_pred)): # Use y_pred or y_true, they should be the same length
            sentence = test_data[i][0] # Assuming sentence is the first element
            sentence_list.append(sentence)
            img = imgs[i]
            samp_pred_label = y_pred[i]
            samp_true_label = y_true[i]
            fout.write(img + '\n')
            fout.write(' '.join(sentence) + '\n') # Assuming sentence is a list of words
            fout.write(' '.join(samp_pred_label) + '\n')
            fout.write(' '.join(samp_true_label) + '\n' + '\n')

    reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
    # Assuming evaluate function exists and takes these arguments correctly
    acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
    print("Overall: ", p, r, f1)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)
        writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')