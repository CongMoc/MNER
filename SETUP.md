# Setup

## Directory layout

```
train/
├── without_external_context/
│   ├── train_pixelcnn_cl.py         # UMT + PixelCNN + contrastive loss
│   └── train_pixelcnn_cl_mbert.py   # same, mBERT backbone
├── external_context/
│   └── train_external_context.py    # above + external context branch (retrieval + re-ranking)
├── ablations/
│   └── train_pixelcnn_wo_cl.py      # no contrastive loss
├── baselines/
│   ├── train_umt.py
│   ├── train_umt_mbert.py
│   ├── train_maf.py
│   ├── train_cross_attention_softmax.py
│   ├── train_cross_attention_softmax_gate.py
│   ├── train_cross_attention_crf.py
│   ├── train_cross_attention_crf_gate.py
│   └── train_cross_attention_crf_gate_cl.py
└── legacy/
    ├── train_EXCT_draft.py          # superseded by external_context/train_external_context.py
    └── train_maf_legacy_broken.py   # imports a class that no longer exists, kept for reference only
```

## Requirements

- Python 3.8/3.9
- CUDA GPU (ResNet-152 + PhoBERT on CPU is not practical)

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

requirements.txt: transformers, seqeval, boto3, pytorch-crf==0.7.2, pytorch_pretrained_bert==0.4.0, torchvision.

## Pretrained ResNet-152

```bash
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth -O modules/resnet/resnet152.pth
```

Matches `--resnet_root modules/resnet`.

## Data

Every model here is multimodal, text + image. There's no text-only path.

- Text: `train.txt` / `dev.txt` / `test.txt` in CoNLL format, one directory (`--data_dir`). See `sample_data/VLSP/VLSP2016/`.
- Images: a directory of images keyed by the `IMGID:` in the text files (`--path_image`). Download the matching `ner_image.zip` (e.g. `origin+image/VLSP2016/ner_image.zip`) and unzip separately — text-only downloads won't work with any script under `train/`.

Set LABELS to match the dataset, e.g. VLSP2016:

```bash
export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,X,<s>,</s>"
```

VLSP2018/VLSP2021 use different label sets, see README.md.

## Training

```bash
export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,X,<s>,</s>"

python train/without_external_context/train_pixelcnn_cl.py \
    --do_train \
    --do_eval \
    --output_dir output/vlsp2016_run1 \
    --bert_model "vinai/phobert-base-v2" \
    --alpha 0.5 --beta 0.5 --sigma 0.005 --theta 0.05 \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 1 \
    --weight_decay_pixelcnn 0.00005 \
    --lr_pixelcnn 0.001 \
    --learning_rate 2.2e-5 \
    --data_dir "sample_data/VLSP/VLSP2016" \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --path_image "<path-to-image-dir>" \
    --task_name "vlsp2016" \
    --resnet_root "modules/resnet" \
    --cache_dir "cache" \
    --max_seq_length 256 \
    --seed 37
```

Checkpoints and config land in `--output_dir`.

## Reproducibility

Seeding covers `random`, `numpy`, and `torch` CPU/GPU, with cuDNN determinism forced on:

```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This costs a bit of speed. For reported numbers, run 3-5 seeds and report mean ± std rather than a single run.
