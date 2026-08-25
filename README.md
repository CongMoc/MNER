# MNER Vietnamese

## Guide
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Training](#training)
  - [External Context model (main)](#external-context-model-main)
  - [Other training modes](#other-training-modes)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.7+. To run our code, install:

```
pip install -r requirements.txt
```

The External Context model uses a ViT image encoder pulled automatically from HuggingFace
(`--vit_model`, default `google/vit-large-patch16-224-in21k`), so no manual weight download is
needed.

## Datasets

Public sources used for this project:
- VLSP2016 / VLSP2018: [vlsp.org.vn](https://vlsp.org.vn/vlsp2021/eval/ner), also mirrored with
  images at [jester6136/vlsp_all](https://huggingface.co/datasets/jester6136/vlsp_all) on HuggingFace
  (`origin/` = text only, `origin+image/` = text + a `ner_image.zip` of per-sentence images).
- NewsMNER: images + text at [Taurus2304/MNER](https://huggingface.co/datasets/Taurus2304/MNER)
  on HuggingFace.

Datasets are **not** committed to this repo (see `.gitignore`: `data/`, `sample_data/`,
`data_export/`) — download/build them locally before training.

### Data format

Each split (`train.txt` / `dev.txt` / `test.txt`) is CoNLL-style, one token per line,
`token<TAB>label`, sentences separated by a blank line, each sentence preceded by an
`IMGID:<image_filename_without_extension>` header line:

```
IMGID:example_001
Chủ	B-PER
tịch	O
...
<EOS>	E
Chủ_tịch	E
là	E
...

```

For the **External Context model**, each sentence's original tokens are followed by an
`<EOS>` token (label `E`) and then the external-context tokens (also labeled `E` — the model
only needs to know which span is context vs. the sentence to tag). Sample data illustrating
this format is under `sample_data/`.

`LABELS` (env var, comma-separated, read by every training script) must list every tag that
appears in your data, plus the fixed housekeeping tags `X`, `<s>`, `</s>` — and `E` for the
External Context model specifically. Example for VLSP2016-style 4-entity data:

```
export LABELS="B-LOC,B-MISC,B-ORG,B-PER,I-LOC,I-MISC,I-ORG,I-PER,O,E,X,<s>,</s>"
```

## Training

### External Context model (main)

`train/external_context/train_external_context_xlmr_vit.py` trains the flagship model:
an XLM-R-large text encoder + ViT image encoder, cross-attention between the original
sentence and its external context, with a CRF tagging head.

```bash
export LABELS="B-LOC,B-MISC,B-ORG,B-PER,I-LOC,I-MISC,I-ORG,I-PER,O,E,X,<s>,</s>"

python train/external_context/train_external_context_xlmr_vit.py \
    --do_train \
    --do_eval \
    --data_dir "path/to/your/dataset"      `# dir with train.txt / dev.txt / test.txt` \
    --path_image "path/to/your/dataset/images" \
    --output_dir "output/my_external_context_run" \
    --bert_model "xlm-roberta-large" \
    --vit_model "google/vit-large-patch16-224-in21k" \
    --image_source crawled \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --learning_rate 2.2e-5 \
    --warmup_proportion 0.4 \
    --max_seq_length 256 \
    --cache_dir cache \
    --seed 37
```

Key flags:
- `--data_dir`: folder with `train.txt`/`dev.txt`/`test.txt` in the format above.
- `--path_image`: folder with per-sentence images. What's expected inside depends on
  `--image_source`:
  - `crawled` (default): one image per sentence, filename = the `IMGID:` value + `.jpg`.
  - `random`: same folder, a random other image from it is substituted per sentence
    (ablation: does the *specific* image matter, or just *an* image).
  - `blank`: folder just needs one `background.jpg` (ablation: no visual signal at all).
  - `generated`: one image per sentence named `<split>-<index>.jpg` (e.g. `train-0.jpg`,
    matching example order in that split's `.txt` file) — e.g. text-to-image generated from
    the sentence, used to test whether a *real* photo is necessary.
- `--bert_model` / `--vit_model`: any HuggingFace model id compatible with `RobertaModel` /
  `ViTModel` respectively.
- Output: `output_dir/pytorch_model.bin` (cross-attention + CRF head),
  `output_dir/pytorch_encoder.bin` (fine-tuned text+image encoders), `output_dir/eval_results.txt`.

To evaluate cross-dataset (train on one dataset, test on another's `test.txt` with a
compatible label set), add `--eval_data_dir path/to/other/dataset`.

### Other training modes

- `train/text_only/train_xlmr_crf.py` — text-only XLM-R/PhoBERT + CRF, no images at all.
  Supports an optional `--init_checkpoint <prev_run>/pytorch_model.bin` to continue training
  the same model on a second dataset (continual fine-tuning) instead of starting from the
  base pretrained encoder — requires the second dataset's `LABELS` to match the first's in
  both count and order.

This follows the same `--do_train --do_eval --data_dir ... --output_dir ...` shape as the
External Context example above; check its `argparse` block for its exact flags before running.

The repo also has older, ResNet-152-backed training entrypoints
(`train/without_external_context/train_pixelcnn_cl.py`, `train/baselines/train_umt.py`,
`train/ablations/train_pixelcnn_wo_cl.py`) that predate the External Context / ViT work and
are no longer part of the maintained pipeline.
