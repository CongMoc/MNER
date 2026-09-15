"""Step 0 (run once, before training) -- for every image referenced by the
dataset, produce the three text views the ITA paper's input concatenation needs:

  LA  (Local Alignment):  object tags from a zero-shot detector (OWL-ViT here)
  GA  (Global Alignment): image captions (BLIP-2), 5 beams joined by [SEP]
  OCA (Optical Character Alignment): OCR text (PaddleOCR)

Output: visual_contexts.json = {img_id: {"la": ..., "ga": ..., "oca": ...}}.
Caching: any img_id already present in an existing output file is skipped, so an
interrupted run just resumes. Failures are logged to preprocessing_errors.log and
that image gets empty strings for whichever field(s) failed -- never crashes the
whole run over one bad file.

NOTE on LA: OWL-ViT is a zero-shot detector over a *given* list of candidate
label strings -- it doesn't invent free-form "attribute + noun" phrases on its
own. This script queries it with the COCO-80 category names (COCO_CLASSES below)
and keeps the object noun only (no separate "attr1 obj1" attribute prefix, since
OWL-ViT has no attribute vocabulary to draw one from without a captioning-style
model doing the attribute work instead) -- see the README note before assuming
LA output looks exactly like the paper's "attr1 obj1, attr2 obj2" example.

Usage: python3 preprocess_visual_contexts.py --data_dir <dir with train/dev/test.txt> \
    --images_dir <dir with {img_id}.jpg> --out visual_contexts.json [--limit N]
"""
import argparse
import json
import logging
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_base_conll

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess_visual_contexts")

ERROR_LOG = "preprocessing_errors.log"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "flag", "building", "sign", "crowd", "stage", "microphone",
]


def collect_img_ids(data_dir):
    ids = set()
    for split in ("train", "dev", "test"):
        p = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(p):
            for sent in read_base_conll(p):
                if sent["img_id"]:
                    ids.add(sent["img_id"])
    return sorted(ids)


def resolve_image_path(images_dir, img_id):
    for ext in ("", ".jpg", ".jpeg", ".png"):
        p = os.path.join(images_dir, img_id + ext) if ext or not os.path.splitext(img_id)[1] else os.path.join(images_dir, img_id)
        if os.path.exists(p):
            return p
    return None


class LocalAlignment:
    """OWL-ViT zero-shot object tagging -- see module docstring's LA note."""

    def __init__(self, device, model_name="google/owlvit-base-patch32", top_k=5, score_thresh=0.1):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.top_k = top_k
        self.score_thresh = score_thresh

    @torch.no_grad()
    def run(self, image):
        inputs = self.processor(text=[COCO_CLASSES], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.score_thresh, target_sizes=target_sizes)[0]
        scores, labels = results["scores"], results["labels"]
        order = scores.argsort(descending=True)[: self.top_k]
        tags = [COCO_CLASSES[labels[i].item()] for i in order]
        # de-dupe, keep confidence order
        seen, out = set(), []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return ", ".join(out)


class GlobalAlignment:
    """BLIP-2 captioning, 5 captions joined by [SEP] (per spec).

    Two fixes versus the naive call the spec implies, both confirmed by direct
    testing against this transformers version (4.46.3):
      1. Blip2Processor needs an explicit text="" (not omitted) to build
         input_ids containing the 32 image-placeholder tokens generate() needs;
         with pixel_values alone it emits input_ids with NO placeholders and
         crashes inside modeling_blip_2.py's special_image_mask indexing.
      2. Passing max_length=20 then means "20 tokens total", but input_ids is
         already 33 tokens (32 placeholders + BOS) long, making the requested
         max_new tokens negative -- must use max_new_tokens instead.
      3. num_beams=5 + num_return_sequences=5 in one generate() call hits a
         confirmed library bug in this version: internally it does
         `torch.cat([bos_tokens, outputs])` where bos_tokens keeps batch size 1
         but outputs has batch size 5 (one per returned sequence), raising a
         shape-mismatch RuntimeError. Verified this reproduces on a plain
         Blip2ForConditionalGeneration.generate call, i.e. it's not something
         wrong with how this script invokes it. Worked around by doing 5
         independent nucleus-sampling calls (each with the default
         num_return_sequences=1, avoiding the broken code path) instead of one
         beam-search call asking for 5 sequences at once -- still "5 captions
         from the same model, joined by [SEP]", just sampled rather than the
         top-5 beams specifically.
    """

    def __init__(self, device, model_name="Salesforce/blip2-opt-2.7b", num_captions=5, max_new_tokens=20):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        self.processor = Blip2Processor.from_pretrained(model_name)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.num_captions = num_captions
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def run(self, image):
        inputs = self.processor(images=image, text="", return_tensors="pt").to(self.device, self.model.dtype)
        captions = []
        for _ in range(self.num_captions):
            out_ids = self.model.generate(
                **inputs, do_sample=True, top_p=0.9, temperature=1.0,
                max_new_tokens=self.max_new_tokens)
            cap = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            if cap:
                captions.append(cap)
        return " [SEP] ".join(captions)


class OpticalCharAlignment:
    """PaddleOCR -- chosen (per spec) over Tesseract for better Vietnamese text
    support."""

    def __init__(self, lang="vi"):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def run(self, image_path):
        result = self.ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for line in result[0]]
        return " ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", default="visual_contexts.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_la", action="store_true")
    ap.add_argument("--skip_ga", action="store_true")
    ap.add_argument("--skip_oca", action="store_true")
    ap.add_argument("--num_captions", type=int, default=5,
                     help="BLIP-2 captions per image joined by [SEP] (paper: 5; reduced to cut preprocessing time is fine)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_ids = collect_img_ids(args.data_dir)
    if args.limit:
        img_ids = img_ids[: args.limit]
    logger.info("%d unique img_ids referenced by the dataset", len(img_ids))

    contexts = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            contexts = json.load(f)
        logger.info("resuming: %d already cached", len(contexts))

    la = None if args.skip_la else LocalAlignment(device)
    ga = None if args.skip_ga else GlobalAlignment(device, num_captions=args.num_captions)
    oca = None if args.skip_oca else OpticalCharAlignment()

    err_f = open(ERROR_LOG, "a", encoding="utf-8")
    todo = [i for i in img_ids if i not in contexts]
    logger.info("%d to process", len(todo))

    for i, img_id in enumerate(tqdm(todo, desc="visual contexts")):
        path = resolve_image_path(args.images_dir, img_id)
        entry = {"la": "", "ga": "", "oca": ""}
        if path is None:
            err_f.write(f"{img_id}\timage not found\n")
            contexts[img_id] = entry
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            err_f.write(f"{img_id}\tfailed to open: {e}\n")
            contexts[img_id] = entry
            continue

        if la is not None:
            try:
                entry["la"] = la.run(image)
            except Exception as e:
                err_f.write(f"{img_id}\tLA failed: {e}\n")
        if ga is not None:
            try:
                entry["ga"] = ga.run(image)
            except Exception as e:
                err_f.write(f"{img_id}\tGA failed: {e}\n")
        if oca is not None:
            try:
                entry["oca"] = oca.run(path)
            except Exception as e:
                err_f.write(f"{img_id}\tOCA failed: {e}\n")

        contexts[img_id] = entry
        if (i + 1) % 200 == 0:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(contexts, f, ensure_ascii=False)
            err_f.flush()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False)
    err_f.close()
    logger.info("DONE. %d img_ids in %s", len(contexts), args.out)


if __name__ == "__main__":
    main()
