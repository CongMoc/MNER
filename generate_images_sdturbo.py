#!/usr/bin/env python3
"""Generate one SD-turbo image per example, prompted with the full sentence plus its
extracted named entities. Saves <out_dir>/<split>-<index>.jpg, matching the guid scheme
("%s-%s" % (set_type, i)) that MNERProcessor uses -- convert_mm_examples_to_features
looks up exactly f"{example.guid}.jpg" when --image_source generated.

Usage:
  python generate_images_sdturbo.py --data_dir <dir with train/dev/test.txt> \
      --labels B-LOC,B-MISC,B-ORG,B-PER --out_dir <dir> [--cache_dir cache]
"""
import argparse
import os

import torch
from diffusers import AutoPipelineForText2Image

SPECIAL_TOKENS = ["️", "‍", "​", "\x92"]
URL_PREFIX = "http"


def read_conll(path):
    """Yields (tokens, labels) per sentence. In *-EXT/converted files, a literal "<EOS>"/
    "<eos>" token marks the start of appended external-context text for that sentence --
    not part of the original sentence, so it shouldn't feed entity/prompt extraction. Once
    seen, skip the rest of that sentence's lines (but keep the tokens already collected) and
    resume normal reading at the next blank line.
    """
    sentences = []
    tokens, labels = [], []
    in_context_tail = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("IMGID:") or line.startswith("IMID:"):
                continue
            if line.strip() == "":
                if tokens:
                    sentences.append((tokens, labels))
                tokens, labels = [], []
                in_context_tail = False
                continue
            if in_context_tail:
                continue
            stripped = line.rstrip("\n").rstrip("\r")
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            if token in ("<EOS>", "<eos>"):
                in_context_tail = True
                continue
            if token == "" or token.isspace() or token in SPECIAL_TOKENS or token.startswith(URL_PREFIX):
                token = "<unk>"
            tokens.append(token.replace("_", " "))
            labels.append(label)
        if tokens:
            sentences.append((tokens, labels))
    return sentences


def extract_entities(tokens, labels, entity_types):
    entities = []
    cur_tokens, cur_type = [], None
    for tok, lab in zip(tokens, labels):
        lab_type = lab[2:] if len(lab) > 2 and lab[1] == "-" else None
        is_start = lab.startswith("B-") and lab_type in entity_types
        is_continue = lab.startswith("I-") and cur_tokens and lab_type == cur_type
        if is_start:
            if cur_tokens:
                entities.append((cur_type, " ".join(cur_tokens)))
            cur_tokens, cur_type = [tok], lab_type
        elif is_continue:
            cur_tokens.append(tok)
        else:
            if cur_tokens:
                entities.append((cur_type, " ".join(cur_tokens)))
            cur_tokens, cur_type = [], None
    if cur_tokens:
        entities.append((cur_type, " ".join(cur_tokens)))
    return entities


def build_prompt(tokens, labels, entity_types):
    sentence = " ".join(tokens)
    entities = extract_entities(tokens, labels, entity_types)
    if entities:
        ent_str = "; ".join(f"{typ}: {text}" for typ, text in entities)
        return f"{sentence} Entities: {ent_str}"
    return sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--labels", required=True, help="comma-separated B-tag entity types, e.g. B-LOC,B-MISC,B-ORG,B-PER")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--num_inference_steps", type=int, default=1)
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    entity_types = {t.split("-", 1)[1] if t.startswith(("B-", "I-")) else t for t in args.labels.split(",") if t}

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=args.cache_dir,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    for split in ["train", "dev", "test"]:
        path = os.path.join(args.data_dir, f"{split}.txt")
        if not os.path.exists(path):
            continue
        sentences = read_conll(path)
        print(f"[{split}] {len(sentences)} examples", flush=True)

        todo = []
        for i, (tokens, labels) in enumerate(sentences):
            guid = f"{split}-{i}"
            out_path = os.path.join(args.out_dir, f"{guid}.jpg")
            if os.path.exists(out_path):
                continue
            todo.append((guid, build_prompt(tokens, labels, entity_types)))
        print(f"[{split}] {len(todo)} images to generate ({len(sentences) - len(todo)} already done)", flush=True)

        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            gen = torch.Generator(device=device).manual_seed(args.seed + start)
            images = pipe(
                prompt=[p for _, p in batch],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
            ).images
            for (guid, _), img in zip(batch, images):
                img.save(os.path.join(args.out_dir, f"{guid}.jpg"), quality=90)
            done = start + len(batch)
            if done % (args.batch_size * 25) == 0 or done == len(todo):
                print(f"[{split}] {done}/{len(todo)}", flush=True)

    with open(os.path.join(args.out_dir, ".done"), "w") as f:
        f.write("ok\n")
    print("IMAGE GENERATION DONE")


if __name__ == "__main__":
    main()
