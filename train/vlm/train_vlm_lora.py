import os
import sys
import argparse

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from vlm_ner_common import load_jsonl, build_system_prompt, build_messages

IGNORE_INDEX = -100


def load_image(image_dir, image_name):
    path = os.path.join(image_dir, image_name)
    if not os.path.exists(path):
        path = os.path.join(image_dir, 'background.jpg')
    return Image.open(path).convert('RGB')


class VLMNERDataset(Dataset):
    """Each item is tokenized individually (unpadded) so the collator can
    right-pad per-batch and mask exactly the prompt-token prefix out of the
    loss, mirroring train_llm_lora.py's text-only prefix-masking approach.
    """

    def __init__(self, rows, processor, system_prompt, image_dir, max_length=1024):
        self.rows = rows
        self.processor = processor
        self.system_prompt = system_prompt
        self.image_dir = image_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        entities = [tuple(e) for e in row['entities']]
        image = load_image(self.image_dir, row['image'])

        full_messages = build_messages(self.system_prompt, row['sentence'], image, entities)
        prompt_messages = build_messages(self.system_prompt, row['sentence'], image, None)

        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_enc = self.processor(text=[full_text], images=[image], return_tensors='pt')
        prompt_enc = self.processor(text=[prompt_text], images=[image], return_tensors='pt')

        input_ids = full_enc['input_ids'][0][:self.max_length]
        prefix_len = min(prompt_enc['input_ids'].shape[1], input_ids.shape[0])

        labels = input_ids.clone()
        labels[:prefix_len] = IGNORE_INDEX

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': full_enc['pixel_values'],
            'image_grid_thw': full_enc['image_grid_thw'][0],
        }


class VLMCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(x['input_ids'].shape[0] for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            n_pad = max_len - x['input_ids'].shape[0]
            input_ids.append(torch.cat([x['input_ids'], torch.full((n_pad,), self.pad_token_id, dtype=torch.long)]))
            labels.append(torch.cat([x['labels'], torch.full((n_pad,), IGNORE_INDEX, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(x['input_ids'].shape[0], dtype=torch.long), torch.zeros(n_pad, dtype=torch.long)]))

        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attn),
            'pixel_values': torch.cat([x['pixel_values'] for x in batch], dim=0),
            'image_grid_thw': torch.stack([x['image_grid_thw'] for x in batch]),
        }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--data_dir', required=True, help='dir with train.jsonl/dev.jsonl from prepare_vlm_ner_data.py')
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--labels', required=True, help='comma-separated entity type list (no B-/I- prefix)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--num_train_epochs', type=float, default=3.0)
    ap.add_argument('--train_batch_size', type=int, default=2)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=8)
    ap.add_argument('--learning_rate', type=float, default=1e-4)
    ap.add_argument('--max_length', type=int, default=1536)
    ap.add_argument('--seed', type=int, default=37)
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    entity_types = args.labels.split(',')
    system_prompt = build_system_prompt(entity_types)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    # max_pixels kept well under max_length: at up to ~768 image tokens plus
    # system prompt + sentence + response, truncation (if it ever triggers)
    # only trims the tail of the response, never the image token block.
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_jsonl(os.path.join(args.data_dir, 'train.jsonl'))
    dev_rows = load_jsonl(os.path.join(args.data_dir, 'dev.jsonl'))
    train_ds = VLMNERDataset(train_rows, processor, system_prompt, args.image_dir, args.max_length)
    dev_ds = VLMNERDataset(dev_rows, processor, system_prompt, args.image_dir, args.max_length)
    collator = VLMCollator(processor.tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(os.path.join(args.output_dir, 'adapter'))
    processor.save_pretrained(os.path.join(args.output_dir, 'adapter'))
    print(f"saved LoRA adapter to {os.path.join(args.output_dir, 'adapter')}")
