import os
import sys
import argparse

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from llm_ner_common import load_jsonl, build_system_prompt, build_messages

IGNORE_INDEX = -100


class NERSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, system_prompt, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        entities = [tuple(e) for e in row['entities']]
        messages = build_messages(self.system_prompt, row['sentence'], entities)
        prompt_messages = messages[:-1]

        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_ids = self.tokenizer(full_text, add_special_tokens=False, truncation=True,
                                   max_length=self.max_length)['input_ids']
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, truncation=True,
                                     max_length=self.max_length)['input_ids']
        prefix_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        for i in range(prefix_len):
            labels[i] = IGNORE_INDEX

        return {'input_ids': full_ids, 'labels': labels}


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x['input_ids']) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            n_pad = max_len - len(x['input_ids'])
            input_ids.append(x['input_ids'] + [self.pad_token_id] * n_pad)
            labels.append(x['labels'] + [IGNORE_INDEX] * n_pad)
            attn.append([1] * len(x['input_ids']) + [0] * n_pad)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
        }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--data_dir', required=True, help='dir with train.jsonl/dev.jsonl from prepare_llm_ner_data.py')
    ap.add_argument('--labels', required=True, help='comma-separated entity type list (no B-/I- prefix)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--num_train_epochs', type=float, default=3.0)
    ap.add_argument('--train_batch_size', type=int, default=8)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=2)
    ap.add_argument('--learning_rate', type=float, default=2e-4)
    ap.add_argument('--max_length', type=int, default=512)
    ap.add_argument('--seed', type=int, default=37)
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    entity_types = args.labels.split(',')
    system_prompt = build_system_prompt(entity_types)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_jsonl(os.path.join(args.data_dir, 'train.jsonl'))
    dev_rows = load_jsonl(os.path.join(args.data_dir, 'dev.jsonl'))
    train_ds = NERSFTDataset(train_rows, tokenizer, system_prompt, args.max_length)
    dev_ds = NERSFTDataset(dev_rows, tokenizer, system_prompt, args.max_length)
    collator = Collator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,  # float < 1 -> interpreted as ratio of total training steps
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        seed=args.seed,
        report_to=[],
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
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'adapter'))
    print(f"saved LoRA adapter to {os.path.join(args.output_dir, 'adapter')}")
