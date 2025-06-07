import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def preprocess_batch(batch, tokenizer, eos_token, max_length=1024):
    all_input_ids, all_attention_masks, all_labels = [], [], []

    for inp, out in zip(batch["input"], batch["output"]):
        prefix_text = f"Улучши промпт: <prompt>{inp}</prompt>"
        prefix_ids = tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"][0].tolist()

        out_ids = tokenizer(
            out + eos_token,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"][0].tolist()

        input_ids = prefix_ids + out_ids
        labels = [-100] * len(prefix_ids) + out_ids

        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

        attention_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def compute_eos_accuracy(eval_pred, eos_id):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask_labels = labels == eos_id
    mask_preds = np.roll(mask_labels, shift=-1, axis=1)
    mask_preds[:, -1] = False

    if mask_preds.sum() == 0:
        return {"eos_accuracy": 0.0}

    acc = (preds[mask_preds] == eos_id).mean(dtype=np.float64)
    return {"eos_accuracy": float(acc)}


def main(model_name: str, data_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.label_pad_token_id = -100
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()

    ds = load_dataset("json", data_files={"full": data_path}, split="full")
    ds = ds.train_test_split(test_size=0.01, seed=42)

    tokenize_fn = lambda batch: preprocess_batch(batch, tokenizer, eos_token)
    tds = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    args = TrainingArguments(
        output_dir="qwen3-lora-stage1",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_steps=4000,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_strategy="no",
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=200,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tds["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_eos_accuracy(p, eos_id),
    )

    print("\n─── Stage 1: training on Dataset.train ───")
    trainer.train()

    print("\nStage 1 → metrics on Dataset.test")
    print(trainer.evaluate(tds["test"]))

    model.save_pretrained("qwen3-lora-adapter")
    print("Adapter saved to qwen3-lora-adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA-fine-tune Qwen3 (Stage 1 only)")
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name, e.g. 'Qwen/Qwen3-0.6B'"
    )
    parser.add_argument(
        "--data-path", required=True,
        help="JSON/JSONL с полями input/output для Stage 1"
    )
    args = parser.parse_args()

    main(args.model_name, args.data_path)
