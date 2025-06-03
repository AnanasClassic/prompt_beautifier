import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


def main():
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    data_path = "data/prompts.jsonl"  # Убедитесь: каждая строка вида {"input": "...", "output": "..."}

    # ─── 1) Токенизатор ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # pad_token = eos_token, чтобы не было ошибок в датаколлаторе
    tokenizer.pad_token = tokenizer.eos_token

    # ─── 2) Базовая модель в fp16 на GPU ────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("cuda")

    # Обязательно обновим конфиг модели, чтобы DataCollatorForSeq2Seq знал, что 
    # pad_token_id = tokenizer.pad_token_id, label_pad_token_id = -100
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.label_pad_token_id = -100

    # ─── 3) Подключаем LoRA (PEFT) ─────────────────────────────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # ─── 4) Загружаем датасет (JSON Lines) ────────────────────────────────────
    # Файл data/prompts.jsonl должен содержать по одной паре на строку:
    # {"input": "сырой промт", "output": "улучшенный промт"}
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")

    # ─── 5) Препроцессинг: склеиваем "input<eos>output" и формируем labels───
    def preprocess(batch):
        input_texts = batch["input"]      # список строк
        output_texts = batch["output"]    # список строк

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for inp, out in zip(input_texts, output_texts):
            # 1) Собираем «полну́ю» строку: "<сырой промт><eos><улучшенный промт>"
            full_text = inp + tokenizer.eos_token + out

            # 2) Токенизируем полный текст (может обрезаться до max_length=512)
            tokenized_full = tokenizer(
                full_text,
                truncation=True,
                max_length=512,
            )

            # 3) Токенизируем только «сырой» промт, чтобы понять длину input-сегмента
            tokenized_input = tokenizer(
                inp,
                truncation=True,
                max_length=512,
            )
            len_input_ids = len(tokenized_input["input_ids"])

            input_ids = tokenized_full["input_ids"]
            attention_mask = tokenized_full["attention_mask"]

            # 4) Строим labels:
            # для позиций, соответствующих «сырому» промту (0..len_input_ids-1), ставим -100
            # дальше идут ID токенов «улучшенного» текста (смотри tokenized_full["input_ids"][len_input_ids:])
            labels = [-100] * len_input_ids + tokenized_full["input_ids"][len_input_ids:]
            # Если labels короче, чем input_ids, добираем -100 (хотя токенизация full_text гарантирует,
            # что len(labels) == len(input_ids) или labels короче, если произошло усечение).
            if len(labels) < len(input_ids):
                diff = len(input_ids) - len(labels)
                labels = labels + [-100] * diff

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }

    # 6) Применяем map(batched=True) для tokenized_dataset
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # ─── 7) DataCollatorForSeq2Seq ──────────────────────────────────────────────
    # Он автоматически паддит input_ids, attention_mask и labels в батче.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,                     # чтобы знать pad_token и label_pad_token_id
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    # ─── 8) Параметры обучения ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="qwen3-lora-output",
        per_device_train_batch_size=4,       # подберите по вашей A100
        gradient_accumulation_steps=4,       # если VRAM мало, увеличивайте
        learning_rate=3e-4,
        weight_decay=0.0,
        max_steps=2000,                      # либо num_train_epochs
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False,
    )

    # ─── 9) Инициализируем Trainer ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ─── 10) Запускаем обучение ────────────────────────────────────────────────
    trainer.train()

    # ─── 11) Сохраняем LoRA-адаптер ─────────────────────────────────────────────
    # В папке "qwen3-lora-adapter" будут только адаптерные веса (~десятки MB), без всей 8B-модели.
    model.save_pretrained("qwen3-lora-adapter")


if __name__ == "__main__":
    main()
