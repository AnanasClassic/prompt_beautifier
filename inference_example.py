import os
import argparse
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model_for_inference(
    base_model_name: str,
    lora_adapter_path: str,
    device: str = "cuda",
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch_dtype,
    )

    model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def generate_improved_prompt(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    user_prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    eos_token = tokenizer.eos_token
    eos_id    = tokenizer.eos_token_id

    prefix = f"Улучши промт: <prompt>{user_prompt}</prompt>"
    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
    )[0]

    decoded = tokenizer.decode(output_ids, skip_special_tokens=False)
    generated = decoded[len(prefix):] if decoded.startswith(prefix) else decoded.split(eos_token, 1)[-1]

    
    improved_prompt = generated.rstrip().removesuffix(eos_token).strip()
    return improved_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference for Qwen3+LoRA: улучшение промта"
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        help="Базовая модель (HF repo)",
    )
    parser.add_argument(
        "--adapter-path",
        default="qwen3-lora-adapter-stage2",
        help="Папка с LoRA-адаптером",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Исходный промт, который нужно улучшить",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Устройство "cuda" | "cpu"',
    )
    args = parser.parse_args()

    tokenizer, model = load_model_for_inference(
        args.model_name, args.adapter_path, args.device
    )

    improved = generate_improved_prompt(
        tokenizer, model,
        user_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print("\n=== Исходный промт ===")
    print(args.prompt)
    print("\n=== Улучшенный промт ===")
    print(improved)


if __name__ == "__main__":
    main()
