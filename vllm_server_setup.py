import argparse
from typing import Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from fastapi.middleware.cors import CORSMiddleware


def load_llm(model_name: str,
             lora_path: str,
             device: str) -> Tuple[LLM, AutoTokenizer, LoRARequest]:
    
    if device.startswith("cuda"):
        if ":" in device:
            device_idx = int(device.split(":")[1])
        else:
            device_idx = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(0.5, device_idx)

    dtype = "float16" if device.startswith("cuda") else "float32"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=dtype,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32
    )

    lora_request = LoRARequest(
        lora_name="qwen3_stage2",
        lora_int_id=1,
        lora_path=lora_path
    )

    return llm, tokenizer, lora_request


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9


def create_app(llm: LLM,
               tokenizer: AutoTokenizer,
               lora_request: LoRARequest) -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/improve")
    async def improve(req: PromptRequest):
        prefix = f"Улучши промgт: <prompt>{req.prompt}</prompt>"
        sampling = SamplingParams(
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_token_ids=[tokenizer.eos_token_id],
        )

        outputs = llm.generate(
            prompts=[prefix],
            sampling_params=sampling,
            lora_request=lora_request
        )
        text = outputs[0].outputs[0].text.lstrip()

        if text.startswith(prefix):
            text = text[len(prefix):]
        text = text.split(tokenizer.eos_token, 1)[0].strip()
        return {"improved_prompt": text}

    return app

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Qwen3 + LoRA via vLLM with GPU memory limit"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-path",
                        default="qwen3-lora-adapter")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    llm, tokenizer, lora_request = load_llm(
        args.model_name, args.adapter_path, args.device
    )
    app = create_app(llm, tokenizer, lora_request)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
