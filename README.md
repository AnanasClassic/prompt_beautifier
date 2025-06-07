# prompt_beautifier

## Тренировка LoRa над Qwen3-1.7B для улучшения промптов пользователя
## Проект для курса "Инфраструктура ML"

### Попробовать можно [тут](https://ananasclassic.github.io/prompt_beautifier/)

### Настройка окружения через conda:

```bash
conda deactivate
conda create -n prompt_beautifier python=3.10
conda activate prompt_beautifier
pip install --upgrade pip
pip install -r requirements.txt
```

### Воспроизведение тренировки:

```bash
python train.py --model-name Qwen/Qwen3-1.7B --data-path data/prompt_dataset_chain_0-8000.json
```

### Инференс через vllm-сервер

```bash
python vllm_server_setup.py --adapter-path qwen3-lora-adapter --port 8000
```
Чтобы получить улучшенный промпт отправьте POST-запрос на  `http://localhost:8000/improve` с JSON-ом вида `{"prompt": "..."}`.
