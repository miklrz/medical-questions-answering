# Medical Questions Answering

RAG-система для автоматических ответов на медицинские вопросы с использованием fine-tuned BERT, векторного поиска и LLM. Реализован full-stack pipeline: LangGraph multi-step reasoning, structured output (Pydantic), prompt engineering, A/B тестирование промптов.

## Проблема

Поиск ответов на медицинские вопросы в большой базе знаний занимает много времени. Простой семантический поиск часто возвращает нерелевантные результаты.

## Решение

1. **Retrieval** — FAISS находит топ-5 кандидатов через vector search (sentence-transformers)
2. **Reranking** — Fine-tuned BERT переранжирует по реальной релевантности
3. **LangGraph pipeline** — Query Analysis → Retrieval → Reranking → Answer Generation → Quality Check
4. **Generation** — LLM генерирует ответ с structured output (Pydantic)
5. **Quality Check** — при низкой уверенности добавляются предупреждения и рекомендация визита к врачу

## Технологии

- **Fine-tuning**: BERT-base
- **Retrieval**: FAISS, sentence-transformers (paraphrase-MiniLM-L3-v2)
- **LLM**: LM Studio (локальные модели)
- **Orchestration**: LangGraph, LangChain
- **Structured Output**: Pydantic
- **Evaluation**: rouge-score, BLEU, semantic similarity
- **MLOps**: Weights & Biases
- **API**: FastAPI
- **Deployment**: Docker, Telegram Bot (Aiogram)

## Как работает

1. Пользователь задает вопрос через Telegram
2. **Query Analysis** — анализ запроса, детекция срочности
3. **Retrieval** — FAISS ищет top-5 похожих документов
4. **Reranking** — BERT оценивает релевантность, оставляем top-3
5. **Generation** — LLM генерирует ответ в JSON (structured output)
6. **Quality Check** — при confidence < 0.7 добавляются предупреждения
7. Ответ возвращается с кнопками «Полезно» / «Не полезно» (human feedback)

## Project structure

```
├── src/api/
│   ├── main.py          # FastAPI: /answer, /feedback
│   ├── graph.py         # LangGraph pipeline
│   ├── schemas.py       # Pydantic: MedicalAnswer, Query
│   ├── prompts.py       # System prompt, few-shot, CoT
│   ├── llm_generation.py # Structured output generation
│   ├── retrieval.py     # FAISS + sentence-transformers
│   ├── models.py        # BERT reranker (SimilarityModel)
│   ├── dataset.py       # MedicalDataset, tokenize
│   ├── train.py         # Обучение BERT reranker
│   ├── evaluation.py    # ROUGE, BLEU, semantic similarity
│   └── feedback_store.py
├── src/bot/main.py      # Telegram bot с кнопками feedback
├── data/test_questions.json  # Тестовый набор для A/B
└── scripts/
    └── ab_test_prompts.py   # A/B тестирование промптов
```

## Запуск

```bash
# Установка
uv sync --group api --group bot --group dev

# Конфиги приложения
cp config/api.example.yaml config/api.yaml
cp config/bot.example.yaml config/bot.yaml

# Секреты лучше хранить в .env или окружении
TELEGRAM_TOKEN=xxx

# Обучение BERT reranker (если ещё не обучен)
# См. src/api/train.py — требует датасет с question_1, question_2 (curaihealth/medical_questions_pairs)
uv run python -m src.api.train

# Запуск API
uv run uvicorn src.api.main:app --reload

# Запуск бота
uv run python -m src.bot.main

# A/B тестирование промптов
uv run python scripts/ab_test_prompts.py
```

### Пример curl 
```
curl -X POST http://localhost:8000/answer \
    -H "Content-Type: application/json" \
    -d '{"question":"Что делать при кашле без температуры?"}'
```

## Датасеты

Проект использует два разных датасета с Hugging Face. Оба скачиваются автоматически через `datasets.load_dataset` при первом запуске, если их еще нет в локальном кеше.

1. `ruslanmv/ai-medical-chatbot` — основной QA-корпус для RAG.
    
    [ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)

   Он используется при старте API в `MedicalDataset`: из полей `Description` и `Doctor` собираются строки вида `Q: ... A: ...`, затем по ним строится FAISS-индекс. В конфиге это задается так:

   ```yaml
   dataset:
     qa_name: ruslanmv/ai-medical-chatbot
     qa_split: train
     qa_cache_dir: data/ds_qa
   ```

   Если `qa_cache_dir` указан, Hugging Face положит кеш датасета туда. Если не указан, будет использован стандартный кеш `~/.cache/huggingface/datasets`.

2. `curaihealth/medical_questions_pairs` — пары медицинских вопросов для обучения BERT-реранкера.

    [curaihealth/medical_questions_pairs](https://huggingface.co/datasets/curaihealth/medical_questions_pairs)

   Он используется в `src/api/train.py` через `RerankerDataset`. Ожидаемые поля: `question_1`, `question_2`, `label`. В конфиге это задается так:

   ```yaml
   dataset:
     pairs_name: curaihealth/medical_questions_pairs
     pairs_cache_dir: data/ds_pairs
   ```

   Этот датасет нужен только для обучения реранкера командой `uv run python -m src.api.train`. Для обычного запуска API уже обученная модель должна лежать в `reranker.saved_model_path` / `training.output_dir`, например `data/saved_model`.

## Конфигурация

API и бот используют отдельные YAML-конфиги:

- `config/api.yaml` — LLM endpoint/model/timeout, датасеты, retrieval/reranker, кеш FAISS-индекса, параметры графа, TrainingArguments для обучения и evaluation.
- `config/bot.yaml` — URL API, таймауты HTTP-клиента и Telegram parse mode.

Примеры лежат в `config/api.example.yaml` и `config/bot.example.yaml`. Локальные `config/api.yaml` и `config/bot.yaml` добавлены в `.gitignore`. Значения из `.env` и переменных окружения перекрывают YAML для совместимости с Docker и секретами: `TELEGRAM_TOKEN`, `API_URL`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `SAVED_MODEL_PATH`, `SAVED_DATASET_QA_PATH`, `SAVED_DATASET_PAIRS_PATH`.

### vllm setup
1. ```uv tool install vllm```
2. 
```
  vllm serve HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.65 \
    --max-num-seqs 1 \
    --api-key local-token
```
3. В yaml config:
```
  llm:
    base_url: http://127.0.0.1:8001/v1
    model: HuggingFaceTB/SmolLM2-1.7B-Instruct
    api_key: local-token
    temperature: 0.0
    timeout_seconds: 60.0
```

## Эксперименты

- **Промпты**: `base`, `few_shot`, `cot`, `full` — сравниваются по ROUGE-L, BLEU, semantic similarity.

## Demo
(В процессе)
[Telegram Bot]() | [Video Demo]()
