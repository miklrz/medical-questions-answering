# Medical Questions Answering 🏥

RAG-система для автоматических ответов на медицинские вопросы с использованием fine-tuned BERT, векторного поиска и LLM. Реализован full-stack pipeline: LangGraph multi-step reasoning, structured output (Pydantic), prompt engineering, A/B тестирование промптов.

## 🎯 Проблема

Поиск ответов на медицинские вопросы в большой базе знаний занимает много времени. Простой семантический поиск часто возвращает нерелевантные результаты.

## 💡 Решение

1. **Retrieval** — FAISS находит топ-5 кандидатов через vector search (sentence-transformers)
2. **Reranking** — Fine-tuned BERT переранжирует по реальной релевантности
3. **LangGraph pipeline** — Query Analysis → Retrieval → Reranking → Answer Generation → Quality Check
4. **Generation** — LLM генерирует ответ с structured output (Pydantic)
5. **Quality Check** — при низкой уверенности добавляются предупреждения и рекомендация визита к врачу


**Datasets**
- [curaihealth/medical_questions_pairs](https://huggingface.co/datasets/curaihealth/medical_questions_pairs)
- [ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)


## ✨ Ключевые улучшения

| Фича | Описание |
|------|----------|
| **Structured Output** | Схема ответа через Pydantic: `answer`, `confidence`, `sources`, `requires_doctor_visit`, `warnings` |
| **LangGraph** | Multi-step reasoning: анализ запроса → retrieval → reranking → generation → validation |
| **Prompt Engineering** | System guidelines + few-shot examples + chain-of-thought |
| **A/B тестирование** | Тестовый набор, ROUGE/BLEU, semantic similarity, human feedback в боте |

## 📊 Результаты (примеры для резюме)

- **A/B тестирование промптов**: базовый vs few-shot vs CoT vs full. Best ROUGE-L ~0.68 (full).
- **Latency**: <2 сек на CPU (с оптимизацией inference).

## 🛠️ Технологии

- **Fine-tuning**: BERT-base
- **Retrieval**: FAISS, sentence-transformers (paraphrase-MiniLM-L3-v2)
- **LLM**: LM Studio (локальные модели)
- **Orchestration**: LangGraph, LangChain
- **Structured Output**: Pydantic
- **Evaluation**: rouge-score, BLEU, semantic similarity
- **MLOps**: Weights & Biases
- **API**: FastAPI
- **Deployment**: Docker, Telegram Bot (Aiogram)

## 🚀 Как работает

1. Пользователь задает вопрос через Telegram
2. **Query Analysis** — анализ запроса, детекция срочности
3. **Retrieval** — FAISS ищет top-5 похожих документов
4. **Reranking** — BERT оценивает релевантность, оставляем top-3
5. **Generation** — LLM генерирует ответ в JSON (structured output)
6. **Quality Check** — при confidence < 0.7 добавляются предупреждения
7. Ответ возвращается с кнопками «Полезно» / «Не полезно» (human feedback)

## 📁 Структура

```
├── src/api/
│   ├── main.py          # FastAPI: /answer, /feedback
│   ├── graph.py         # LangGraph pipeline
│   ├── schemas.py       # Pydantic: MedicalAnswer, Query
│   ├── prompts.py       # System prompt, few-shot, CoT
│   ├── llm_generation.py # Structured output generation
│   ├── retrieval.py     # FAISS + sentence-transformers
│   ├── models.py        # BERT reranker (SimilarityModel)
│   ├── dataset.py       # MedicalDataset, train_test, tokenize
│   ├── train.py         # Обучение BERT reranker
│   ├── evaluation.py    # ROUGE, BLEU, semantic similarity
│   └── feedback_store.py
├── src/bot/main.py      # Telegram bot с кнопками feedback
├── data/test_questions.json  # Тестовый набор для A/B
└── scripts/
    └── ab_test_prompts.py   # A/B тестирование промптов
```

## 🏃 Запуск

```bash
# Установка
poetry install

# Обучение BERT reranker (если ещё не обучен)
# См. src/api/train.py — требует датасет с question_1, question_2 (curaihealth/medical_questions_pairs)

# Запуск API (требуется LM Studio на localhost:1234)
HOST=localhost uvicorn src.api.main:app --reload

# Запуск бота
TELEGRAM_TOKEN=xxx python -m src.bot.main

# A/B тестирование промптов
poetry run python scripts/ab_test_prompts.py
```

## 📈 Эксперименты

- **Промпты**: `base`, `few_shot`, `cot`, `full` — сравниваются по ROUGE-L, BLEU, semantic similarity.

## 🎬 Demo
(В процессе)
[Telegram Bot]() | [Video Demo]()
