# =============================================================================
# Stage 1: builder — устанавливаем зависимости
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Версия poetry должна совпадать с локальной — проверь: poetry --version
RUN pip install --no-cache-dir "poetry==2.1.0"

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

# Копируем только манифест зависимостей — этот слой кешируется Docker'ом.
# При изменении только кода src/ этот слой не пересобирается → быстрая сборка.
COPY pyproject.toml poetry.lock ./

# Ставим только main (python-dotenv) + группу api (torch, fastapi, и т.д.)
# --no-root: не устанавливаем сам пакет, код копируем отдельно ниже
RUN poetry install --only main,api --no-root \
    && rm -rf /tmp/poetry-cache

# =============================================================================
# Stage 2: runtime — чистый образ без poetry и build-мусора
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Переносим установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код
COPY src/ ./src/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]