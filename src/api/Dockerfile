# =============================================================================
# Stage 1: builder — устанавливаем зависимости
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv==0.11.16"

ENV UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1

# Копируем только манифест зависимостей — этот слой кешируется Docker'ом.
# При изменении только кода src/ этот слой не пересобирается → быстрая сборка.
COPY pyproject.toml uv.lock ./

# Ставим только main (python-dotenv) + группу api (torch, fastapi, и т.д.)
# --no-install-project: не устанавливаем сам пакет, код копируем отдельно ниже
RUN uv sync --locked --no-dev --group api --no-install-project

# =============================================================================
# Stage 2: runtime — чистый образ без uv и build-мусора
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Переносим виртуальное окружение из builder
COPY --from=builder /app/.venv /app/.venv

# Копируем код
COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]