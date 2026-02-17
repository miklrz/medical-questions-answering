# =============================================================================
# Stage 1: builder
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir "poetry==2.1.0"

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

COPY pyproject.toml poetry.lock ./

# Ставим только main (python-dotenv) + bot (aiogram, httpx)
# torch, faiss и весь ML стек сюда НЕ попадают → образ ~200MB вместо ~8GB
RUN poetry install --only main,bot --no-root \
    && rm -rf /tmp/poetry-cache

# =============================================================================
# Stage 2: runtime
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/bot/ ./src/bot/

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.bot.main"]