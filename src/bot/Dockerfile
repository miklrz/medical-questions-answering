# =============================================================================
# Stage 1: builder
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir "uv==0.11.16"

ENV UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

# Ставим только main (python-dotenv) + bot (aiogram, httpx)
# torch, faiss и весь ML стек сюда НЕ попадают → образ ~200MB вместо ~8GB
RUN uv sync --locked --no-dev --group bot --no-install-project

# =============================================================================
# Stage 2: runtime
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/bot/ ./src/bot/

ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "src.bot.main"]