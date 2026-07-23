.PHONY: install train run-local docker-up docker-down

install:
	uv sync --group api --group bot --group dev

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

up:
	docker compose up -d --build

down:
	docker compose down

train: 
	PYTHONPATH=. uv run python -m src.api.train

run-local:
	PYTHONPATH=. uv run uvicorn src.api.main:app --reload --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down
