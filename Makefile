.PHONY: train run-local docker-up docker-down

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

up:
	docker compose up -d --build

down:
	docker compose down

train: 
	PTHONPATH=. python -m src.api.train

run-local:
	PYHTONPATH=. uvicorn src.api.main:app --reload --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down