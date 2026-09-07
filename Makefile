.PHONY: help setup build up services down restart ingest ingest-container evaluate app test lint format-check typecheck connect-db

help:
	@echo "BizCompass KH commands"
	@echo ""
	@echo "  setup             Install locked development dependencies with uv"
	@echo "  build             Build the Compose application image"
	@echo "  up                Start the full Compose stack"
	@echo "  services          Start PostgreSQL and Grafana only"
	@echo "  down              Stop the Compose stack without deleting volumes"
	@echo "  restart           Restart the Compose stack"
	@echo "  ingest            Ingest sources from the local Python environment"
	@echo "  ingest-container  Ingest sources in a one-off Compose container"
	@echo "  evaluate          Run retrieval and answer evaluation"
	@echo "  app               Run Streamlit locally on port 8501"
	@echo "  test              Run the test suite"
	@echo "  lint              Run Ruff checks"
	@echo "  format-check      Check Ruff formatting"
	@echo "  typecheck         Run Pyright"
	@echo "  connect-db        Open psql in the PostgreSQL container"

setup:
	uv sync --dev

build:
	docker compose build

up:
	docker compose up -d

services:
	docker compose up -d postgres grafana

down:
	docker compose down

restart:
	docker compose restart

ingest:
	uv run bizcompass-ingest

ingest-container:
	docker compose run --rm streamlit uv run bizcompass-ingest

evaluate:
	uv run bizcompass-evaluate

app:
	uv run streamlit run src/bizcompass_kh/ui/app.py

test:
	uv run pytest

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

connect-db:
	docker compose exec postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB
