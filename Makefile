.PHONY: help build up down restart down-volume connect-db

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help        Show this help message"
	@echo "  build       Build the Docker containers"
	@echo "  up          Start the Docker containers"
	@echo "  down        Stop the Docker containers"
	@echo "  restart     Restart the Docker containers"
	@echo "  down-volume Remove the Docker volume"
	@echo "  connect-db  Connect to the database"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

down-volume:
	docker-compose down -v

connect-db:
	uvx pgcli -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB}
