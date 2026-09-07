from pathlib import Path


def test_app_containers_use_the_postgres_service_hostname() -> None:
    compose_file = Path(__file__).parents[1] / "docker-compose.yml"

    assert "POSTGRES_HOST: postgres" in compose_file.read_text(encoding="utf-8")
