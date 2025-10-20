.PHONY: dev-up test lint type sec profile release

dev-up:
	docker compose -f lab/docker-compose.yml up -d
	uvicorn orchestrator.app.main:app --reload

lint:
	ruff check .
	black --check .

type:
	mypy --strict orchestrator agents analysis

sec:
	bandit -r orchestrator agents analysis -ll

test:
	pytest -q --maxfail=1 --disable-warnings

profile:
	bash tools/run_profile.sh

release:
	@echo "Run tests, build images, tag + draft release (implement your script here)"
