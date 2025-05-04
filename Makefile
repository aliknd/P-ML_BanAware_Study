.PHONY: all deps global_supervised global_ssl personal_ssl

all: deps global_supervised global_ssl personal_ssl

deps:
	@echo "→ Installing Python dependencies…"
	pip install -r requirements.txt

global_supervised:
	@echo "→ Running global_supervised_pipeline.py"
	python3 -m src.global_supervised_pipeline

global_ssl:
	@echo "→ Running global_ssl_pipeline.py"
	python3 -m src.global_ssl_pipeline

personal_ssl:
	@echo "→ Running personal_ssl_pipeline.py"
	python3 -m src.personal_ssl_pipeline