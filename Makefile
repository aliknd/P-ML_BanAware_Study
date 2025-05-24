.PHONY: all deps compare

all: deps compare

deps:
	@echo "→ Installing Python dependencies…"
	pip install -r requirements.txt

compare:
	@echo "→ Running compare_pipelines for all users/scenarios…"
	python3 -m src.compare_pipelines --user ID5  --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID9  --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID10 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID10 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID10 --fruit Carrot    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID10 --fruit Nectarine --scenario Crave && \
	python3 -m src.compare_pipelines --user ID11 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID11 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID11 --fruit Almond    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID11 --fruit Carrot    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID11 --fruit Nectarine --scenario Crave && \
	python3 -m src.compare_pipelines --user ID11 --fruit Almond    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID12 --fruit Melon     --scenario Use   && \
	python3 -m src.compare_pipelines --user ID12 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID12 --fruit GHB       --scenario Use   && \
	python3 -m src.compare_pipelines --user ID12 --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID12 --fruit Nectarine --scenario Crave && \
	python3 -m src.compare_pipelines --user ID13 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID13 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID13 --fruit Almond    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID14 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID14 --fruit Carrot    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID15 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID15 --fruit Carrot    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID18 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID18 --fruit Carrot    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID19 --fruit Melon     --scenario Use   && \
	python3 -m src.compare_pipelines --user ID19 --fruit Almond    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID19 --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID19 --fruit Almond    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID20 --fruit Melon     --scenario Use   && \
	python3 -m src.compare_pipelines --user ID20 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID20 --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID20 --fruit Nectarine --scenario Crave && \
	python3 -m src.compare_pipelines --user ID21 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID21 --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID21 --fruit Nectarine --scenario Crave && \
	python3 -m src.compare_pipelines --user ID25 --fruit Almond    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID25 --fruit Almond    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID25 --fruit Carrot    --scenario Crave && \
	python3 -m src.compare_pipelines --user ID26 --fruit Carrot    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID27 --fruit Melon     --scenario Use   && \
	python3 -m src.compare_pipelines --user ID27 --fruit Nectarine --scenario Use   && \
	python3 -m src.compare_pipelines --user ID27 --fruit Melon     --scenario Crave && \
	python3 -m src.compare_pipelines --user ID27 --fruit Nectarine --scenario Crave && \
	python3 -m src.compare_pipelines --user ID28 --fruit Coffee    --scenario Use   && \
	python3 -m src.compare_pipelines --user ID28 --fruit Almond    --scenario Use
