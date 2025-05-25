SHELL := /bin/bash

.PHONY: all deps compare_original compare_undersample compare_oversample

# list of user:fruit:scenario specs
SCENARIOS := \
	ID5:Melon:Crave \
	ID9:Melon:Crave \
	ID10:Nectarine:Use \
	ID10:Carrot:Use \
	ID10:Carrot:Crave \
	ID10:Nectarine:Crave \
	ID11:Carrot:Use \
	ID11:Nectarine:Use \
	ID11:Almond:Use \
	ID11:Carrot:Crave \
	ID11:Nectarine:Crave \
	ID11:Almond:Crave \
	ID12:Melon:Use \
	ID12:Nectarine:Use \
	ID12:GHB:Use \
	ID12:Melon:Crave \
	ID12:Nectarine:Crave \
	ID13:Nectarine:Use \
	ID13:Carrot:Use \
	ID13:Almond:Use \
	ID14:Carrot:Use \
	ID14:Carrot:Crave \
	ID15:Carrot:Use \
	ID15:Carrot:Crave \
	ID18:Carrot:Use \
	ID18:Carrot:Crave \
	ID19:Melon:Use \
	ID19:Almond:Use \
	ID19:Melon:Crave \
	ID19:Almond:Crave \
	ID20:Melon:Use \
	ID20:Nectarine:Use \
	ID20:Melon:Crave \
	ID20:Nectarine:Crave \
	ID21:Nectarine:Use \
	ID21:Melon:Crave \
	ID21:Nectarine:Crave \
	ID25:Almond:Use \
	ID25:Almond:Crave \
	ID25:Carrot:Crave \
	ID26:Carrot:Use \
	ID27:Melon:Use \
	ID27:Nectarine:Use \
	ID27:Melon:Crave \
	ID27:Nectarine:Crave \
	ID28:Coffee:Use \
	ID28:Almond:Use

all: deps compare_original compare_undersample compare_oversample

deps:
	@echo "→ Installing Python dependencies…"
	pip install -r requirements.txt

compare_original:
	@echo "→ Running ORIGINAL sampling…"
	@for spec in $(SCENARIOS); do \
	  IFS=':' read -r user fruit scenario <<< $$spec; \
	  python3 -m src.compare_pipelines \
	    --user $$user \
	    --fruit $$fruit \
	    --scenario $$scenario \
	    --output-dir original \
	    --results-subdir results \
	    --sample-mode original; \
	done

compare_undersample:
	@echo "→ Running UNDERSAMPLE sampling…"
	@for spec in $(SCENARIOS); do \
	  IFS=':' read -r user fruit scenario <<< $$spec; \
	  python3 -m src.compare_pipelines \
	    --user $$user \
	    --fruit $$fruit \
	    --scenario $$scenario \
	    --output-dir undersample \
	    --results-subdir results \
	    --sample-mode undersample; \
	done

compare_oversample:
	@echo "→ Running OVERSAMPLE sampling…"
	@for spec in $(SCENARIOS); do \
	  IFS=':' read -r user fruit scenario <<< $$spec; \
	  python3 -m src.compare_pipelines \
	    --user $$user \
	    --fruit $$fruit \
	    --scenario $$scenario \
	    --output-dir oversample \
	    --results-subdir results \
	    --sample-mode oversample; \
	done
