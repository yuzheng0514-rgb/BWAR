PYTHON ?= python3

.PHONY: install test smoke simulations divvy artifacts verify all

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

smoke:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py \
		--fixed-reps 1 --drift-reps 1 \
		--output-root results/generated/smoke \
		--artifact-root artifacts/generated/smoke

simulations:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py

divvy:
	PYTHONPATH=src $(PYTHON) scripts/run_divvy.py

artifacts:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py

verify: artifacts
	PYTHONPATH=src $(PYTHON) scripts/verify_repository.py

all: simulations divvy artifacts verify
