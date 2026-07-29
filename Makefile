PYTHON ?= python3

.PHONY: install smoke simulations divvy figures data-figure

install:
	$(PYTHON) -m pip install -e .

smoke:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py \
		--fixed-reps 1 --drift-reps 1 \
		--output-root results/generated/smoke \
		--artifact-root artifacts/generated/smoke

simulations:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py

divvy:
	PYTHONPATH=src $(PYTHON) scripts/run_divvy.py

figures:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py

data-figure:
	PYTHONPATH=src $(PYTHON) scripts/build_divvy_background_figure.py
