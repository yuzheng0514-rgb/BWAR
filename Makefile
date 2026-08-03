PYTHON ?= python

.PHONY: install test smoke s1 s2 simulations divvy pems figures

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py \
		--reps 1 --output-root results/generated/smoke/s1_geometry \
		--artifact-root artifacts/generated/smoke
	PYTHONPATH=src $(PYTHON) scripts/run_matched_reference_drift.py \
		--reps 2 --workers 1 --tag smoke_s2

s1:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py --reps 80 --ar-model full

s2:
	PYTHONPATH=src $(PYTHON) scripts/run_matched_reference_drift.py \
		--workers 1 --tag matched_start_full

simulations: s1 s2

divvy:
	PYTHONPATH=src $(PYTHON) scripts/run_divvy.py

pems:
	PYTHONPATH=src $(PYTHON) scripts/run_pems_bay.py \
		--out results/generated/pems_bay

figures:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py
