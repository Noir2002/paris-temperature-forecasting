PY ?= .venv/bin/python

.PHONY: all test data venv

all:            ## run every experiment, write results/ (~2 min, offline)
	$(PY) -m src.run_all

test:           ## reproduction + no-look-ahead tests
	$(PY) -m pytest -q

data:           ## re-download Météo-France files and re-check stage-1 facts
	$(PY) -m src.data && $(PY) -m src.verify_data

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
