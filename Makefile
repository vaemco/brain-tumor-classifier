PYTHON ?= python
PROJECT_ROOT := $(CURDIR)
PYTHONPATH := $(PROJECT_ROOT)/src

.PHONY: web train evaluate test lint

web:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m website.app

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train.py

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/evaluate.py

test:
	PYTHONPATH=$(PYTHONPATH) pytest

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall src scripts website
