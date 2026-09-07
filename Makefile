PYTHON ?= venv/bin/python

.PHONY: audit evaluate test notebook docx latex slides serve verify

audit:
	$(PYTHON) scripts/audit_data.py --output outputs/data_audit.json

evaluate:
	$(PYTHON) evaluate.py --split test

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -v

notebook:
	$(PYTHON) scripts/build_notebook.py
	$(PYTHON) scripts/execute_notebook.py

docx:
	$(PYTHON) docs/presentation/build_docx.py

latex:
	cd latex && tectonic --keep-logs main.tex

slides:
	$(PYTHON) docs/presentation/build_pptx.py

serve:
	$(PYTHON) api.py

verify: audit test notebook latex
