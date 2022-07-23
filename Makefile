.PHONY: tests
tests:
	PY_MAJOR_VERSION=py`python -c 'import sys; print(sys.version_info[0])'` pytest -v --cov=dnmf --cov-config=.coveragerc tests
	flake8 dnmfx
