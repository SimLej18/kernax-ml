.PHONY: help install install-dev test test-cov test-allure benchmarks benchmarks-compare lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install package with development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make test-allure   - Run tests and generate an allure one-file HTML report"
	@echo "  make benchmarks         - Run performance benchmarks"
	@echo "  make benchmarks-compare - Run cross-library comparison benchmarks"
	@echo "  make lint          - Run linters (ruff, mypy)"
	@echo "  make format        - Format code with tabs (ruff)"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make build         - Build distribution packages"
	@echo "  make upload        - Upload package to PyPI (requires credentials)"
	@echo "  make docs          - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest

test-cov:
	mkdir -p tests/out
	pytest --cov=kernax --cov-report=html:tests/out/htmlcov --cov-report=term

test-allure:
	mkdir -p tests/out/allure-results
	pytest --alluredir=tests/out/allure-results
	allure awesome tests/out/allure-results --single-file --output=tests/out/allure-report

benchmarks:
	mkdir -p benchmarks/out
	pytest benchmarks/base_kernels --benchmark-only -v --benchmark-time-unit=ms --benchmark-autosave --benchmark-compare --benchmark-group-by=func --benchmark-sort=name

benchmarks-compare:
	mkdir -p benchmarks/out
	pytest benchmarks/comparison/ --benchmark-only -v --benchmark-time-unit=ms --benchmark-autosave --benchmark-group-by=fullfunc

lint:
	ruff check kernax tests
	mypy kernax --ignore-missing-imports

format:
	ruff format kernax tests
	ruff check --fix kernax tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf tests/out/
	rm -rf benchmarks/out/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

docs:
	cd docs && make html
