.PHONY: help install install-dev test test-cov lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install package with development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run linters (flake8, mypy)"
	@echo "  make format        - Format code (black, isort)"
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
	pytest --cov=kernax --cov-report=html --cov-report=term

lint:
	flake8 kernax tests
	mypy kernax --ignore-missing-imports

format:
	black kernax tests
	isort kernax tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

docs:
	cd docs && make html