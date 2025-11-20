# Kernax Package Setup Checklist

## ✅ Configuration Completed

All placeholders have been filled with the following information:
- **Author**: S. Lejoly (simon.lejoly@unamur.be)
- **Repository**: https://github.com/SimLej18/Kernax
- **License**: MIT (2025)
- **Python version**: 3.14
- **JAX version**: >= 0.8.0
- **Environment**: Conda (environment name: Kernax)
- **Release date**: 2025-11-20

## Post-Setup Steps

### 1. Initialize Git Repository (if not already done)
```bash
cd /Users/simonlejoly/Documents/Work/Kernax
git init
git add .
git commit -m "Initial commit: Complete Python package structure"
```

### 2. Create GitHub Repository
1. Create a new repository on GitHub
2. Update all repository URLs in the files mentioned above
3. Push your code:
```bash
git remote add origin <your-repository-url>
git push -u origin main
```

### 3. Set Up Continuous Integration
- The GitHub Actions workflows are already configured in `.github/workflows/`
- They will run automatically once you push to GitHub

### 4. Set Up Code Coverage
- Sign up for [Codecov](https://codecov.io/)
- Add your repository
- The coverage reports will be automatically uploaded by the test workflow

### 5. Optional: Set Up Documentation Hosting
- Sign up for [Read the Docs](https://readthedocs.org/)
- Connect your GitHub repository
- Configure to use the `docs/` directory

### 6. Set Up Development Environment

**Using Conda (recommended)**:
```bash
# Run the setup script
bash scripts/setup_dev.sh

# Or manually:
conda create -n Kernax python=3.14
conda activate Kernax
pip install -e .

# Test import
python -c "import kernax; print(kernax.__version__)"

# Run tests
make test
```

**Using pip**:
```bash
pip install -e .
python -c "import kernax; print(kernax.__version__)"
make test
```

### 7. Before First Release
- [x] Complete all placeholder information ✅
- [x] Update CHANGELOG.md with release date ✅
- [ ] Run full test suite: `make test`
- [ ] Check code style: `make lint`
- [ ] Format code: `make format`
- [ ] Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Push tag: `git push origin v0.1.0`

### 8. Publishing to PyPI (when ready)
```bash
# Install build tools
pip install build twine

# Build package
make build

# Test upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# If successful, upload to PyPI
make upload
```

## Files Created

### Core Package Files
- `pyproject.toml` - Modern Python package configuration
- `setup.py` - Backward compatibility setup script
- `MANIFEST.in` - Package distribution manifest
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies
- `kernax/py.typed` - Type hints marker file

### Documentation
- `README.md` - Main package documentation
- `CLAUDE.md` - Architecture guide for Claude Code
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `LICENSE` - MIT License
- `docs/` - Sphinx documentation structure

### Development Tools
- `.gitignore` - Git ignore patterns
- `.editorconfig` - Editor configuration
- `.python-version` - Python version for pyenv
- `Makefile` - Common development tasks
- `.github/workflows/` - CI/CD workflows

### Testing
- `tests/` - Complete test suite
  - `conftest.py` - Pytest configuration
  - `test_base_kernels.py` - Base kernel tests
  - `test_composite_kernels.py` - Composite kernel tests
  - `test_batched_operations.py` - Advanced feature tests

## Quick Reference Commands

### Environment Management (Conda)
```bash
conda activate Kernax     # Activate environment
conda deactivate          # Deactivate environment
bash scripts/setup_dev.sh # Run setup script
```

### Development Commands
```bash
make install-dev          # Install with dev dependencies
make test                 # Run tests
make test-cov            # Run tests with coverage
make lint                # Check code quality
make format              # Format code
```

### Building
```bash
make build               # Build distribution
make clean               # Clean build artifacts
```

### Documentation
```bash
make docs                # Build documentation (when Sphinx is configured)
```

## Notes

- The package is currently in **Alpha** stage (version 0.1.0)
- All core functionality is implemented and tested
- Documentation structure is in place but needs expansion
- Package is ready for local development and testing
- Not yet published to PyPI

## Support

For questions or issues during setup, refer to:
- `CONTRIBUTING.md` - Contribution guidelines
- `CLAUDE.md` - Architecture documentation
- GitHub Issues (once repository is created)