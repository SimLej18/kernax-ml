# Kernax Scripts

This directory contains utility scripts for Kernax development.

## Available Scripts

### `setup_dev.sh`
**Purpose**: Automated development environment setup using Conda

**Usage**:
```bash
bash scripts/setup_dev.sh
```

**What it does**:
1. Checks if Conda is installed
2. Creates a Conda environment named "Kernax" with Python 3.14
3. Installs all dependencies (core + development)
4. Installs Kernax in editable mode
5. Verifies the installation

**Features**:
- Checks for existing environment and asks before overwriting
- Provides helpful feedback during setup
- Exits gracefully on errors
- Shows next steps after completion

**Requirements**:
- Conda (Anaconda or Miniconda) must be installed
- Bash shell

### `verify_setup.py`
**Purpose**: Verify that all package files are correctly set up

**Usage**:
```bash
python scripts/verify_setup.py
```

**What it checks**:
- Core package files exist (pyproject.toml, setup.py, etc.)
- Documentation files exist (README.md, CONTRIBUTING.md, etc.)
- Kernax package files exist (all kernel implementations)
- Test suite is complete
- Development tools are configured
- Placeholders have been filled
- Package can be imported successfully

**Output**:
- ✅ Green checkmarks for passed checks
- ⚠️  Yellow warnings for placeholders that need filling
- ❌ Red X for failed checks
- Summary with next steps

**Exit codes**:
- 0: All checks passed
- 1: Some checks failed or placeholders remain

## Common Workflows

### Initial Setup
```bash
# Clone repository
git clone https://github.com/SimLej18/kernax-ml
cd kernax-ml

# Run setup script
bash scripts/setup_dev.sh

# Verify everything is working
python scripts/verify_setup.py
```

### Updating Environment
```bash
# Activate environment
conda activate kernax-ml

# Update dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Reinstall package
pip install -e .
```

### Troubleshooting

#### Conda not found
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Or use Homebrew
brew install miniconda
```

#### Environment already exists
The setup script will prompt you to remove and recreate or keep the existing environment.

#### Import errors
```bash
# Make sure environment is activated
conda activate kernax-ml

# Reinstall in editable mode
pip install -e .

# Check installation
python -c "import kernax; print(kernax.__version__)"
```

## Script Maintenance

When modifying scripts:
1. Test thoroughly before committing
2. Update this README with any changes
3. Keep scripts POSIX-compliant where possible
4. Add error handling for common failure cases
5. Provide helpful error messages

## Future Scripts

Potential additions:
- `run_benchmarks.sh` - Run performance benchmarks
- `build_docs.sh` - Build documentation locally
- `release.sh` - Automate release process
- `test_environments.sh` - Test across multiple Python versions