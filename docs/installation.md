# Installation

## Requirements

- Python >= 3.14
- JAX >= 0.8.0
- JAXlib >= 0.8.0

## From Source (Current)

Since Kernax is not yet on PyPI, you need to install from source:

```bash
git clone https://github.com/SimLej18/Kernax
cd Kernax
pip install -e .
```

## Development Installation

To install with development dependencies:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Or use the Makefile:

```bash
make install-dev
```

## Verification

Verify the installation:

```python
import kernax
print(kernax.__version__)
```

You should see the version number printed without errors.