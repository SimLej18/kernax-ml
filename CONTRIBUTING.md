# Contributing to Kernax

Thank you for your interest in contributing to Kernax! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/SimLej18/kernax-ml
cd kernax-ml
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Install Kernax in editable mode:
```bash
pip install -e .
```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all formatters:
```bash
black kernax tests
isort kernax tests
```

Check code quality:
```bash
flake8 kernax tests
mypy kernax
```

### Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=kernax --cov-report=html
```

### Adding a New Kernel

To add a new kernel type:

1. Create a new file in the `kernax/` directory (e.g., `NewKernel.py`)

2. Implement the static class:
```python
from functools import partial
from jax import jit
import jax.numpy as jnp
from kernax import StaticAbstractKernel

class StaticNewKernel(StaticAbstractKernel):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        # Implement your kernel computation here
        pass
```

3. Implement the instance class:
```python
from jax.tree_util import register_pytree_node_class
from kernax import AbstractKernel

@register_pytree_node_class
class NewKernel(AbstractKernel):
    def __init__(self, hyperparam1=None, hyperparam2=None):
        super().__init__(hyperparam1=hyperparam1, hyperparam2=hyperparam2)
        self.static_class = StaticNewKernel
```

4. Add imports to `kernax/__init__.py`

5. Write tests in `tests/test_base_kernels.py` or a new test file

6. Update documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture guidelines.

## Pull Request Process

1. Fork the repository and create a new branch for your feature or bug fix

2. Make your changes, ensuring all tests pass and code style is maintained

3. Add tests for new functionality

4. Update documentation as needed

5. Submit a pull request with a clear description of changes

## Code Review

All submissions require review. We use GitHub pull requests for this purpose. Please:

- Write clear commit messages
- Keep pull requests focused on a single feature or fix
- Respond to feedback promptly
- Be patient and respectful

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (Python version, JAX version, OS)
- Minimal code example if applicable

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.