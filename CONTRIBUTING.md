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

We use the following tools to maintain code quality:

- **ruff** for code formatting and linting
- **mypy** for type checking

Format code:
```bash
make format
```

Check code quality:
```bash
make lint
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

1. Create a new file in the appropriate `kernax/` subdirectory (e.g., `kernax/stationary/MyKernel.py`)

2. Implement the kernel class:
```python
from __future__ import annotations
import equinox as eqx
from equinox import filter_jit
from jax import Array
import jax.numpy as jnp
from .StationaryKernel import AbstractStationaryKernel  # or AbstractKernel
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class MyKernel(AbstractStationaryKernel):
    engine: AbstractEngine = eqx.field(static=True)
    _my_param_parametrisation: AbstractParametrisation = eqx.field()
    _my_param: Array = eqx.field(converter=jnp.asarray)  # param that needs parametrisation
    other_param: Array = eqx.field(converter=jnp.asarray)  # param that doesn't need parametrisation

    @property
    def my_param(self) -> Array:
        return self._my_param_parametrisation.unwrap(self._my_param)

    def __init__(self, my_param: float | Array,
                 other_param: float | Array,
                 my_param_parametrisation: AbstractParametrisation = LogExpParametrisation(),
                 engine: AbstractEngine = DenseEngine):
        my_param = jnp.asarray(my_param)
        self._my_param_parametrisation = my_param_parametrisation
        self._my_param = self._my_param_parametrisation.wrap(my_param)
        self.other_param = other_param  # We can assign/access it directly, as there is no parametrisation
        self.engine = engine

    @filter_jit
    def pairwise(self, x1: Array, x2: Array) -> Array:
        return ...  # implement computation

    def replace(self, 
                my_param: None | float | Array = None, 
                other_param: None | float | Array = None,
                **kwargs) -> MyKernel:
        new_kernel = self
        
        if my_param is not None:
            new_kernel =  eqx.tree_at(
                lambda k: k._my_param,
                new_kernel,
                new_kernel._my_param_parametrisation.wrap(jnp.asarray(my_param)))
        if other_param is not None:
            new_kernel = eqx.tree_at(
                lambda k: k.other_param,  # Assign directly
                new_kernel,
                jnp.asarray(other_param)  # No parametrisation
            )
            
        return new_kernel
```

3. Export from the subdirectory's `__init__.py` and from `kernax/__init__.py`

4. Write tests in the appropriate test files

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