# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kernax is a JAX-based kernel library for Gaussian Processes, implementing various covariance functions with automatic differentiation and JIT compilation support. The library is built on Equinox and follows a dual-class architecture pattern that separates static computation methods from instance-based kernel objects.

## Architecture

### Dual-Class Pattern

Each kernel type follows a consistent pattern with two classes:

1. **Static Class** (e.g., `StaticSEKernel`): Inherits from `StaticAbstractKernel`
   - Contains static `@classmethod` implementations decorated with `@filter_jit` (Equinox's JIT)
   - Implements `pairwise_cov(cls, kern, x1, x2)` for scalar-to-scalar covariance computation
   - All computation logic lives here for JIT optimization

2. **Instance Class** (e.g., `SEKernel`): Inherits from `AbstractKernel` (which extends `eqx.Module`)
   - Automatically registered as PyTree through Equinox Module system
   - Holds hyperparameters as instance attributes with `eqx.field` (e.g., `length_scale`)
   - Sets `static_class` as class attribute pointing to corresponding static class
   - Hyperparameters automatically converted to JAX arrays via Equinox converters

### AbstractKernel Base Class

The `AbstractKernel` class (kernax/AbstractKernel.py:10-63) extends `eqx.Module` and provides:

- **Equinox Module integration**: Inherits from `eqx.Module` for automatic PyTree registration and clean gradient computation

- **Automatic dimension handling** via `__call__`: Detects input dimensions and dispatches to appropriate computation method
  - 1D x 1D → `pairwise_cov_if_not_nan` (scalar output)
  - 2D x 1D → `cross_cov_vector_if_not_nan` (vector output)
  - 2D x 2D → `cross_cov_matrix` (matrix output)

- **NaN handling**: `pairwise_cov_if_not_nan` and `cross_cov_vector_if_not_nan` methods check for NaN inputs

- **Vectorization**: Uses `vmap` to efficiently build up from scalar operations to vector/matrix operations

- **Operator overloading**: Supports `+`, `*`, and `-` operators to create composite kernels
  - `kernel1 + kernel2` → `SumKernel(kernel1, kernel2)`
  - `kernel1 * kernel2` → `ProductKernel(kernel1, kernel2)`
  - `-kernel` → `NegKernel(kernel)`

### Kernel Categories

1. **Base Kernels** (implement `pairwise_cov` in static class):
   - SE (Squared Exponential, aka RBF or Gaussian)
   - Linear
   - Matern (1/2, 3/2, 5/2)
   - Periodic
   - Rational Quadratic
   - Constant

2. **Operator Kernels** (kernax/OperatorKernels.py): Combine two kernels
   - `SumKernel`: Adds outputs of two kernels
   - `ProductKernel`: Multiplies outputs of two kernels
   - Both auto-convert non-kernel arguments to `ConstantKernel`

3. **Wrapper Kernels** (kernax/WrapperKernels.py): Transform or modify kernel behavior
   - `ExpKernel`: Applies exponential
   - `LogKernel`: Applies logarithm
   - `NegKernel`: Negates output
   - `DiagKernel`: Returns value only when inputs are equal (creates diagonal matrices)
   - `BatchKernel`: Adds batch handling with distinct hyperparameters per batch element
   - `ActiveDimsKernel`: Selects specific input dimensions before kernel computation
   - `ARDKernel`: Applies Automatic Relevance Determination (different length scale per dimension)
   - Transform wrappers auto-convert non-kernel arguments to `ConstantKernel`

## Development Commands

### Running Python Code
```bash
# Navigate to the kernax directory
cd kernax

# Run Python scripts that import Kernax
python3 script.py
```

### Testing Kernels
```bash
# Import and test a kernel in Python REPL
cd kernax
python3
>>> from Kernax import SEKernel
>>> import jax.numpy as jnp
>>> kernel = SEKernel(length_scale=1.0)
>>> kernel(jnp.array([1.0]), jnp.array([2.0]))
```

## Implementation Guidelines

### Adding a New Kernel

1. Create static class inheriting from `StaticAbstractKernel`
2. Implement `pairwise_cov(cls, kern, x1, x2)` as a `@classmethod` with `@filter_jit` decorator
3. Create instance class inheriting from `AbstractKernel` (which extends `eqx.Module`)
4. Define hyperparameters as class attributes using `eqx.field(converter=jnp.asarray)` for automatic array conversion
5. Set `static_class` as a class attribute pointing to the static class
6. In `__init__`, call `super().__init__()` then set hyperparameter values
7. Add both classes to `__init__.py` imports and `__all__`

Example:
```python
from equinox import filter_jit
import equinox as eqx
from jax import Array
import jax.numpy as jnp
from kernax import StaticAbstractKernel, AbstractKernel

class StaticMyKernel(StaticAbstractKernel):
    @classmethod
    @filter_jit
    def pairwise_cov(cls, kern: AbstractKernel, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        kern = eqx.combine(kern)  # If needed to access hyperparameters
        # Implement kernel computation
        return ...

class MyKernel(AbstractKernel):
    my_param: Array = eqx.field(converter=jnp.asarray)
    static_class = StaticMyKernel

    def __init__(self, my_param):
        super().__init__()
        self.my_param = my_param
```

### Import Patterns

- Within kernax modules: Use `from Kernax import` or relative imports `from .AbstractKernel import`
- Import inconsistency exists: Some files use `from Kernax import`, others use `from .AbstractKernel import` (both work)

### JAX and Equinox Considerations

- All kernel computations must use `jax.numpy` instead of `numpy`
- Use `@filter_jit` from Equinox for JIT compilation (handles PyTrees correctly)
- Hyperparameters are automatically converted to JAX arrays via `eqx.field(converter=jnp.asarray)`
- PyTree registration is automatic through `eqx.Module` inheritance
- Use `eqx.field(static=True)` for non-differentiable parameters (e.g., dimensions, boolean flags)
- Equinox provides clean separation between differentiable and static fields