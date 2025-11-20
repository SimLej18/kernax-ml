# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kernax is a JAX-based kernel library for Gaussian Processes, implementing various covariance functions with automatic differentiation and JIT compilation support. The library is designed around a dual-class architecture pattern that separates static computation methods from instance-based kernel objects.

## Architecture

### Dual-Class Pattern

Each kernel type follows a consistent pattern with two classes:

1. **Static Class** (e.g., `StaticRBFKernel`): Inherits from `StaticAbstractKernel`
   - Contains static `@classmethod` implementations decorated with `@partial(jit, static_argnums=(0,))`
   - Implements `pairwise_cov(cls, kern, x1, x2)` for scalar-to-scalar covariance computation
   - All computation logic lives here for JIT optimization

2. **Instance Class** (e.g., `RBFKernel`): Inherits from `AbstractKernel`
   - Decorated with `@register_pytree_node_class` for JAX PyTree compatibility
   - Holds hyperparameters as instance attributes (e.g., `length_scale`, `variance`)
   - Sets `self.static_class` to point to corresponding static class
   - Hyperparameters are passed to static methods at runtime

### AbstractKernel Base Class

The `AbstractKernel` class (kernax/AbstractKernel.py:103-207) provides:

- **Automatic dimension handling** via `__call__`: Detects input dimensions and dispatches to appropriate computation method
  - 1D x 1D → `pairwise_cov_if_not_nan` (scalar output)
  - 2D x 1D → `cross_cov_vector_if_not_nan` (vector output)
  - 2D x 2D → `cross_cov_matrix` (matrix output)
  - 3D x 3D → `cross_cov_batch` (batched matrices)

- **NaN handling**: `pairwise_cov_if_not_nan` and `cross_cov_vector_if_not_nan` methods check for NaN inputs

- **Vectorization**: Uses `vmap` to efficiently build up from scalar operations to vector/matrix/batch operations

- **PyTree integration**: Implements `tree_flatten` and `tree_unflatten` for JAX transformations (gradient computation, vmap, etc.)

- **Operator overloading**: Supports `+`, `*`, and `-` operators to create composite kernels
  - `kernel1 + kernel2` → `SumKernel(kernel1, kernel2)`
  - `kernel1 * kernel2` → `ProductKernel(kernel1, kernel2)`
  - `-kernel` → `NegKernel(kernel)`

- **Hyperparameter batching**: Supports distinct hyperparameters per input via `has_distinct_hyperparameters()` and `get_hp_vmap_in_axes()`

### Kernel Categories

1. **Base Kernels** (implement `pairwise_cov` in static class):
   - RBF/Squared Exponential
   - Linear
   - Matern (1/2, 3/2, 5/2)
   - Periodic
   - Rational Quadratic
   - Constant
   - SEMagma (specialized exponential parameterization)

2. **Operator Kernels** (kernax/OperatorKernels.py): Combine two kernels
   - `SumKernel`: Adds outputs of two kernels
   - `ProductKernel`: Multiplies outputs of two kernels
   - Both auto-convert non-kernel arguments to `ConstantKernel`

3. **Wrapper Kernels** (kernax/WrapperKernels.py): Transform single kernel output
   - `ExpKernel`: Applies exponential
   - `LogKernel`: Applies logarithm
   - `NegKernel`: Negates output
   - `DiagKernel`: Returns value only when inputs are equal (creates diagonal covariance matrices)
   - All auto-convert non-kernel arguments to `ConstantKernel`

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
>>> from Kernax import RBFKernel
>>> import jax.numpy as jnp
>>> kernel = RBFKernel(length_scale=1.0, variance=1.0)
>>> kernel(jnp.array([1.0]), jnp.array([2.0]))
```

## Implementation Guidelines

### Adding a New Kernel

1. Create static class inheriting from `StaticAbstractKernel`
2. Implement `pairwise_cov(cls, kern, x1, x2)` as a `@classmethod` with `@partial(jit, static_argnums=(0,))`
3. Create instance class decorated with `@register_pytree_node_class`
4. In `__init__`, call `super().__init__(**hyperparameters)` then set `self.static_class`
5. Add both classes to `__init__.py` imports and `__all__`

### Import Patterns

- Within kernax modules: Use `from Kernax import` or relative imports `from .AbstractKernel import`
- Import inconsistency exists: Some files use `from Kernax import`, others use `from .AbstractKernel import` (both work)

### JAX-Specific Considerations

- All kernel computations must use `jax.numpy` instead of `numpy`
- Use `@jit` decorator for performance on methods that don't need static arguments
- Use `@partial(jit, static_argnums=(0,))` for static class methods (class itself is static argument)
- Hyperparameters are stored as JAX arrays and participate in automatic differentiation
- PyTree registration (`@register_pytree_node_class`) is required for gradient computation through kernels