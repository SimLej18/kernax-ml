# Kernax

A JAX-based kernel library for Gaussian Processes with automatic differentiation, JIT compilation, and composable kernel operations.

> **⚠️ Project Status**: Kernax is in early development. The API may change, and some features are still experimental.

## Features

- **Fast JIT-compiled computations** using JAX's `@jit` decorator
- **Automatic dimension handling** for scalars, vectors, matrices, and batched operations
- **NaN-aware computations** for working with padded/masked data
- **Composable kernels** through operator overloading (`+`, `*`, `-`)
- **Distinct hyperparameters per batch** for efficient multi-task learning
- **PyTree integration** for seamless use with JAX transformations (grad, vmap, etc.)

## Installation

Install from PyPI:

```bash
pip install kernax-ml
```

Or clone the repository for development:

```bash
git clone https://github.com/SimLej18/kernax-ml
cd kernax-ml
```

**Requirements**:
- Python >= 3.14
- JAX >= 0.8.0

**Using Conda** (recommended):

```bash
conda create -n kernax-ml python=3.14
conda activate kernax-ml
pip install -e .
```

**Using pip**:

```bash
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from kernax import SEKernel, LinearKernel, DiagKernel, ExpKernel, BatchKernel, ARDKernel

# Create a simple Squared Exponential kernel
kernel = SEKernel(length_scale=1.0)

# Compute covariance between two points
x1 = jnp.array([1.0, 2.0])
x2 = jnp.array([1.5, 2.5])
cov = kernel(x1, x2)

# Compute covariance matrix for a set of points
X = jnp.array([[1.0], [2.0], [3.0]])
K = kernel(X, X)  # Returns 3x3 covariance matrix

# Compose kernels using operators
composite_kernel = SEKernel(length_scale=1.0) + DiagKernel(ExpKernel(0.1))  # SE + noise

# Use BatchKernel for distinct hyperparameters per batch
base_kernel = SEKernel(length_scale=1.0)
batched_kernel = BatchKernel(base_kernel, batch_size=10, batch_in_axes=0, batch_over_inputs=True)

# Use ARDKernel for Automatic Relevance Determination
length_scales = jnp.array([1.0, 2.0, 0.5])  # Different scale per dimension
ard_kernel = ARDKernel(SEKernel(length_scale=1.0), length_scales=length_scales)
```

## Available Kernels

### Base Kernels

- **`SEKernel`** (Squared Exponential, aka RBF or Gaussian)
  - Hyperparameters: `length_scale`

- **`LinearKernel`**
  - Hyperparameters: `variance_b`, `variance_v`, `offset_c`

- **`MaternKernel`** family
  - `Matern12Kernel` (ν=1/2, equivalent to Exponential)
  - `Matern32Kernel` (ν=3/2)
  - `Matern52Kernel` (ν=5/2)
  - Hyperparameters: `length_scale`

- **`PeriodicKernel`**
  - Hyperparameters: `length_scale`, `variance`, `period`

- **`RationalQuadraticKernel`**
  - Hyperparameters: `length_scale`, `variance`, `alpha`

- **`ConstantKernel`**
  - Hyperparameters: `value`

### Composite Kernels

- **`SumKernel`**: Adds two kernels (use `kernel1 + kernel2`)
- **`ProductKernel`**: Multiplies two kernels (use `kernel1 * kernel2`)

### Wrapper Kernels

Transform or modify kernel behavior:

- **`DiagKernel`**: Returns value only when inputs are equal (creates diagonal matrices)
- **`ExpKernel`**: Applies exponential to kernel output
- **`LogKernel`**: Applies logarithm to kernel output
- **`NegKernel`**: Negates kernel output (use `-kernel`)
- **`BatchKernel`**: Adds batch handling with distinct hyperparameters per batch
- **`ActiveDimsKernel`**: Selects specific input dimensions before kernel computation
- **`ARDKernel`**: Applies Automatic Relevance Determination (different length scale per dimension)

## Architecture

Kernax is built on [Equinox](https://github.com/patrick-kidger/equinox), making kernels PyTorch-like modules with clean differentiation.

Each kernel uses a dual-class pattern:

1. **Static Class** (e.g., `StaticSEKernel`): Contains JIT-compiled computation logic
2. **Instance Class** (e.g., `SEKernel`): Extends `eqx.Module`, holds hyperparameters

This design enables:
- Efficient JIT compilation with Equinox's `filter_jit`
- Automatic PyTree registration through `eqx.Module`
- Seamless integration with JAX transformations (grad, vmap, etc.)
- Clean hyperparameter management with automatic array conversion

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Benchmarks

Kernax is designed for performance. Preliminary benchmarks show:

- **Scalar operations**: ~13-15 μs per covariance computation
- **Matrix operations** (10k × 15k): ~674-855 ms
- **Batched operations** (50 batches, 100×150): ~2.35-6.37 ms
- **Composite kernels**: Minimal overhead compared to base kernels

See `benchmarks/` directory for detailed performance comparisons.

## Development Status

### ✅ Completed

- Core kernel implementations (SE, Linear, Matern, Periodic, etc.)
- Kernel composition via operators
- Automatic dimension handling
- NaN-aware computations
- Equinox Module integration
- BatchKernel wrapper for batched hyperparameters
- ARDKernel wrapper for Automatic Relevance Determination
- ActiveDimsKernel wrapper for dimension selection

### 🚧 In Progress / Planned

- Rewrite inheritance with StationaryKernel and IsotropicKernel base classes
- Add computation engines for special cases (diagonal-only, etc.)
- Comprehensive test suite covering all new features
- Expanded documentation and tutorials
- PyPI package distribution
- Benchmarks against other libraries (GPJax, TinyGP, etc.)


## Contributing

This project is in early development. Contributions, bug reports, and feature requests are welcome!

## Related Projects

Kernax is developed alongside [MagmaClust](https://github.com/SimLej18/MagmaClustPy), a clustering and Gaussian Process library.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

[Citation information to be added]