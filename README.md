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

Currently, Kernax is not available on PyPI. Clone the repository and use it locally:

```bash
git clone https://github.com/SimLej18/Kernax
cd Kernax
```

**Requirements**:
- Python >= 3.14
- JAX >= 0.8.0

**Using Conda** (recommended):

```bash
conda create -n Kernax python=3.14
conda activate Kernax
pip install -e .
```

**Using pip**:

```bash
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from Kernax import RBFKernel, LinearKernel, DiagKernel, ExpKernel

# Create a simple RBF kernel
kernel = RBFKernel(length_scale=1.0, variance=1.0)

# Compute covariance between two points
x1 = jnp.array([1.0, 2.0])
x2 = jnp.array([1.5, 2.5])
cov = kernel(x1, x2)

# Compute covariance matrix for a set of points
X = jnp.array([[1.0], [2.0], [3.0]])
K = kernel(X, X)  # Returns 3x3 covariance matrix

# Compose kernels using operators
composite_kernel = RBFKernel(length_scale=1.0, variance=1.0) + \
                   DiagKernel(ExpKernel(0.1))  # RBF + noise

# Use with batched inputs and distinct hyperparameters
X_batched = jnp.array([...])  # Shape: (batch_size, n_points, n_dims)
length_scales = jnp.array([...])  # Shape: (batch_size,)
variances = jnp.array([...])      # Shape: (batch_size,)

batched_kernel = RBFKernel(length_scale=length_scales, variance=variances)
K_batched = batched_kernel(X_batched, X_batched)  # Shape: (batch_size, n_points, n_points)
```

## Available Kernels

### Base Kernels

- **`RBFKernel`** (Radial Basis Function / Squared Exponential)
  - Hyperparameters: `length_scale`, `variance`

- **`SEMagmaKernel`** (Squared Exponential with specialized parameterization)
  - Hyperparameters: `length_scale`, `variance`

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

- **`DiagKernel`**: Returns value only when inputs are equal (creates diagonal matrices)
- **`ExpKernel`**: Applies exponential to kernel output
- **`LogKernel`**: Applies logarithm to kernel output
- **`NegKernel`**: Negates kernel output (use `-kernel`)

## Architecture

Kernax uses a dual-class pattern for each kernel type:

1. **Static Class** (e.g., `StaticRBFKernel`): Contains JIT-compiled computation logic
2. **Instance Class** (e.g., `RBFKernel`): Holds hyperparameters as PyTree nodes

This design enables:
- Efficient JIT compilation with static methods
- Flexible hyperparameter management
- Seamless integration with JAX's gradient computation and transformations

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Benchmarks

Kernax is designed for performance. Preliminary benchmarks show:

- **Scalar operations**: ~13-15 μs per covariance computation
- **Matrix operations** (10k × 15k): ~674-855 ms
- **Batched operations** (50 batches, 100×150): ~2.35-6.37 ms
- **Composite kernels**: Minimal overhead compared to base kernels

See `benchmarks/` directory for detailed performance comparisons.

## Development Status

### Working

✅ Core kernel implementations (RBF, Linear, Matern, Periodic, etc.)
✅ Kernel composition via operators
✅ Automatic dimension handling
✅ NaN-aware computations
✅ Batched operations with distinct hyperparameters
✅ JAX PyTree integration

### In Progress / Planned

🚧 Comprehensive test suite
🚧 Documentation and tutorials
🚧 PyPI package distribution
🚧 Additional kernel types
🚧 Performance optimizations for large-scale operations

## Contributing

This project is in early development. Contributions, bug reports, and feature requests are welcome!

## Related Projects

Kernax is developed alongside [MagmaClust](../MagmaClust), a clustering and Gaussian Process library.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

[Citation information to be added]