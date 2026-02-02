# Getting Started

This guide will help you get started with Kernax.

## Basic Usage

### Creating a Kernel

```python
import jax.numpy as jnp
from kernax import RBFKernel

# Create an RBF kernel with specified hyperparameters
kernel = RBFKernel(length_scale=1.0, variance=1.0)
```

### Computing Covariances

#### Scalar to Scalar

```python
x1 = jnp.array([1.0])
x2 = jnp.array([2.0])
cov = kernel(x1, x2)
```

#### Vector to Vector (Covariance Matrix)

```python
X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
K = kernel(X, X)  # 4x4 covariance matrix
```

#### Cross-Covariance

```python
X1 = jnp.array([[1.0], [2.0], [3.0]])
X2 = jnp.array([[1.5], [2.5]])
K_cross = kernel(X1, X2)  # 3x2 cross-covariance matrix
```

## Composing Kernels

Kernax supports kernel composition through operator overloading:

### Sum of Kernels

```python
from kernax import RBFKernel, LinearKernel

k1 = RBFKernel(length_scale=1.0, variance=1.0)
k2 = LinearKernel(variance_b=0.5, variance_v=1.0, offset_c=0.0)
kernel = k1 + k2
```

### Product of Kernels

```python
kernel = k1 * k2
```

### Adding Noise

A common pattern is to add noise on the diagonal:

```python
from kernax import RBFKernel, WhiteNoiseKernel

signal = RBFKernel(length_scale=1.0, variance=1.0)
noise = WhiteNoiseKernel(0.1)
kernel = signal + noise
```

## Batched Operations

Kernax supports batched computations for efficient processing:

### Shared Hyperparameters

```python
# Input: (batch_size, n_points, n_dims)
X_batched = jnp.ones((10, 5, 1))

kernel = RBFKernel(length_scale=1.0, variance=1.0)
K_batched = kernel(X_batched, X_batched)  # (10, 5, 5)
```

### Distinct Hyperparameters

```python
# Different hyperparameters for each batch element
batch_size = 10
length_scales = jnp.linspace(0.5, 2.0, batch_size)
variances = jnp.ones(batch_size)

kernel = RBFKernel(length_scale=length_scales, variance=variances)
K_batched = kernel(X_batched, X_batched)
```

## Working with NaN Values

Kernax handles NaN values gracefully, useful for padded sequences:

```python
import jax.numpy as jnp

# Create padded data (different sequence lengths)
X = jnp.array([
    [1.0],
    [2.0],
    [3.0],
    [jnp.nan],  # Padding
    [jnp.nan],  # Padding
])

kernel = RBFKernel(length_scale=1.0, variance=1.0)
K = kernel(X, X)

# Rows/columns corresponding to NaN will be NaN
# Valid regions will have finite values
```

## Next Steps

- Explore the [API Reference](api/index.md) for all available kernels
- Check out [Examples](examples/index.md) for more use cases
- Read [CLAUDE.md](../CLAUDE.md) for architecture details