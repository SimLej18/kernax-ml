# Quick Start

Once you have [installed kernax](installation.md), you can get started with the following code snippets.

## Basic kernel usage

```python
from kernax import SEKernel

# Create a squared exponential kernel with length scale 1.0
kernel = SEKernel(length_scale=1.0)

# Compute the covariance between two inputs
x1 = jnp.array([0.0])
x2 = jnp.array([1.0])
covariance = kernel(x1, x2)
print(covariance)

# Compute the cross-covariance matrix between two arrays of inputs
X1 = jnp.array([[0.0], [1.0], [2.0]])
X2 = jnp.array([[0.5], [1.5]])
cross_covariance = kernel(X1, X2)
print(cross_covariance)
```

Note the dimensionality of inputs:
* Calling the kernel with scalars (e.g. `kernel(0.0, 1.0)`) will treat them as 1D inputs and return a scalar covariance.
* Calling the kernel with 1D arrays (e.g. `kernel(jnp.array([0.0]), jnp.array([1.0]))`) will also treat them as 1D inputs and return a scalar covariance.
* Calling the kernel with 2D arrays (e.g. `kernel(jnp.array([[0.0], [1.0], [2.0]]), jnp.array([[0.5], [1.5]]))`) will treat them as arrays of inputs and return a cross-covariance matrix (here with shape `(3, 2)`).

You can also omit the second argument to compute the covariance matrix of a single array of inputs:

```python
# Compute the covariance matrix of a single array of inputs
X = jnp.array([[0.0], [1.0], [2.0]])
covariance_matrix = kernel(X)
print(covariance_matrix)  # Shape (3, 3)
```

> [!NOTE]
> Kernel instances are immutable, meaning you cannot change their parameters after creation. To modify parameters, you can create a new kernel instance with the desired values. 
> Check the [mutating parameters](../intermediate/mutate_parameters.md) and [using priors](../intermediate/using_priors.md) sections for more details on how to manage kernel parameters easily.

## Kernel composition

Kernels can be combined using regular Python operators (`+`, `*`, `-`, `/`):

```python
from kernax import SEKernel, PeriodicKernel

# Create two kernels
k1 = SEKernel(length_scale=1.0)
k2 = PeriodicKernel(length_scale=1.0, period=1.0)

# Combine them using addition and multiplication
k_sum = k1 + k2
k_product = k1 * k2

# Compute covariances with the combined kernels
x1 = jnp.array([0.0])
x2 = jnp.array([1.0])
cov_sum = k_sum(x1, x2)
cov_product = k_product(x1, x2)
print(cov_sum)
print(cov_product)
```

> [!WARNING]
> Combining kernels in specific ways (e.g. subtraction, division) may not always yield valid positive semi-definite kernels.

## Wrappers

Wrappers transform a kernel's inputs or outputs:

```python
from kernax import SEKernel, ARDKernel, ActiveDimsModule, NegModule, ExpModule, LogModule

k = SEKernel(length_scale=1.0)

# ARD: one length scale per input dimension
k_ard = ARDKernel(k, length_scales=jnp.array([1.0, 2.0]))

# ActiveDims: restrict kernel to a subset of input dimensions
k_active = ActiveDimsModule(k, dims=[0, 2])  # only use dims 0 and 2

# Unary transforms: negate, exponentiate or take the log of a kernel's output
k_neg = NegModule(k)
k_exp = ExpModule(k)
k_log = LogModule(k)
```

For batched hyperparameters (e.g. multi-task GPs), see [BatchModule](../intermediate/batching_modules.md).

## Mean functions

If you intend to use kernax for Gaussian Processes, you will likely also want to use mean functions. Kernax provides a few simple mean function classes (e.g. `ZeroMean`, `ConstantMean`, `LinearMean`, `AffineMean`) that can be used in conjunction with kernels. 

Both kernels and mean functions are `Module`s, so they expect the same inputs and can be combined, modified, wrapped and optimised in the same ways.

Here's a quick example where we sample a Gaussian Process with a `SEKernel` and a `LinearMean`:

```python
from kernax import SEKernel, LinearMean
import jax.numpy as jnp
import jax.random as jr

# Create a kernel and a mean function
kernel = SEKernel(length_scale=1.0)
mean = LinearMean(slope=1.0, intercept=0.0)

# Sample from the GP at some input locations
key = random.PRNGKey(0)
X = jnp.linspace(0, 10, 100).reshape(-1, 1)  # Shape (100, 1)
samples = jr.multivariate_normal(key, mean(X), kernel(X))  # Shape (100,)
print(samples)  # Shape (100,)
```

## Next steps

This quick start guide only scratches the surface of what you can do with kernax. Here are the next pages to visit depending on your interests:

* [Key concepts](key_concepts.md) — learn about kernels in general, and how kernax implements them to go *fast*
* [All kernels](all_kernels.md) — see the full list of kernels available in kernax, with links to their API docs and examples
* [Should I use kernax?](should_i_use_kernax.md) — find out if kernax is the right choice for your project, and how it compares to other libraries
* [The sharp bits](sharp_bits.md) — avoid common pitfalls and gotchas when using kernax, and learn some pro tips for getting the most out of it
* [Examples](../examples/index.md) — see how to use kernax for various applications like GP regression, SVMs, etc.
* [Intermediate topics](../intermediate/index.md) — learn how to mutate parameters, batch modules, create custom modules, use priors, benchmark and save your models
* [Advanced topics](../advanced/index.md) — dive into computation engines, formatting, reparameterization and constraints for expert-level control over performance and flexibility
* [API reference](../api/index.md) — browse the full API reference for all classes, functions and utilities in kernax.
