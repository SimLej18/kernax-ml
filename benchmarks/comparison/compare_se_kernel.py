"""
Cross-library comparison benchmarks for SE/RBF Kernel.

This module compares kernax SEKernel performance against other popular libraries:
- scikit-learn (RBF)
- GPyTorch (RBFKernel)
- GPJax (RBF)

Each benchmark class represents a specific input scenario, with a single parametrized test
comparing different library implementations. Setup handles data generation and conversions,
while the timed section measures only the kernel computation.

Run with: make benchmarks-compare
Customize: pytest benchmarks/comparison/ --benchmark-only --bench-rounds=50
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from benchmarks.input_generators import (
	add_missing_values,
	generate_1d_regular_grid,
	generate_2d_regular_grid,
	generate_random_inputs,
)


class Benchmark1DRegularGrid:
	"""Compare SE/RBF kernel implementations on 1D regular grid (10000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize("implementation", ["kernax", "sklearn", "gpytorch", "gpjax"])
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark SE/RBF kernel across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		# Branch to select implementation BEFORE benchmarking
		if implementation == "kernax":
			from kernax import SEKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = SEKernel(length_scale=1.0)
				x = generate_1d_regular_grid(n_points=10000)
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "sklearn":
			from sklearn.gaussian_process.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(length_scale=1.0)
				x = generate_1d_regular_grid(n_points=10000)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2)

		elif implementation == "gpytorch":
			import torch
			from gpytorch.kernels import RBFKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBFKernel()
				kernel._set_lengthscale(1.0)
				x = generate_1d_regular_grid(n_points=10000)
				x_torch = torch.tensor(x, dtype=torch.float32)
				return (kernel, x_torch, x_torch), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).evaluate()

		elif implementation == "gpjax":
			from gpjax.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(lengthscale=jnp.array([1.0]))
				x = generate_1d_regular_grid(n_points=10000)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel.cross_covariance(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)


class Benchmark1DRandom:
	"""Compare SE/RBF kernel implementations on 1D random inputs (10000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize("implementation", ["kernax", "sklearn", "gpytorch", "gpjax"])
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark SE/RBF kernel across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		if implementation == "kernax":
			from kernax import SEKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = SEKernel(length_scale=1.0)
				x = generate_random_inputs(subkey, n_points=10000, n_dims=1, min_val=-500, max_val=500)
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "sklearn":
			from sklearn.gaussian_process.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(length_scale=1.0)
				x = generate_random_inputs(subkey, n_points=10000, n_dims=1, min_val=-500, max_val=500)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2)

		elif implementation == "gpytorch":
			import torch
			from gpytorch.kernels import RBFKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBFKernel()
				kernel._set_lengthscale(1.0)
				x = generate_random_inputs(subkey, n_points=10000, n_dims=1, min_val=-500, max_val=500)
				x_torch = torch.tensor(x, dtype=torch.float32)
				return (kernel, x_torch, x_torch), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).evaluate()

		elif implementation == "gpjax":
			from gpjax.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(lengthscale=jnp.array([1.0]))
				x = generate_random_inputs(subkey, n_points=10000, n_dims=1, min_val=-500, max_val=500)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel.cross_covariance(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)


class Benchmark2DRegularGrid:
	"""Compare SE/RBF kernel implementations on 2D regular grid (10000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize("implementation", ["kernax", "sklearn", "gpytorch", "gpjax"])
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark SE/RBF kernel across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		if implementation == "kernax":
			from kernax import SEKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = SEKernel(length_scale=1.0)
				x = generate_2d_regular_grid(n_points_per_dim=100)
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "sklearn":
			from sklearn.gaussian_process.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(length_scale=1.0)
				x = generate_2d_regular_grid(n_points_per_dim=100)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2)

		elif implementation == "gpytorch":
			import torch
			from gpytorch.kernels import RBFKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBFKernel()
				kernel._set_lengthscale(1.0)
				x = generate_2d_regular_grid(n_points_per_dim=100)
				x_torch = torch.tensor(x, dtype=torch.float32)
				return (kernel, x_torch, x_torch), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).evaluate()

		elif implementation == "gpjax":
			from gpjax.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(lengthscale=jnp.array([1.0, 1.0]))
				x = generate_2d_regular_grid(n_points_per_dim=100)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel.cross_covariance(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)


class Benchmark2DRandom:
	"""Compare SE/RBF kernel implementations on 2D random inputs (10000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize("implementation", ["kernax", "sklearn", "gpytorch", "gpjax"])
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark SE/RBF kernel across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		if implementation == "kernax":
			from kernax import SEKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = SEKernel(length_scale=1.0)
				x = generate_random_inputs(subkey, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "sklearn":
			from sklearn.gaussian_process.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(length_scale=1.0)
				x = generate_random_inputs(subkey, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2)

		elif implementation == "gpytorch":
			import torch
			from gpytorch.kernels import RBFKernel

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBFKernel()
				kernel._set_lengthscale(1.0)
				x = generate_random_inputs(subkey, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				x_torch = torch.tensor(x, dtype=torch.float32)
				return (kernel, x_torch, x_torch), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).evaluate()

		elif implementation == "gpjax":
			from gpjax.kernels import RBF

			def setup():
				self.key, subkey = jr.split(self.key)
				kernel = RBF(lengthscale=jnp.array([1.0, 1.0]))
				x = generate_random_inputs(subkey, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel.cross_covariance(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)


class Benchmark2DMissingValues:
	"""Compare SE/RBF kernel implementations on 2D data with missing values (10000 points, 25% NaN)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize("implementation", ["kernax", "sklearn", "gpytorch", "gpjax"])
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark SE/RBF kernel across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		if implementation == "kernax":
			from kernax import SEKernel

			def setup():
				self.key, subkey1, subkey2 = jr.split(self.key, 3)
				kernel = SEKernel(length_scale=1.0)
				x = generate_random_inputs(subkey1, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				x = add_missing_values(subkey2, x, missing_rate=0.25)
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "sklearn":
			import numpy as np
			from sklearn.gaussian_process.kernels import RBF

			def setup():
				self.key, subkey1, subkey2 = jr.split(self.key, 3)
				kernel = RBF(length_scale=1.0)
				x = generate_random_inputs(subkey1, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				x = add_missing_values(subkey2, x, missing_rate=0.25)
				x_clean = x[~np.isnan(x).any(axis=1)]
				return (kernel, x_clean, x_clean), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2)

		elif implementation == "gpytorch":
			import numpy as np
			import torch
			from gpytorch.kernels import RBFKernel

			def setup():
				self.key, subkey1, subkey2 = jr.split(self.key, 3)
				kernel = RBFKernel()
				kernel._set_lengthscale(1.0)
				x = generate_random_inputs(subkey1, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				x = add_missing_values(subkey2, x, missing_rate=0.25)
				x_clean = x[~np.isnan(x).any(axis=1)]
				x_torch = torch.tensor(x_clean, dtype=torch.float32)
				return (kernel, x_torch, x_torch), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).evaluate()

		elif implementation == "gpjax":
			import numpy as np
			from gpjax.kernels import RBF

			def setup():
				self.key, subkey1, subkey2 = jr.split(self.key, 3)
				kernel = RBF(lengthscale=jnp.array([1.0, 1.0]))
				x = generate_random_inputs(subkey1, n_points=10000, n_dims=2, min_val=-20, max_val=20)
				x = add_missing_values(subkey2, x, missing_rate=0.25)
				x_clean = x[~np.isnan(x).any(axis=1)]
				return (kernel, x_clean, x_clean), {}

			def run_kernel(kernel, x1, x2):
				kernel.cross_covariance(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)
