"""
Benchmarks for SE (Squared Exponential) Kernel.

This module benchmarks SEKernel performance across various input scenarios:
- Different dimensions (1D, 2D)
- Different input types (regular grids, random, missing values)
- Batch processing with shared and distinct hyperparameters

Each test:
1. Instantiates the kernel in setup()
2. Generates appropriate inputs in setup() (with varying random keys per round)
3. Warms up JIT compilation with those exact inputs in setup()
4. Measures execution time (JIT overhead already handled)

Run with: make benchmarks
Customize: pytest benchmarks/ --benchmark-only --bench-rounds=50
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from kernax import SEKernel, BatchKernel
from benchmarks.input_generators import (
	generate_1d_regular_grid,
	generate_2d_regular_grid,
	generate_random_inputs,
	add_missing_values,
	generate_batched_random_inputs,
)


class BenchmarkSEKernel:
	"""
	Benchmark suite for Squared Exponential (SE) Kernel.

	Tests performance across various input scenarios to understand:
	- Scaling with dimensionality
	- Performance on regular vs random data
	- Impact of missing values (NaN handling)
	- Batch processing efficiency
	"""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	# ==================== 1D Benchmarks ====================

	def benchmark_1d_regular_grid(self, benchmark, request):
		"""Benchmark SE kernel on 1D regular grid (10000 points)."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey = jr.split(self.key)

			# Instantiate kernel
			kernel = SEKernel(length_scale=1.0)

			# Generate inputs (regular grid, so key not used but we split for consistency)
			x = generate_1d_regular_grid(n_points=10000)

			# Warmup JIT with exact input dimensions
			kernel(x, x).block_until_ready()

			return (kernel, x, x), {}

		def run_kernel(kernel, x1, x2):
			result = kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	def benchmark_1d_random(self, benchmark, request):
		"""Benchmark SE kernel on 1D random inputs (10000 points)."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey = jr.split(self.key)

			# Instantiate kernel
			kernel = SEKernel(length_scale=1.0)

			# Generate random inputs (varies per round)
			x = generate_random_inputs(subkey, n_points=10000, n_dims=1, min_val=-500, max_val=500)

			# Warmup JIT with exact input dimensions
			kernel(x, x).block_until_ready()

			return (kernel, x, x), {}

		def run_kernel(kernel, x1, x2):
			result = kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	# ==================== 2D Benchmarks ====================

	def benchmark_2d_regular_grid(self, benchmark, request):
		"""Benchmark SE kernel on 2D regular grid (10000 points)."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey = jr.split(self.key)

			# Instantiate kernel
			kernel = SEKernel(length_scale=1.0)

			# Generate 2D grid inputs
			x = generate_2d_regular_grid(n_points_per_dim=100)

			# Warmup JIT with exact input dimensions
			kernel(x, x).block_until_ready()

			return (kernel, x, x), {}

		def run_kernel(kernel, x1, x2):
			result = kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	def benchmark_2d_random(self, benchmark, request):
		"""Benchmark SE kernel on 2D random inputs (10000 points)."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey = jr.split(self.key)

			# Instantiate kernel
			kernel = SEKernel(length_scale=1.0)

			# Generate random inputs (varies per round)
			x = generate_random_inputs(subkey, n_points=10000, n_dims=2, min_val=-20, max_val=20)

			# Warmup JIT with exact input dimensions
			kernel(x, x).block_until_ready()

			return (kernel, x, x), {}

		def run_kernel(kernel, x1, x2):
			result = kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	def benchmark_2d_missing_values(self, benchmark, request):
		"""Benchmark SE kernel on 2D data with 25% missing values (10000 points)."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey1, subkey2 = jr.split(self.key, 3)

			# Instantiate kernel
			kernel = SEKernel(length_scale=1.0)

			# Generate random inputs (varies per round)
			x = generate_random_inputs(subkey1, n_points=10000, n_dims=2, min_val=-20, max_val=20)

			# Add missing values
			x = add_missing_values(subkey2, x, missing_rate=0.25)

			# Warmup JIT with exact input dimensions (including NaN handling path)
			kernel(x, x).block_until_ready()

			return (kernel, x, x), {}

		def run_kernel(kernel, x1, x2):
			result = kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	# ==================== Batch 1D Benchmarks ====================

	def benchmark_batch_1d_common_hps(self, benchmark, request):
		"""Benchmark BatchKernel on 1D inputs (100 batches × 100 points) with shared hyperparameters."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey = jr.split(self.key)

			# Instantiate base kernel
			base_kernel = SEKernel(length_scale=1.0)

			# Wrap in BatchKernel with shared hyperparameters
			batch_kernel = BatchKernel(
				base_kernel,
				batch_size=100,
				batch_in_axes=None,  # Shared hyperparameters
				batch_over_inputs=True,
			)

			# Generate batched random inputs (varies per round)
			x = generate_batched_random_inputs(
				subkey, batch_size=100, n_points=100, n_dims=1, min_val=-25, max_val=25
			)

			# Warmup JIT with exact input dimensions
			batch_kernel(x, x).block_until_ready()

			return (batch_kernel, x, x), {}

		def run_kernel(batch_kernel, x1, x2):
			result = batch_kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	def benchmark_batch_1d_batched_hps(self, benchmark, request):
		"""Benchmark BatchKernel on 1D inputs (100 batches × 100 points) with distinct hyperparameters per batch."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey1, subkey2 = jr.split(self.key, 3)

			# Instantiate base kernel
			base_kernel = SEKernel(length_scale=1.0)

			# Wrap in BatchKernel with batched hyperparameters
			batch_kernel = BatchKernel(
				base_kernel,
				batch_size=100,
				batch_in_axes=0,  # Batched hyperparameters
				batch_over_inputs=True,
			)

			# Modify hyperparameters of inner kernel with random multipliers
			random_multipliers = jr.uniform(subkey1, (100,), minval=0.5, maxval=1.5)
			new_inner_kernel = jtu.tree_map(
				lambda param: param * random_multipliers if param.shape[0] == 100 else param,
				batch_kernel.inner_kernel,
			)
			batch_kernel = eqx.tree_at(lambda bk: bk.inner_kernel, batch_kernel, new_inner_kernel)

			# Generate batched random inputs (varies per round)
			x = generate_batched_random_inputs(
				subkey2, batch_size=100, n_points=100, n_dims=1, min_val=-25, max_val=25
			)

			# Warmup JIT with exact input dimensions
			batch_kernel(x, x).block_until_ready()

			return (batch_kernel, x, x), {}

		def run_kernel(batch_kernel, x1, x2):
			result = batch_kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	# ==================== Batch 2D Benchmarks ====================

	def benchmark_batch_2d_common_hps(self, benchmark, request):
		"""Benchmark BatchKernel on 2D inputs (100 batches × 100 points) with shared hyperparameters."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey = jr.split(self.key)

			# Instantiate base kernel
			base_kernel = SEKernel(length_scale=1.0)

			# Wrap in BatchKernel with shared hyperparameters
			batch_kernel = BatchKernel(
				base_kernel,
				batch_size=100,
				batch_in_axes=None,  # Shared hyperparameters
				batch_over_inputs=True,
			)

			# Generate batched random inputs (varies per round)
			x = generate_batched_random_inputs(
				subkey, batch_size=100, n_points=100, n_dims=2, min_val=-2, max_val=2
			)

			# Warmup JIT with exact input dimensions
			batch_kernel(x, x).block_until_ready()

			return (batch_kernel, x, x), {}

		def run_kernel(batch_kernel, x1, x2):
			result = batch_kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)

	def benchmark_batch_2d_batched_hps(self, benchmark, request):
		"""Benchmark BatchKernel on 2D inputs (100 batches × 100 points) with distinct hyperparameters per batch."""
		rounds = int(request.config.getoption("--bench-rounds"))

		def setup():
			# Split key to get new data each round
			self.key, subkey1, subkey2 = jr.split(self.key, 3)

			# Instantiate base kernel
			base_kernel = SEKernel(length_scale=1.0)

			# Wrap in BatchKernel with batched hyperparameters
			batch_kernel = BatchKernel(
				base_kernel,
				batch_size=100,
				batch_in_axes=0,  # Batched hyperparameters
				batch_over_inputs=True,
			)

			# Modify hyperparameters of inner kernel with random multipliers
			random_multipliers = jr.uniform(subkey1, (100,), minval=0.5, maxval=1.5)
			new_inner_kernel = jtu.tree_map(
				lambda param: param * random_multipliers if param.shape[0] == 100 else param,
				batch_kernel.inner_kernel,
			)
			batch_kernel = eqx.tree_at(lambda bk: bk.inner_kernel, batch_kernel, new_inner_kernel)

			# Generate batched random inputs (varies per round)
			x = generate_batched_random_inputs(
				subkey2, batch_size=100, n_points=100, n_dims=2, min_val=-2, max_val=2
			)

			# Warmup JIT with exact input dimensions
			batch_kernel(x, x).block_until_ready()

			return (batch_kernel, x, x), {}

		def run_kernel(batch_kernel, x1, x2):
			result = batch_kernel(x1, x2)
			result.block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)
