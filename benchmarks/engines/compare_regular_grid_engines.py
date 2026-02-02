"""
Comparison benchmarks for regular grid covariance implementations.

This module compares three approaches to computing covariance matrices on regular grids:
- DenseEngine (classical implementation)
- SafeRegularGridEngine (new engine with runtime checks)
- FastRegularGridEngine (new engine without checks, assumes regular grid)

Run with: pytest benchmarks/comparison/compare_regular_grid_engines.py --benchmark-only --bench-rounds=50
"""
import jax.random as jr
import jax.numpy as jnp
import pytest
from equinox import error_if

import copy

from benchmarks.input_generators import generate_1d_regular_grid
from kernax import SEKernel, WhiteNoiseKernel
from kernax.engines import SafeRegularGridEngine, FastRegularGridEngine


class BenchmarkRegularGridSEKernel:
	"""Compare regular grid implementations for SE Kernel on 1D regular grid (1000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize(
		"implementation", ["DenseEngine", "SafeRegularGridEngine", "FastRegularGridEngine"]
	)
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark covariance computation on regular grid across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		# Branch to select implementation BEFORE benchmarking
		if implementation == "DenseEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# Classical approach with default DenseEngine
				kernel = SEKernel(length_scale=1.0)
				x = generate_1d_regular_grid(n_points=10000)
				# Warmup JIT with exact input dimensions
				res = kernel(x, x).block_until_ready()
				error_if(res, jnp.any(jnp.isnan(res)), "DenseEngine produced NaNs on regular grid input.")
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "SafeRegularGridEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# New engine approach - safe version with runtime checks
				kernel = SEKernel(length_scale=1.0, computation_engine=SafeRegularGridEngine)
				x = generate_1d_regular_grid(n_points=10000)
				# Warmup JIT with exact input dimensions
				res = kernel(x, x).block_until_ready()
				error_if(res, jnp.any(jnp.isnan(res)), "SafeRegularGridEngine produced NaNs on regular grid input.")
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "FastRegularGridEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# New engine approach - fast version (assumes regular grid)
				kernel = SEKernel(length_scale=1.0, computation_engine=FastRegularGridEngine)
				x = generate_1d_regular_grid(n_points=10000)
				# Warmup JIT with exact input dimensions
				res = kernel(x, x).block_until_ready()
				error_if(res, jnp.any(jnp.isnan(res)), "FastRegularGridEngine produced NaNs on regular grid input.")
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)


class BenchmarkRegularGridCompositeKernel:
	"""Compare regular grid implementations for composite kernel: var * SE + noise on 1D regular grid (10000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize(
		"implementation", ["DenseEngine", "SafeRegularGridEngine", "FastRegularGridEngine"]
	)
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark covariance computation with composite kernel on regular grid across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		# Kernel parameters
		variance = 2.0
		noise = 0.1

		# Branch to select implementation BEFORE benchmarking
		if implementation == "DenseEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# Classical approach: signal + noise with default DenseEngine
				signal_kernel = variance * SEKernel(length_scale=1.0)
				noise_kernel = WhiteNoiseKernel(noise)
				kernel = signal_kernel + noise_kernel
				x = generate_1d_regular_grid(n_points=10000)
				# Warmup JIT with exact input dimensions
				res = kernel(x, x).block_until_ready()
				error_if(res, jnp.any(jnp.isnan(res)), "DenseEngine produced NaNs on regular grid input.")
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "SafeRegularGridEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# Create composite kernel
				signal_kernel = variance * SEKernel(length_scale=1.0)
				noise_kernel = WhiteNoiseKernel(noise)
				kernel = signal_kernel + noise_kernel
				# Deepcopy and modify computation_engine (before any JIT compilation)
				kernel = copy.deepcopy(kernel)
				object.__setattr__(kernel, 'computation_engine', SafeRegularGridEngine)
				x = generate_1d_regular_grid(n_points=10000)
				# Warmup JIT with exact input dimensions
				res = kernel(x, x).block_until_ready()
				error_if(res, jnp.any(jnp.isnan(res)), "SafeRegularGridEngine produced NaNs on regular grid input.")
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "FastRegularGridEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# Create composite kernel
				signal_kernel = variance * SEKernel(length_scale=1.0)
				noise_kernel = WhiteNoiseKernel(noise)
				kernel = signal_kernel + noise_kernel
				# Deepcopy and modify computation_engine (before any JIT compilation)
				kernel = copy.deepcopy(kernel)
				object.__setattr__(kernel, 'computation_engine', FastRegularGridEngine)
				x = generate_1d_regular_grid(n_points=10000)
				# Warmup JIT with exact input dimensions
				res = kernel(x, x).block_until_ready()
				error_if(res, jnp.any(jnp.isnan(res)), "FastRegularGridEngine produced NaNs on regular grid input.")
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)
