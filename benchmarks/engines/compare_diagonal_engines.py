"""
Comparison benchmarks for diagonal covariance implementations.

This module compares three approaches to computing diagonal covariance matrices:
- DiagKernel (current wrapper implementation)
- SafeDiagonalEngine (new engine, safe for any inputs)
- FastDiagonalEngine (new engine, assumes x1 == x2)

Each benchmark class represents a specific kernel type, with a single parametrized test
comparing the three implementations. Setup handles data generation and conversions,
while the timed section measures only the kernel computation.

Run with: pytest benchmarks/comparison/compare_diagonal_implementations.py --benchmark-only --bench-rounds=50
"""
import jax.random as jr
import pytest

from benchmarks.input_generators import generate_random_inputs
from kernax import SEKernel, PolynomialKernel, DiagKernel
from kernax.engines import SafeDiagonalEngine, FastDiagonalEngine


class BenchmarkDiagonalSEKernel:
	"""Compare diagonal implementations for SE Kernel on 2D random inputs (1000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize(
		"implementation", ["DiagKernel", "SafeDiagonalEngine", "FastDiagonalEngine"]
	)
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark diagonal covariance computation across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		# Branch to select implementation BEFORE benchmarking
		if implementation == "DiagKernel":

			def setup():
				self.key, subkey = jr.split(self.key)
				# Current wrapper approach
				base_kernel = SEKernel(length_scale=1.0)
				kernel = DiagKernel(base_kernel)
				x = generate_random_inputs(subkey, n_points=1000, n_dims=2, min_val=-20, max_val=20)
				# Warmup JIT with exact input dimensions
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "SafeDiagonalEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# New engine approach - safe version
				kernel = SEKernel(length_scale=1.0, computation_engine=SafeDiagonalEngine)
				x = generate_random_inputs(subkey, n_points=1000, n_dims=2, min_val=-20, max_val=20)
				# Warmup JIT with exact input dimensions
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "FastDiagonalEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# New engine approach - fast version (assumes x1 == x2)
				kernel = SEKernel(length_scale=1.0, computation_engine=FastDiagonalEngine)
				x = generate_random_inputs(subkey, n_points=1000, n_dims=2, min_val=-20, max_val=20)
				# Warmup JIT with exact input dimensions
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)


class BenchmarkDiagonalPolynomialKernel:
	"""Compare diagonal implementations for Polynomial Kernel on 2D random inputs (1000 points)."""

	@classmethod
	def setup_class(cls):
		"""Initialize PRNG key for the class."""
		cls.key = jr.PRNGKey(42)

	@pytest.mark.parametrize(
		"implementation", ["DiagKernel", "SafeDiagonalEngine", "FastDiagonalEngine"]
	)
	def test_compare(self, benchmark, request, implementation):
		"""Benchmark diagonal covariance computation across implementations."""
		rounds = int(request.config.getoption("--bench-rounds"))

		# Branch to select implementation BEFORE benchmarking
		if implementation == "DiagKernel":

			def setup():
				self.key, subkey = jr.split(self.key)
				# Current wrapper approach
				base_kernel = PolynomialKernel(degree=3, gamma=1.0, constant=1.0)
				kernel = DiagKernel(base_kernel)
				x = generate_random_inputs(subkey, n_points=1000, n_dims=2, min_val=-20, max_val=20)
				# Warmup JIT with exact input dimensions
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "SafeDiagonalEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# New engine approach - safe version
				kernel = PolynomialKernel(
					degree=3, gamma=1.0, constant=1.0, computation_engine=SafeDiagonalEngine
				)
				x = generate_random_inputs(subkey, n_points=1000, n_dims=2, min_val=-20, max_val=20)
				# Warmup JIT with exact input dimensions
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		elif implementation == "FastDiagonalEngine":

			def setup():
				self.key, subkey = jr.split(self.key)
				# New engine approach - fast version (assumes x1 == x2)
				kernel = PolynomialKernel(
					degree=3, gamma=1.0, constant=1.0, computation_engine=FastDiagonalEngine
				)
				x = generate_random_inputs(subkey, n_points=1000, n_dims=2, min_val=-20, max_val=20)
				# Warmup JIT with exact input dimensions
				kernel(x, x).block_until_ready()
				return (kernel, x, x), {}

			def run_kernel(kernel, x1, x2):
				kernel(x1, x2).block_until_ready()

		benchmark.pedantic(run_kernel, setup=setup, rounds=rounds, iterations=1)