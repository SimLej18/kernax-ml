"""
Tests for wrapper kernels (BatchKernel, ActiveDimsKernel, ARDKernel).
"""

import jax.numpy as jnp
import pytest
import allure

from kernax import (
	ActiveDimsKernel,
	ARDKernel,
	BatchKernel,
	SEKernel,
	DiagKernel,
	ConstantKernel,
)


class TestDiagKernel:
	"""Tests for Diagonal Kernel (White Noise)."""

	@allure.title("DiagKernel Instantiation")
	@allure.description("Test that Diagonal kernel can be instantiated.")
	def test_instantiation(self):
		inner_kernel = ConstantKernel(value=1.0)
		kernel = DiagKernel(inner_kernel)
		assert kernel.inner_kernel == inner_kernel

	@allure.title("DiagKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		inner_kernel = ConstantKernel(value=2.0)
		kernel = DiagKernel(inner_kernel)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		x3 = jnp.array([1.0])

		# Different points should give 0
		result_diff = kernel(x1, x2)
		assert result_diff.shape == ()
		assert jnp.allclose(result_diff, 0.0)

		# Same points should give inner kernel value
		result_same = kernel(x1, x3)
		assert result_same.shape == ()
		assert jnp.allclose(result_same, 2.0)

	@allure.title("DiagKernel cross-cov computation")
	@allure.description("Test cross-covariance computation creates diagonal matrix.")
	def test_cross_cov_computation(self, sample_1d_data):
		inner_kernel = ConstantKernel(value=1.5)
		kernel = DiagKernel(inner_kernel)
		x1, _ = sample_1d_data

		# Compute covariance matrix with itself
		result = kernel(x1, x1)
		assert result.shape == (x1.shape[0], x1.shape[0])
		assert jnp.all(jnp.isfinite(result))

		# Should be diagonal matrix with inner kernel values on diagonal
		expected_diag = jnp.ones(x1.shape[0]) * 1.5
		assert jnp.allclose(jnp.diag(result), expected_diag)

		# Off-diagonal elements should be 0
		mask = ~jnp.eye(x1.shape[0], dtype=bool)
		assert jnp.allclose(result[mask], 0.0)

	@allure.title("DiagKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		inner_kernel = ConstantKernel(value=1.0)
		kernel = DiagKernel(inner_kernel)
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check it's a diagonal matrix
		assert jnp.allclose(K, jnp.diag(jnp.diag(K)))

		# Check diagonal elements equal inner kernel value
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("DiagKernel with SE inner kernel")
	@allure.description("Test DiagKernel with SE kernel as inner kernel.")
	def test_with_se_inner_kernel(self):
		inner_kernel = SEKernel(length_scale=1.0)
		kernel = DiagKernel(inner_kernel)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		# Different points should give 0
		result_diff = kernel(x1, x2)
		assert jnp.allclose(result_diff, 0.0)

		# Same point should give SE kernel value (which is 1 for identical points)
		result_same = kernel(x1, x1)
		assert jnp.allclose(result_same, 1.0)

	@allure.title("DiagKernel comparison with scikit-learn")
	@allure.description("Compare Diagonal kernel results against scikit-learn WhiteKernel.")
	def test_against_scikitlearn(self, sample_1d_data):
		from sklearn.gaussian_process.kernels import WhiteKernel

		inner_kernel = ConstantKernel(value=1.0)
		kernel = DiagKernel(inner_kernel)
		sklearn_kernel = WhiteKernel(noise_level=1.0)

		x1, _ = sample_1d_data
		result = kernel(x1)
		expected = sklearn_kernel(x1)

		assert jnp.allclose(result, expected)


class TestBatchKernel:
	"""Tests for BatchKernel wrapper."""

	@allure.title("BatchKernel Instantiation")
	@allure.description("Test that BatchKernel can be instantiated.")
	def test_instantiation(self):
		base_kernel = SEKernel(length_scale=1.0)
		batch_kernel = BatchKernel(
			base_kernel, batch_size=5, batch_in_axes=0, batch_over_inputs=True
		)
		assert batch_kernel.inner_kernel is not None
		assert batch_kernel.batch_over_inputs == 0

	@allure.title("BatchKernel batch over hyperparameters")
	@allure.description("Test batching with distinct hyperparameters per batch element.")
	def test_batch_over_hyperparameters(self):
		# Create base kernel with single length_scale
		base_kernel = SEKernel(length_scale=1.0)
		batch_size = 3

		# Wrap in BatchKernel to handle batched hyperparameters
		batch_kernel = BatchKernel(
			base_kernel,
			batch_size=batch_size,
			batch_in_axes=0,  # Batch over all hyperparameters
			batch_over_inputs=False,  # Same inputs for all batches
		)

		# Create non-batched inputs
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5]])

		# Compute covariance - should produce batched output
		result = batch_kernel(x1, x2)

		# Result should have batch dimension
		assert result.shape == (batch_size, x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BatchKernel batch over inputs and hyperparameters")
	@allure.description("Test batching over both inputs and hyperparameters.")
	def test_batch_over_inputs_and_hyperparameters(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x1_batched, x2_batched = sample_batched_data
		batch_size = x1_batched.shape[0]

		batch_kernel = BatchKernel(
			base_kernel, batch_size=batch_size, batch_in_axes=0, batch_over_inputs=True
		)

		result = batch_kernel(x1_batched, x1_batched)

		# Should produce batch_size covariance matrices
		assert result.shape == (batch_size, x1_batched.shape[1], x1_batched.shape[1])
		assert jnp.all(jnp.isfinite(result))

		# Each batch element should be symmetric
		for i in range(batch_size):
			assert jnp.allclose(result[i], result[i].T)

	@allure.title("BatchKernel batch over inputs only")
	@allure.description("Test batching over inputs with shared hyperparameters.")
	def test_batch_over_inputs_only(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		batch_size = x_batched.shape[0]

		# Batch over inputs but share hyperparameters
		batch_kernel = BatchKernel(
			base_kernel,
			batch_size=batch_size,
			batch_in_axes=None,  # Shared hyperparameters
			batch_over_inputs=True,
		)

		result = batch_kernel(x_batched, x_batched)

		assert result.shape == (batch_size, x_batched.shape[1], x_batched.shape[1])
		assert jnp.all(jnp.isfinite(result))


class TestActiveDimsKernel:
	"""Tests for ActiveDimsKernel wrapper."""

	@allure.title("ActiveDimsKernel Instantiation")
	@allure.description("Test that ActiveDimsKernel can be instantiated.")
	def test_instantiation(self):
		base_kernel = SEKernel(length_scale=1.0)
		active_dims = jnp.array([0, 2])
		kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

		assert kernel.inner_kernel is not None
		assert jnp.array_equal(kernel.active_dims, active_dims)

	@allure.title("ActiveDimsKernel dimension selection")
	@allure.description("Test that kernel only uses specified dimensions.")
	def test_dimension_selection(self):
		base_kernel = SEKernel(length_scale=1.0)

		# Only use first and third dimensions
		active_dims = jnp.array([0, 2])
		kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

		# Create 3D input
		x1 = jnp.array([[1.0, 5.0, 2.0]])  # Shape: (1, 3)
		x2 = jnp.array([[1.5, 99.0, 2.5]])  # Shape: (1, 3), middle dim very different

		# Compute with active dims kernel
		result = kernel(x1, x2)

		# Compute expected result using only selected dimensions
		x1_selected = x1[:, active_dims]  # [[1.0, 2.0]]
		x2_selected = x2[:, active_dims]  # [[1.5, 2.5]]
		expected = base_kernel(x1_selected, x2_selected)

		# Results should match
		assert jnp.allclose(result, expected)
		assert jnp.isfinite(result)

	@allure.title("ActiveDimsKernel with matrix inputs")
	@allure.description("Test ActiveDimsKernel with matrix inputs.")
	def test_with_matrix_inputs(self, sample_2d_data):
		base_kernel = SEKernel(length_scale=1.0)
		active_dims = jnp.array([1])

		# Expand sample data to more dimensions
		x1, x2 = sample_2d_data
		# Add extra dimensions
		x1_expanded = jnp.concatenate([x1, jnp.ones((x1.shape[0], 3))], axis=1)
		x2_expanded = jnp.concatenate([x2, jnp.ones((x2.shape[0], 3))], axis=1)

		kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

		result = kernel(x1_expanded, x2_expanded)

		# Should produce covariance matrix
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))

	@allure.title("ActiveDimsKernel with single dimension")
	@allure.description("Test ActiveDimsKernel with single active dimension.")
	def test_single_dimension(self):
		base_kernel = SEKernel(length_scale=1.0)
		active_dims = jnp.array([2])  # Only third dimension
		kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

		x1 = jnp.array([[1.0, 2.0, 3.0, 4.0]])
		x2 = jnp.array([[5.0, 6.0, 3.5, 8.0]])

		result = kernel(x1, x2)

		# Should only depend on dimension 2
		x1_dim2 = x1[:, 2:3]  # [[3.0]]
		x2_dim2 = x2[:, 2:3]  # [[3.5]]
		expected = base_kernel(x1_dim2, x2_dim2)

		assert jnp.allclose(result, expected)


class TestARDKernel:
	"""Tests for ARDKernel (Automatic Relevance Determination) wrapper."""

	@allure.title("ARDKernel Instantiation")
	@allure.description("Test that ARDKernel can be instantiated.")
	def test_instantiation(self):
		base_kernel = SEKernel(length_scale=1.0)
		length_scales = jnp.array([1.0, 2.0, 0.5])
		kernel = ARDKernel(base_kernel, length_scales=length_scales)

		assert kernel.inner_kernel is not None
		assert jnp.array_equal(kernel.length_scales, length_scales)

	@allure.title("ARDKernel different scales per dimension")
	@allure.description("Test that ARD applies different length scales per dimension.")
	def test_different_scales_per_dimension(self):
		base_kernel = SEKernel(length_scale=1.0)

		# Different relevance for each dimension
		length_scales = jnp.array([1.0, 0.1, 10.0])  # middle dim most relevant
		kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# Create inputs
		x1 = jnp.array([[0.0, 2.0, 1.5]])
		x2 = jnp.array([[-1.0, 1.0, 1.0]])

		result = kernel(x1, x2)

		# Manually compute ARD result
		scaled_x1 = x1 / length_scales
		scaled_x2 = x2 / length_scales
		base_kernel_unit = SEKernel(length_scale=1.0)
		expected = base_kernel_unit(scaled_x1, scaled_x2)

		assert jnp.allclose(result, expected, rtol=1e-5)
		assert jnp.isfinite(result)

	@allure.title("ARDKernel isotropic equivalence")
	@allure.description("Test that uniform length scales give isotropic kernel.")
	def test_isotropic_equivalence(self):
		base_kernel = SEKernel(length_scale=1.0)

		# All dimensions have same scale
		length_scales = jnp.array([2.0, 2.0, 2.0])
		ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# Compare with isotropic kernel with same scale
		iso_kernel = SEKernel(length_scale=2.0)

		x1 = jnp.array([[1.0, 2.0, 3.0]])
		x2 = jnp.array([[1.5, 2.5, 3.5]])

		ard_result = ard_kernel(x1, x2)
		iso_result = iso_kernel(x1, x2)

		# Should be approximately equal
		assert jnp.allclose(ard_result, iso_result, rtol=1e-5)

	@allure.title("ARDKernel with matrix inputs")
	@allure.description("Test ARDKernel with matrix inputs.")
	def test_matrix_inputs(self):
		base_kernel = SEKernel(length_scale=1.0)
		length_scales = jnp.array([1.0, 0.5, 2.0])
		kernel = ARDKernel(base_kernel, length_scales=length_scales)

		n_points = 5
		n_dims = 3
		x1 = jnp.linspace(0, 1, n_points * n_dims).reshape(n_points, n_dims)
		x2 = jnp.linspace(0.5, 1.5, n_points * n_dims).reshape(n_points, n_dims)

		result = kernel(x1, x2)

		assert result.shape == (n_points, n_points)
		assert jnp.all(jnp.isfinite(result))

	@allure.title("ARDKernel relevance interpretation")
	@allure.description("Test that smaller length scales indicate higher relevance.")
	def test_relevance_interpretation(self):
		base_kernel = SEKernel(length_scale=1.0)

		# First dimension very relevant (small scale), last less relevant (large scale)
		length_scales = jnp.array([0.1, 10.0])
		kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# Points differ only in first dimension
		x1 = jnp.array([[0.0, 0.0]])
		x2_first_dim = jnp.array([[1.0, 0.0]])  # Differ in first dim
		x2_second_dim = jnp.array([[0.0, 1.0]])  # Differ in second dim

		cov_first = kernel(x1, x2_first_dim)
		cov_second = kernel(x1, x2_second_dim)

		# Difference in first (relevant) dim should matter more
		# So covariance should be lower when first dim differs
		assert cov_first < cov_second


class TestWrapperCombinations:
	"""Test combinations of different wrapper kernels."""

	@allure.title("Wrapper combinations ARD with ActiveDims")
	@allure.description("Test combining ARD and ActiveDims wrappers.")
	def test_ard_with_active_dims(self):
		base_kernel = SEKernel(length_scale=1.0)

		# First, define ARD
		length_scales = jnp.array([1.0, 0.5, 2.0])  # Defined only on 3 dims, as we later use ARD!
		ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# ActiveDims must always be the outer-most kernel
		active_dims = jnp.array([0, 2, 4])
		active_kernel = ActiveDimsKernel(ard_kernel, active_dims=active_dims)

		# Create 5D inputs
		x1 = jnp.ones((5,))
		x2 = jnp.ones((5,)) * 1.5

		result = active_kernel(x1, x2)

		assert jnp.isfinite(result)
		assert result.shape == ()  # Scalar output

	@allure.title("Wrapper combinations Batch with ARD")
	@allure.description("Test combining Batch and ARD wrappers.")
	def test_batch_with_ard(self):
		base_kernel = SEKernel(length_scale=1.0)

		# Apply ARD first
		length_scales = jnp.array([1.0, 2.0])
		ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# Then batch
		batch_size = 3
		batch_kernel = BatchKernel(
			ard_kernel,
			batch_size=batch_size,
			batch_in_axes=None,  # Shared ARD scales
			batch_over_inputs=True,
		)

		x_batched = jnp.array([[[1.0, 2.0]], [[1.5, 2.5]], [[2.0, 3.0]]])

		result = batch_kernel(x_batched, x_batched)

		assert result.shape == (batch_size, 1, 1)
		assert jnp.all(jnp.isfinite(result))
