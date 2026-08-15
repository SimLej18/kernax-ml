"""
Tests for wrapper kernels (BatchModule, ActiveDimsModule, ARDKernel).
"""

import allure
import jax.numpy as jnp
import pytest

from kernax import (
	ActiveDimsModule,
	ARDKernel,
	BatchModule,
	InputSpecificParamModule,
	SEKernel,
	WhiteNoiseKernel,
)
from kernax.mask import create_mask
from kernax.parametrisations import NonTrainableParametrisation


class TestBatchModule:
	"""Tests for BatchModule wrapper."""

	@allure.title("BatchModule Instantiation")
	@allure.description("Test that BatchModule can be instantiated.")
	def test_instantiation(self):
		base_kernel = SEKernel(length_scale=1.0)
		batch_kernel = BatchModule(
			base_kernel, batch_size=5, batch_in_axes=0, batch_over_inputs=True
		)
		assert batch_kernel.inner is not None
		assert batch_kernel.batch_over_inputs == 0

	@allure.title("BatchModule batch over hyperparameters")
	@allure.description("Test batching with distinct hyperparameters per batch element.")
	def test_batch_over_hyperparameters(self):
		# Create base kernel with single length_scale
		base_kernel = SEKernel(length_scale=1.0)
		batch_size = 3

		# Wrap in BatchModule to handle batched hyperparameters
		batch_kernel = BatchModule(
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

	@allure.title("BatchModule batch over inputs and hyperparameters")
	@allure.description("Test batching over both inputs and hyperparameters.")
	def test_batch_over_inputs_and_hyperparameters(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x1_batched, x2_batched = sample_batched_data
		batch_size = x1_batched.shape[0]

		batch_kernel = BatchModule(
			base_kernel, batch_size=batch_size, batch_in_axes=0, batch_over_inputs=True
		)

		result = batch_kernel(x1_batched, x1_batched)

		# Should produce batch_size covariance matrices
		assert result.shape == (batch_size, x1_batched.shape[1], x1_batched.shape[1])
		assert jnp.all(jnp.isfinite(result))

		# Each batch element should be symmetric
		for i in range(batch_size):
			assert jnp.allclose(result[i], result[i].T)

	@allure.title("BatchModule batch over inputs only")
	@allure.description("Test batching over inputs with shared hyperparameters.")
	def test_batch_over_inputs_only(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		batch_size = x_batched.shape[0]

		# Batch over inputs but share hyperparameters
		batch_kernel = BatchModule(
			base_kernel,
			batch_size=batch_size,
			batch_in_axes=None,  # Shared hyperparameters
			batch_over_inputs=True,
		)

		result = batch_kernel(x_batched, x_batched)

		assert result.shape == (batch_size, x_batched.shape[1], x_batched.shape[1])
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BatchModule with shared hyperparameters and shared inputs")
	@allure.description(
		"Test BatchModule with batch_in_axes=None and batch_over_inputs=False. "
		"All batch matrices should be identical since same HPs and inputs are used."
	)
	def test_shared_hyperparameters_shared_inputs(self):
		base_kernel = SEKernel(length_scale=1.0)
		batch_size = 4

		# Shared hyperparameters AND shared inputs
		batch_kernel = BatchModule(
			base_kernel,
			batch_size=batch_size,
			batch_in_axes=None,  # Shared hyperparameters
			batch_over_inputs=False,  # Shared inputs
		)

		# Non-batched inputs
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5]])

		result = batch_kernel(x1, x2)

		# Result should have batch dimension
		assert result.shape == (batch_size, x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))

		# All batch matrices should be identical (same HPs + same inputs)
		expected_matrix = base_kernel(x1, x2)
		for i in range(batch_size):
			assert jnp.allclose(
				result[i], expected_matrix, rtol=1e-6
			), f"Batch {i} differs from expected"

		# Verify that all batch matrices are identical to each other
		for i in range(1, batch_size):
			assert jnp.allclose(
				result[i], result[0], rtol=1e-6
			), f"Batch {i} differs from batch 0"


	@allure.title("BatchModule double batch with masks")
	@allure.description(
		"Test doubly-nested BatchModule where the inner batch (size 2) varies length_scale "
		"and the outer batch (size 4) varies noise, using create_mask at each level. "
		"Result shape must be (4, 2, 3, 3). Diagonal entries equal 1 + noise; "
		"off-diagonal entries are noise-independent and match SE covariance."
	)
	def test_double_batch_with_masks(self):
		inputs = jnp.array([1., 2., 3.])[:, None]

		k = SEKernel(.5) + WhiteNoiseKernel(.3)
		bk = BatchModule(k, 2, create_mask(k, default=None, length_scale=0), False)
		bbk = BatchModule(bk, 4, create_mask(bk, default=None, noise=0), False)

		noise_vals = jnp.array([0., 1., 2., 3.])
		ls_vals = jnp.array([.5, 1.5])

		bbk = bbk.replace(noise=noise_vals)
		bbk = bbk.replace(length_scale=ls_vals)

		result = bbk(inputs)

		# shape: (outer_batch=4, inner_batch=2, n=3, n=3)
		assert result.shape == (4, 2, 3, 3)
		assert jnp.all(jnp.isfinite(result))

		# reference values from known output
		expected = jnp.array([[[[1.0000000e+00, 1.3533528e-01, 3.3546262e-04],
		                         [1.3533528e-01, 1.0000000e+00, 1.3533528e-01],
		                         [3.3546262e-04, 1.3533528e-01, 1.0000000e+00]],
		                        [[1.0000000e+00, 8.0073738e-01, 4.1111228e-01],
		                         [8.0073738e-01, 1.0000000e+00, 8.0073738e-01],
		                         [4.1111228e-01, 8.0073738e-01, 1.0000000e+00]]],
		                       [[[2.0000000e+00, 1.3533528e-01, 3.3546262e-04],
		                         [1.3533528e-01, 2.0000000e+00, 1.3533528e-01],
		                         [3.3546262e-04, 1.3533528e-01, 2.0000000e+00]],
		                        [[2.0000000e+00, 8.0073738e-01, 4.1111228e-01],
		                         [8.0073738e-01, 2.0000000e+00, 8.0073738e-01],
		                         [4.1111228e-01, 8.0073738e-01, 2.0000000e+00]]],
		                       [[[3.0000000e+00, 1.3533528e-01, 3.3546262e-04],
		                         [1.3533528e-01, 3.0000000e+00, 1.3533528e-01],
		                         [3.3546262e-04, 1.3533528e-01, 3.0000000e+00]],
		                        [[3.0000000e+00, 8.0073738e-01, 4.1111228e-01],
		                         [8.0073738e-01, 3.0000000e+00, 8.0073738e-01],
		                         [4.1111228e-01, 8.0073738e-01, 3.0000000e+00]]],
		                       [[[4.0000000e+00, 1.3533528e-01, 3.3546262e-04],
		                         [1.3533528e-01, 4.0000000e+00, 1.3533528e-01],
		                         [3.3546262e-04, 1.3533528e-01, 4.0000000e+00]],
		                        [[4.0000000e+00, 8.0073738e-01, 4.1111228e-01],
		                         [8.0073738e-01, 4.0000000e+00, 8.0073738e-01],
		                         [4.1111228e-01, 8.0073738e-01, 4.0000000e+00]]]])
		assert jnp.allclose(result, expected, rtol=1e-5)


class TestActiveDimsModule:
	"""Tests for ActiveDimsModule wrapper."""

	@allure.title("ActiveDimsModule Instantiation")
	@allure.description("Test that ActiveDimsModule can be instantiated.")
	def test_instantiation(self):
		base_kernel = SEKernel(length_scale=1.0)
		active_dims = jnp.array([0, 2])
		kernel = ActiveDimsModule(base_kernel, active_dims=active_dims)

		assert kernel.inner is not None
		assert jnp.array_equal(kernel.active_dims, active_dims)

	@allure.title("ActiveDimsModule dimension selection")
	@allure.description("Test that kernel only uses specified dimensions.")
	def test_dimension_selection(self):
		base_kernel = SEKernel(length_scale=1.0)

		# Only use first and third dimensions
		active_dims = jnp.array([0, 2])
		kernel = ActiveDimsModule(base_kernel, active_dims=active_dims)

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

	@allure.title("ActiveDimsModule with matrix inputs")
	@allure.description("Test ActiveDimsModule with matrix inputs.")
	def test_with_matrix_inputs(self, sample_2d_data):
		base_kernel = SEKernel(length_scale=1.0)
		active_dims = jnp.array([1])

		# Expand sample data to more dimensions
		x1, x2 = sample_2d_data
		# Add extra dimensions
		x1_expanded = jnp.concatenate([x1, jnp.ones((x1.shape[0], 3))], axis=1)
		x2_expanded = jnp.concatenate([x2, jnp.ones((x2.shape[0], 3))], axis=1)

		kernel = ActiveDimsModule(base_kernel, active_dims=active_dims)

		result = kernel(x1_expanded, x2_expanded)

		# Should produce covariance matrix
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))

	@allure.title("ActiveDimsModule with single dimension")
	@allure.description("Test ActiveDimsModule with single active dimension.")
	def test_single_dimension(self):
		base_kernel = SEKernel(length_scale=1.0)
		active_dims = jnp.array([2])  # Only third dimension
		kernel = ActiveDimsModule(base_kernel, active_dims=active_dims)

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

		assert kernel.inner is not None
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
		base_kernel = SEKernel(length_scale=1.0,
		                       length_scale_parametrisation=NonTrainableParametrisation())

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
		base_kernel = SEKernel(length_scale=1.0,
		                       length_scale_parametrisation=NonTrainableParametrisation())
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
		base_kernel = SEKernel(length_scale=1.0,
		                       length_scale_parametrisation=NonTrainableParametrisation())

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
		base_kernel = SEKernel(length_scale=1.0,
		                       length_scale_parametrisation=NonTrainableParametrisation())

		# First, define ARD
		length_scales = jnp.array([1.0, 0.5, 2.0])  # Defined only on 3 dims, as we later use ARD!
		ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# ActiveDims must always be the outer-most kernel
		active_dims = jnp.array([0, 2, 4])
		active_kernel = ActiveDimsModule(ard_kernel, active_dims=active_dims)

		# Create 5D inputs
		x1 = jnp.ones((5,))
		x2 = jnp.ones((5,)) * 1.5

		result = active_kernel(x1, x2)

		assert jnp.isfinite(result)
		assert result.shape == ()  # Scalar output

	@allure.title("Wrapper combinations Batch with ARD")
	@allure.description("Test combining Batch and ARD wrappers.")
	def test_batch_with_ard(self):
		base_kernel = SEKernel(length_scale=1.0,
		                       length_scale_parametrisation=NonTrainableParametrisation())

		# Apply ARD first
		length_scales = jnp.array([1.0, 2.0])
		ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

		# Then batch
		batch_size = 3
		batch_kernel = BatchModule(
			ard_kernel,
			batch_size=batch_size,
			batch_in_axes=None,  # Shared ARD scales
			batch_over_inputs=True,
		)

		x_batched = jnp.array([[[1.0, 2.0]], [[1.5, 2.5]], [[2.0, 3.0]]])

		result = batch_kernel(x_batched, x_batched)

		assert result.shape == (batch_size, 1, 1)
		assert jnp.all(jnp.isfinite(result))


class TestInputSpecificParamModule:
	"""Tests for InputSpecificParamModule wrapper."""

	@allure.title("InputSpecificParamModule output equals per-input noise on diagonal")
	@allure.description(
		"Test that wrapping WhiteNoiseKernel with InputSpecificParamModule and setting "
		"per-input noise values produces a diagonal matrix whose entries match those values."
	)
	@pytest.mark.parametrize("base_noise,extra", [(1., jnp.arange(3) + 1), (0.5, jnp.zeros(3))])
	def test_whitenoise_per_input_diagonal(self, base_noise, extra):
		x = jnp.array([1., 2., 3.])[:, None]
		k = InputSpecificParamModule(WhiteNoiseKernel(base_noise), input_size=len(x), vmap_in_axes=0)
		noise_values = k.inner.noise + extra
		k = k.replace(noise=noise_values)
		result = k(x)
		expected = jnp.diag(noise_values)
		assert jnp.allclose(result, expected), (
			f"Expected diagonal {noise_values}, got {jnp.diag(result)}"
		)
