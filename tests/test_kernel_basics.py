"""
Tests for base kernel implementations.
"""
import allure
import jax
import jax.numpy as jnp
import pytest

import kernax
from kernax import (
	AffineKernel,
	ConstantKernel,
	LinearKernel,
	Matern12Kernel,
	Matern32Kernel,
	Matern52Kernel,
	PeriodicKernel,
	PolynomialKernel,
	RationalQuadraticKernel,
	RBFKernel,
	SEKernel,
	WhiteNoiseKernel,
)


class TestSEKernel:
	"""Tests for SE (Squared Exponential) Kernel."""

	@allure.title("SEKernel Instantiation")
	@allure.description("Test that SE kernel can be instantiated.")
	def test_instantiation(self):
		kernel = SEKernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

	@allure.title("SEKernel string representation")
	@allure.description("Test that SE kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = SEKernel(length_scale=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("SEKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = SEKernel(length_scale=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		x3 = jnp.array([3.0])
		result_1 = kernel(x1, x2)
		expected_1 = jnp.array(0.60653067)
		result_2 = kernel(x1, x3)
		expected_2 = jnp.array(0.13533528)
		assert result_1.shape == ()
		assert jnp.isfinite(result_1) and jnp.isfinite(result_2)
		assert jnp.allclose(kernel(x1, x2), kernel(x2, x3))
		assert jnp.allclose(result_1, expected_1) and jnp.allclose(result_2, expected_2)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0, 5.0])
	@allure.title("SEKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, length_scale):
		kernel = SEKernel(length_scale=length_scale)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert result[i, j] == kernel(x1[i], x2[j])

	@allure.title("SEKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel_1 = SEKernel(length_scale=1.0)
		kernel_2 = SEKernel(length_scale=2.0)
		x1, _ = sample_1d_data
		K_1 = kernel_1(x1)  # Test optional x2 parameter
		K_2 = kernel_2(x1)  # Test optional x2 parameter

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K_1) > 0)

		# Check matrix is symmetric
		assert jnp.allclose(K_1, K_1.T)

		# Check that K_2 has higher values than K_1 due to bigger length scale
		assert jnp.all(K_2 >= K_1)



class TestRBFKernel:
	"""Tests for RBF (Radial Basis Function) Kernel."""

	# As RBF is a copy of SE, we just test instanciation and equivalence

	@allure.title("SEKernel Instantiation")
	@allure.description("Test that SE kernel can be instantiated.")
	def test_instantiation(self):
		kernel = RBFKernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

	@allure.title("RBFKernel string representation")
	@allure.description("Test that RBF kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = RBFKernel(length_scale=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("RBFKernel comparison with SEKernel")
	@allure.description("Compare RBF kernel results against SEKernel implementation.")
	def test_against_SE(self, sample_1d_data, length_scale):
		RBF_kernel = RBFKernel(length_scale=length_scale)
		SE_kernel = SEKernel(length_scale=length_scale)

		x1, x2 = sample_1d_data
		result = RBF_kernel(x1, x2)
		expected = SE_kernel(x1, x2)

		assert jnp.allclose(result, expected)


class TestLinearKernel:
	"""Tests for Linear Kernel: k(x, x') = slope_var * x.T @ x'"""

	@allure.title("LinearKernel Instantiation")
	@allure.description("Test that Linear kernel can be instantiated with slope_var.")
	def test_instantiation(self):
		kernel = LinearKernel(slope_var=1.5)
		assert kernel.slope_var == 1.5

	@allure.title("LinearKernel string representation")
	@allure.description("Test that Linear kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = LinearKernel(slope_var=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("LinearKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = LinearKernel(slope_var=2.0)
		x1 = jnp.array([2.0])
		x2 = jnp.array([3.0])
		# k(x1, x2) = slope_var * x1.T @ x2 = 2.0 * 2.0 * 3.0 = 12.0
		result = kernel(x1, x2)
		expected = jnp.array(12.0)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("slope_var", [0.5, 1.0, 2.0])
	@allure.title("LinearKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, slope_var):
		kernel = LinearKernel(slope_var=slope_var)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@pytest.mark.parametrize("slope_var", [0.5, 1.0, 2.0])
	@allure.title("LinearKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data, slope_var):
		kernel = LinearKernel(slope_var=slope_var)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements are non-negative (k(x,x) = slope_var * x^2 >= 0)
		assert jnp.all(jnp.diag(K) >= 0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("LinearKernel passes through origin")
	@allure.description("Test that k(0, x) = 0 for any x, reflecting the GP always crossing (0, 0).")
	def test_passes_through_origin(self, sample_1d_data):
		kernel = LinearKernel(slope_var=1.0)
		_, x2 = sample_1d_data
		origin = jnp.array([0.0])
		# k(0, x) = slope_var * 0.T @ x = 0
		result = kernel(origin, x2)
		assert jnp.allclose(result, jnp.zeros_like(result))



class TestAffineKernel:
	"""Tests for Affine Kernel: k(x, x') = slope_var * (x - offset).T @ (x' - offset)"""

	@allure.title("AffineKernel Instantiation")
	@allure.description("Test that Affine kernel can be instantiated with slope_var and offset.")
	def test_instantiation(self):
		kernel = AffineKernel(slope_var=1.0, offset=2.0)
		assert kernel.slope_var == 1.0
		assert kernel.offset == 2.0

	@allure.title("AffineKernel string representation")
	@allure.description("Test that Affine kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = AffineKernel(slope_var=1.0, offset=2.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("AffineKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = AffineKernel(slope_var=2.0, offset=1.0)
		x1 = jnp.array([3.0])
		x2 = jnp.array([4.0])
		# k(x1, x2) = slope_var * (x1 - offset).T @ (x2 - offset)
		# = 2.0 * (3 - 1) * (4 - 1) = 2.0 * 2.0 * 3.0 = 12.0
		result = kernel(x1, x2)
		expected = jnp.array(12.0)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize(
		"slope_var,offset",
		[(0.5, 0.0), (1.0, 1.0), (2.0, -1.0), (1.0, 0.5)],
	)
	@allure.title("AffineKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, slope_var, offset):
		kernel = AffineKernel(slope_var=slope_var, offset=offset)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@pytest.mark.parametrize("slope_var,offset", [(0.5, 0.0), (1.0, 1.0), (2.0, -1.0)])
	@allure.title("AffineKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data, slope_var, offset):
		kernel = AffineKernel(slope_var=slope_var, offset=offset)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements are non-negative (k(x,x) = slope_var * (x-offset)^2 >= 0)
		assert jnp.all(jnp.diag(K) >= 0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("AffineKernel passes through offset")
	@allure.description("Test that k(offset, x) = 0 for any x, reflecting the GP always crossing (offset, 0).")
	def test_passes_through_offset(self, sample_1d_data):
		offset_val = 2.0
		kernel = AffineKernel(slope_var=1.0, offset=offset_val)
		_, x2 = sample_1d_data
		offset_point = jnp.array([offset_val])
		# k(offset, x) = slope_var * (offset - offset).T @ (x - offset) = 0
		result = kernel(offset_point, x2)
		assert jnp.allclose(result, jnp.zeros_like(result))

	@allure.title("AffineKernel with zero offset equals LinearKernel")
	@allure.description("Test that AffineKernel(offset=0) produces identical results to LinearKernel.")
	def test_zero_offset_equals_linear(self, sample_1d_data):
		slope_var = 1.5
		affine_kernel = AffineKernel(slope_var=slope_var, offset=0.0)
		linear_kernel = LinearKernel(slope_var=slope_var)
		x1, x2 = sample_1d_data
		assert jnp.allclose(affine_kernel(x1, x2), linear_kernel(x1, x2))



class TestMatern12Kernel:
	"""Tests for Matern 1/2 Kernel (Exponential)."""

	@allure.title("Matern12Kernel Instantiation")
	@allure.description("Test that Matern 1/2 kernel can be instantiated.")
	def test_instantiation(self):
		kernel = Matern12Kernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

	@allure.title("Matern12Kernel string representation")
	@allure.description("Test that Matern 1/2 kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = Matern12Kernel(length_scale=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("Matern12Kernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = Matern12Kernel(length_scale=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		# Matern 1/2: k(x1, x2) = exp(-r / l) where r = ||x1 - x2||
		# r = 1.0, l = 1.0 => exp(-1) = 0.36787944
		result = kernel(x1, x2)
		expected = jnp.array(0.36787944)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0, 5.0])
	@allure.title("Matern12Kernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, length_scale):
		kernel = Matern12Kernel(length_scale=length_scale)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@allure.title("Matern12Kernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = Matern12Kernel(length_scale=1.0)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements are 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check that higher length scale gives higher values
		kernel2 = Matern12Kernel(length_scale=2.0)
		K2 = kernel2(x1, x1)
		assert jnp.all(K2 >= K)



class TestMatern32Kernel:
	"""Tests for Matern 3/2 Kernel."""

	@allure.title("Matern32Kernel Instantiation")
	@allure.description("Test that Matern 3/2 kernel can be instantiated.")
	def test_instantiation(self):
		kernel = Matern32Kernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

	@allure.title("Matern32Kernel string representation")
	@allure.description("Test that Matern 3/2 kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = Matern32Kernel(length_scale=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("Matern32Kernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = Matern32Kernel(length_scale=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		# Matern 3/2: k(x1, x2) = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
		# r = 1.0, l = 1.0 => (1 + sqrt(3)) * exp(-sqrt(3)) = 0.69982314
		result = kernel(x1, x2)
		expected = jnp.array(0.4833578)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@allure.title("Matern32Kernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0, 5.0])
	def test_cross_cov_computation(self, sample_1d_data, length_scale):
		kernel = Matern32Kernel(length_scale=length_scale)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@allure.title("Matern32Kernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = Matern32Kernel(length_scale=1.0)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements are 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Test that higher length scale gives higher values
		kernel2 = Matern32Kernel(length_scale=2.0)
		K2 = kernel2(x1)  # Test optional x2 parameter
		assert jnp.all(K2 >= K)



class TestMatern52Kernel:
	"""Tests for Matern 5/2 Kernel."""

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern52Kernel Instantiation")
	@allure.description("Test that Matern 5/2 kernel can be instantiated.")
	def test_instantiation(self, length_scale):
		kernel = Matern52Kernel(length_scale=length_scale)
		assert kernel.length_scale == length_scale

	@allure.title("Matern52Kernel string representation")
	@allure.description("Test that Matern 5/2 kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = Matern52Kernel(length_scale=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("Matern52Kernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = Matern52Kernel(length_scale=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		# Matern 5/2: k(x1, x2) = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
		# r = 1.0, l = 1.0 => (1 + sqrt(5) + 5/3) * exp(-sqrt(5)) = 0.5239941
		result = kernel(x1, x2)
		expected = jnp.array(0.5239941)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern52Kernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, length_scale):
		kernel = Matern52Kernel(length_scale=length_scale)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@allure.title("Matern52Kernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = Matern52Kernel(length_scale=1.0)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements are 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check that bigger length scale gives higher values
		kernel2 = Matern52Kernel(length_scale=2.0)
		K2 = kernel2(x1)  # Test optional x2 parameter
		assert jnp.all(K2 >= K)



class TestPeriodicKernel:
	"""Tests for Periodic Kernel."""

	@allure.title("PeriodicKernel Instantiation")
	@allure.description("Test that Periodic kernel can be instantiated.")
	def test_instantiation(self):
		kernel = PeriodicKernel(length_scale=1.0, period=2.0)
		assert kernel.length_scale == 1.0
		assert kernel.period == 2.0

	@allure.title("PeriodicKernel string representation")
	@allure.description("Test that Periodic kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = PeriodicKernel(length_scale=1.0, period=2.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("PeriodicKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = PeriodicKernel(length_scale=1.0, period=2.0)
		x1 = jnp.array([0.5])
		x2 = jnp.array([1.5])
		result = kernel(x1, x2)
		expected = jnp.array(0.13533528)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@allure.title("PeriodicKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data):
		kernel = PeriodicKernel(length_scale=1.0, period=2.0)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@allure.title("PeriodicKernel periodicity property")
	@allure.description("Test that kernel respects periodicity.")
	def test_periodicity(self):
		period = 2.0
		kernel = PeriodicKernel(length_scale=1.0, period=period)
		x = jnp.array([0.5])
		y1 = jnp.array([1.5])
		y2 = jnp.array([1.5 + period])
		# Values separated by one period should have same covariance
		assert jnp.allclose(kernel(x, y1), kernel(x, y2), atol=1e-5)

	@allure.title("PeriodicKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = PeriodicKernel(length_scale=1.0, period=2.0)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements equal 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Test that higher length scale gives higher values
		kernel2 = PeriodicKernel(length_scale=2.0, period=2.0)
		K2 = kernel2(x1)  # Test optional x2 parameter
		assert jnp.all(K2 >= K)

	@allure.title("PeriodicKernel properties verification")
	@allure.description("Verify Periodic kernel computation with exact values.")
	def test_periodic_exact_values(self):
		# Note: GPJax Periodic kernel uses different formula/parameterization
		# Testing with a simple case where we can verify the computation
		kernel = PeriodicKernel(length_scale=1.0, period=2.0)

		# Testing with simple 1D inputs
		x1_simple = jnp.array([[0.5]])
		x2_simple = jnp.array([[1.5]])

		result_simple = kernel(x1_simple, x2_simple)

		# Our implementation: exp(-2 * sin^2(pi * dist / period) / length_scale^2)
		# dist = 1.0, period = 2.0, length_scale = 1.0
		# sin(pi/2) = 1.0, so exp(-2) = 0.13533528
		expected_value = jnp.array([[0.13533528]])

		assert jnp.allclose(result_simple, expected_value, atol=1e-5)


class TestRationalQuadraticKernel:
	"""Tests for Rational Quadratic Kernel."""

	@pytest.mark.parametrize(
		"length_scale,alpha",
		[
			(0.5, 0.5),
			(1.0, 1.0),
			(2.0, 2.0),
			(1.0, 0.5),
			(1.0, 2.0),
		],
	)
	@allure.title("RationalQuadraticKernel Instantiation")
	@allure.description("Test that RQ kernel can be instantiated.")
	def test_instantiation(self, length_scale, alpha):
		kernel = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		assert kernel.length_scale == length_scale
		assert kernel.alpha == alpha

	@allure.title("RationalQuadraticKernel string representation")
	@allure.description("Test that RQ kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = RationalQuadraticKernel(length_scale=1.0, alpha=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("RationalQuadraticKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = RationalQuadraticKernel(length_scale=1.0, alpha=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		result = kernel(x1, x2)
		expected = jnp.array(0.66666667)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@pytest.mark.parametrize(
		"length_scale,alpha",
		[
			(0.5, 1.0),
			(1.0, 0.5),
			(1.0, 1.0),
			(2.0, 2.0),
		],
	)
	@allure.title("RationalQuadraticKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, length_scale, alpha):
		kernel = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@allure.title("RationalQuadraticKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = RationalQuadraticKernel(length_scale=1.0, alpha=1.0)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements equal 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Test that higher length scale gives higher values
		kernel2 = RationalQuadraticKernel(length_scale=2.0, alpha=1.0)
		K2 = kernel2(x1)  # Test optional x2 parameter
		assert jnp.all(K2 >= K)



class TestPolynomialKernel:
	"""Tests for Polynomial Kernel."""

	@allure.title("PolynomialKernel Instantiation")
	@allure.description("Test that Polynomial kernel can be instantiated.")
	def test_instantiation(self):
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=0.0)
		assert kernel.degree == 2
		assert kernel.gamma == 1.0
		assert kernel.constant == 0.0

	@allure.title("PolynomialKernel string representation")
	@allure.description("Test that Polynomial kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=0.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("PolynomialKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=1.0)
		x1 = jnp.array([2.0])
		x2 = jnp.array([3.0])
		# Polynomial: k(x1, x2) = (gamma * x1^T x2 + constant)^degree
		# (1.0 * 2.0 * 3.0 + 1.0)^2 = (6.0 + 1.0)^2 = 49.0
		result = kernel(x1, x2)
		expected = jnp.array(49.0)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected)

	@allure.title("PolynomialKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data):
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=0.0)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@allure.title("PolynomialKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=1.0)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K) > 0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("PolynomialKernel degree variations")
	@allure.description("Test polynomial kernel with different degrees.")
	def test_degree_variations(self):
		x1 = jnp.array([2.0])
		x2 = jnp.array([3.0])

		# Test degree 1 (linear)
		kernel_deg1 = PolynomialKernel(degree=1, gamma=1.0, constant=0.0)
		result_deg1 = kernel_deg1(x1, x2)
		expected_deg1 = jnp.array(6.0)  # 2 * 3 = 6
		assert jnp.allclose(result_deg1, expected_deg1)

		# Test degree 3 (cubic)
		kernel_deg3 = PolynomialKernel(degree=3, gamma=1.0, constant=0.0)
		result_deg3 = kernel_deg3(x1, x2)
		expected_deg3 = jnp.array(216.0)  # (2 * 3)^3 = 6^3 = 216
		assert jnp.allclose(result_deg3, expected_deg3)



class TestConstantKernel:
	"""Tests for Constant Kernel."""

	@allure.title("ConstantKernel Instantiation")
	@allure.description("Test that Constant kernel can be instantiated.")
	def test_instantiation(self):
		kernel = ConstantKernel(value=2.0)
		assert kernel.value == 2.0

	@allure.title("ConstantKernel string representation")
	@allure.description("Test that Constant kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = ConstantKernel(value=2.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0

	@allure.title("ConstantKernel scalar computation")
	@allure.description("Test covariance computation returns constant value.")
	def test_scalar_computation(self):
		kernel = ConstantKernel(value=2.5)
		x1 = jnp.array([1.0])
		x2 = jnp.array([5.0])
		x3 = jnp.array([10.0])
		result_1 = kernel(x1, x2)
		result_2 = kernel(x1, x3)
		result_3 = kernel(x2, x3)
		expected = jnp.array(2.5)
		assert result_1.shape == ()
		assert jnp.allclose(result_1, expected)
		assert jnp.allclose(result_2, expected)
		assert jnp.allclose(result_3, expected)

	@allure.title("ConstantKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data):
		kernel = ConstantKernel(value=3.0)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		# All values should be equal to the constant
		assert jnp.allclose(result, 3.0)

	@allure.title("ConstantKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = ConstantKernel(value=1.5)
		x1, _ = sample_1d_data
		K = kernel(x1)  # Test optional x2 parameter

		# Check all elements are the constant value
		assert jnp.allclose(K, 1.5)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)



class TestWhiteNoiseKernel:
	"""Tests for WhiteNoiseKernel class."""

	# As WhiteNoiseKernel is just a shortcut to a Diag(Constant()) kernel, we only test instantiation and equivalence

	@allure.title("WhiteNoiseKernel Instantiation")
	@allure.description("Test that WhiteNoise kernel can be instantiated.")
	def test_instantiation(self):
		kernel = WhiteNoiseKernel(noise=1.0)
		assert kernel.noise == 1.0

	@allure.title("WhiteNoiseKernel string representation")
	@allure.description("Test that WhiteNoise kernel has a valid string representation.")
	def test_str_representation(self):
		kernel = WhiteNoiseKernel(noise=1.0)
		str_repr = str(kernel)
		assert isinstance(str_repr, str)
		assert len(str_repr) > 0


class TestNaNHandling:
	"""Tests for NaN-aware computations."""

	@allure.title("NaN handling in input")
	@allure.description("Test that NaN in input produces NaN in output.")
	def test_nan_in_input(self):
		kernel = SEKernel(length_scale=1.0)
		x1 = jnp.array([[1.0], [jnp.nan], [3.0]])  # Shape (3, 1) - 2D matrix
		x2 = jnp.array([[1.5], [2.5], [3.5]])  # Shape (3, 1)

		result = kernel(x1, x2)  # Shape (3, 3)

		# Row corresponding to NaN input should be NaN
		assert jnp.all(jnp.isnan(result[1, :]))

	@allure.title("NaN propagation in matrix")
	@allure.description("Test NaN propagation in matrix computations.")
	def test_nan_propagation_in_matrix(self):
		kernel = SEKernel(length_scale=1.0)

		# Create matrix with some NaN values
		x = jnp.array([[1.0], [jnp.nan], [3.0], [4.0]])

		result = kernel(x, x)

		# Row and column corresponding to NaN should be NaN
		assert jnp.all(jnp.isnan(result[1, :]))
		assert jnp.all(jnp.isnan(result[:, 1]))

		# Other elements should be finite - use JAX immutable assignment
		valid_mask = jnp.ones((4, 4), dtype=bool)
		valid_mask = valid_mask.at[1, :].set(False)
		valid_mask = valid_mask.at[:, 1].set(False)
		assert jnp.all(jnp.isfinite(result[valid_mask]))


class TestDimensionHandling:
	"""Tests for automatic dimension handling."""

	@allure.title("Dimension handling scalar to scalar")
	@allure.description("Test scalar x scalar -> scalar.")
	def test_scalar_to_scalar(self):
		kernel = SEKernel(length_scale=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		result = kernel(x1, x2)
		assert result.shape == ()

	@allure.title("Dimension handling vector to scalar")
	@allure.description("Test vector x scalar -> vector.")
	def test_vector_to_scalar(self):
		kernel = SEKernel(length_scale=1.0)
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([2.0])
		result = kernel(x1, x2)
		assert result.shape == (3,)

	@allure.title("Dimension handling vector to vector")
	@allure.description("Test vector x vector -> matrix.")
	def test_vector_to_vector(self):
		kernel = SEKernel(length_scale=1.0)
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([[1.5], [2.5]])
		result = kernel(x1, x2)
		assert result.shape == (3, 2)


@pytest.fixture
def sample_1d_data():
	"""Generate sample 1D data for testing."""
	key = jax.random.PRNGKey(42)
	x1 = jax.random.uniform(key, shape=(10, 1))
	x2 = jax.random.uniform(key, shape=(10, 1))
	return x1, x2


class TestSigmoidKernel:
	"""Test suite for SigmoidKernel."""

	@allure.title("SigmoidKernel instantiation")
	@allure.description("Test that SigmoidKernel can be instantiated with valid parameters.")
	def test_instantiation(self):
		"""Test basic kernel instantiation."""
		# Test with default parameters
		kernel = kernax.SigmoidKernel(alpha=1., constant=0.)
		assert kernel.alpha == 1.0
		assert kernel.constant == 0.0

		# Test with custom parameters
		kernel = kernax.SigmoidKernel(alpha=0.5, constant=1.0)
		assert kernel.alpha == 0.5
		assert kernel.constant == 1.0

		# Test with different values
		kernel = kernax.SigmoidKernel(alpha=2.0, constant=-1.5)
		assert kernel.alpha == 2.0
		assert kernel.constant == -1.5

	@allure.title("SigmoidKernel parameter validation")
	@allure.description("Test that SigmoidKernel rejects invalid parameters.")
	def test_parameter_validation(self):
		"""Test that invalid parameters are rejected."""
		# Alpha must be positive
		with pytest.raises(ValueError):
			kernax.SigmoidKernel(alpha=0.0, constant=0.)

		with pytest.raises(ValueError):
			kernax.SigmoidKernel(alpha=-1.0, constant=0.)

		# Constant can be any value (no error expected)
		kernel = kernax.SigmoidKernel(alpha=1.0, constant=-10.0)
		assert kernel.constant == -10.0

	@allure.title("SigmoidKernel string representation")
	@allure.description("Test that the kernel has a readable string representation.")
	def test_str_representation(self):
		"""Test string representation of the kernel."""
		kernel = kernax.SigmoidKernel(alpha=1.0, constant=0.0)
		str_repr = str(kernel)
		assert "SigmoidKernel" in str_repr

	@allure.title("SigmoidKernel scalar computation")
	@allure.description("Test kernel computation between two scalar inputs.")
	def test_scalar_computation(self):
		"""Test kernel computation on scalar inputs."""
		kernel = kernax.SigmoidKernel(alpha=1.0, constant=0.0)

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = kernel(x1, x2)

		# Check properties
		assert jnp.isfinite(result)
		assert result.shape == ()
		# Sigmoid kernel output is in range (-1, 1) due to tanh
		assert -1.0 <= result <= 1.0

	@pytest.mark.parametrize(
		"alpha,constant",
		[
			(1.0, 0.0),
			(0.5, 1.0),
			(2.0, -0.5),
		],
	)
	@allure.title("SigmoidKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, alpha, constant):
		"""Test cross-covariance matrix computation."""
		kernel = kernax.SigmoidKernel(alpha=alpha, constant=constant)
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)

		# Check shape
		assert result.shape == (x1.shape[0], x2.shape[0])

		# Check all values are finite
		assert jnp.all(jnp.isfinite(result))

		# Check values are in valid range for tanh
		assert jnp.all(result >= -1.0)
		assert jnp.all(result <= 1.0)

		# Check consistency with pairwise computations
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@pytest.mark.parametrize("alpha,constant", [(1.0, 0.0), (0.5, 1.0), (2.0, -1.0)])
	@allure.title("SigmoidKernel mathematical properties")
	@allure.description("Test mathematical properties of the sigmoid kernel.")
	def test_math_properties(self, sample_1d_data, alpha, constant):
		"""Test mathematical properties."""
		kernel = kernax.SigmoidKernel(alpha=alpha, constant=constant)
		x1, _ = sample_1d_data

		K = kernel(x1)  # Test optional x2 parameter

		# Check shape
		assert K.shape == (x1.shape[0], x1.shape[0])

		# Check all values are in range
		assert jnp.all(K >= -1.0)
		assert jnp.all(K <= 1.0)

		# Sigmoid kernel is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("SigmoidKernel output range")
	@allure.description("Verify that sigmoid kernel output is always in (-1, 1) range.")
	def test_output_range(self, sample_1d_data):
		"""Test that output is always in valid range."""
		x1, x2 = sample_1d_data

		# Test with various extreme parameters
		for alpha, constant in [(0.01, -10.0), (10.0, 10.0), (0.1, 0.0)]:
			kernel = kernax.SigmoidKernel(alpha=alpha, constant=constant)
			K = kernel(x1, x2)

			# tanh outputs in [-1, 1] (can equal -1 or 1 with extreme inputs)
			assert jnp.all(K >= -1.0)
			assert jnp.all(K <= 1.0)
			assert jnp.all(jnp.isfinite(K))

	@allure.title("SigmoidKernel edge cases")
	@allure.description("Test kernel behavior with edge case inputs.")
	def test_edge_cases(self):
		"""Test edge cases."""
		kernel = kernax.SigmoidKernel(alpha=1.0, constant=0.0)

		# Test with zero vectors
		x_zero = jnp.array([0.0])
		result = kernel(x_zero, x_zero)
		# tanh(1.0 * 0 + 0) = tanh(0) = 0
		assert jnp.allclose(result, 0.0)

		# Test with identical vectors
		x = jnp.array([5.0])
		result = kernel(x, x)
		# tanh(1.0 * 25 + 0) = tanh(25) ≈ 1.0
		assert jnp.isfinite(result)
		assert -1.0 <= result <= 1.0

	@allure.title("SigmoidKernel NaN handling")
	@allure.description("Test that the kernel properly handles NaN inputs.")
	def test_nan_handling(self):
		"""Test NaN handling."""
		kernel = kernax.SigmoidKernel(alpha=1.0, constant=0.0)

		x1 = jnp.array([1.0])
		x_nan = jnp.array([jnp.nan])

		# Kernel should return NaN when input contains NaN
		result = kernel(x1, x_nan)
		assert jnp.isnan(result)

		result = kernel(x_nan, x1)
		assert jnp.isnan(result)
