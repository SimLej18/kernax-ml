"""
Tests for base kernel implementations.
"""

import jax.numpy as jnp
import pytest
import allure

from kernax import (
	ConstantKernel,
	DiagKernel,
	LinearKernel,
	Matern12Kernel,
	Matern32Kernel,
	Matern52Kernel,
	PeriodicKernel,
	PolynomialKernel,
	RationalQuadraticKernel,
	SEKernel,
	RBFKernel,
	WhiteNoiseKernel,
)


class TestSEKernel:
	"""Tests for SE (Squared Exponential) Kernel."""

	@allure.title("SEKernel Instantiation")
	@allure.description("Test that SE kernel can be instantiated.")
	def test_instantiation(self):
		kernel = SEKernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

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
		K_1 = kernel_1(x1, x1)
		K_2 = kernel_2(x1, x1)

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K_1) > 0)

		# Check matrix is symmetric
		assert jnp.allclose(K_1, K_1.T)

		# Check that K_2 has higher values than K_1 due to bigger length scale
		assert jnp.all(K_2 >= K_1)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("SEKernel comparison with scikit-learn")
	@allure.description("Compare SE kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data, length_scale):
		from sklearn.gaussian_process.kernels import RBF

		kernel = SEKernel(length_scale=length_scale)
		sklearn_kernel = RBF(length_scale=length_scale)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("SEKernel comparison with GPyTorch")
	@allure.description("Compare SE kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data, length_scale):
		import torch
		from gpytorch.kernels import RBFKernel as GPyTorchRBFKernel

		kernel = SEKernel(length_scale=length_scale)
		gpytorch_kernel = GPyTorchRBFKernel()
		gpytorch_kernel._set_lengthscale(length_scale)

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("SEKernel comparison with GPJax")
	@allure.description("Compare SE kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data, length_scale):
		from gpjax.kernels import RBF

		kernel = SEKernel(length_scale=length_scale)
		gpjax_kernel = RBF(lengthscale=length_scale)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestRBFKernel:
	"""Tests for RBF (Radial Basis Function) Kernel."""
	# As RBF is a copy of SE, we just test instanciation and equivalence

	@allure.title("SEKernel Instantiation")
	@allure.description("Test that SE kernel can be instantiated.")
	def test_instantiation(self):
		kernel = RBFKernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

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
	"""Tests for Linear Kernel."""

	@allure.title("LinearKernel Instantiation")
	@allure.description("Test that Linear kernel can be instantiated.")
	def test_instantiation(self):
		kernel = LinearKernel(variance_b=0.5, variance_v=1.0, offset_c=0.0)
		assert kernel.variance_b == 0.5
		assert kernel.variance_v == 1.0
		assert kernel.offset_c == 0.0

	@allure.title("LinearKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = LinearKernel(variance_b=0.5, variance_v=1.0, offset_c=0.0)
		x1 = jnp.array([2.0])
		x2 = jnp.array([3.0])
		# k(x1, x2) = variance_b + variance_v * (x1 - offset_c) * (x2 - offset_c)
		# k(2, 3) = 0.5 + 1.0 * (2 - 0) * (3 - 0) = 0.5 + 6 = 6.5
		result = kernel(x1, x2)
		expected = jnp.array(6.5)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("variance_b,variance_v,offset_c", [
		(0.0, 1.0, 0.0),
		(0.5, 1.0, 0.0),
		(1.0, 0.5, 0.0),
		(0.5, 1.0, 0.5),
	])
	@allure.title("LinearKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data, variance_b, variance_v, offset_c):
		kernel = LinearKernel(variance_b=variance_b, variance_v=variance_v, offset_c=offset_c)
		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				assert jnp.allclose(result[i, j], kernel(x1[i], x2[j]))

	@pytest.mark.parametrize("variance_b,variance_v,offset_c", [(0.5, 1.0, 0.), (1.0, 0.5, 2.), (1.0, 1.0, -3.)])
	@allure.title("LinearKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data, variance_b, variance_v, offset_c):
		kernel = LinearKernel(variance_b=variance_b, variance_v=variance_v, offset_c=offset_c)
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K) > 0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("LinearKernel comparison with scikit-learn")
	@allure.description("Compare Linear kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data):
		from sklearn.gaussian_process.kernels import DotProduct

		# DotProduct in sklearn is simpler: k(x, y) = x^T y (no variance_b or offset_c)
		# We'll use a kernel with no offset for comparison
		kernel = LinearKernel(variance_b=0.0, variance_v=1.0, offset_c=0.0)
		sklearn_kernel = DotProduct(sigma_0=0.0)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("LinearKernel comparison with GPJax")
	@allure.description("Compare Linear kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data):
		from gpjax.kernels import Linear

		# GPJax Linear kernel is simpler
		kernel = LinearKernel(variance_b=0.0, variance_v=1.0, offset_c=0.0)
		gpjax_kernel = Linear()

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)

class TestMatern12Kernel:
	"""Tests for Matern 1/2 Kernel (Exponential)."""

	@allure.title("Matern12Kernel Instantiation")
	@allure.description("Test that Matern 1/2 kernel can be instantiated.")
	def test_instantiation(self):
		kernel = Matern12Kernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

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
		kernel = Matern12Kernel(length_scale=1.)
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check diagonal elements are 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check that higher length scale gives higher values
		kernel2 = Matern12Kernel(length_scale=2.)
		K2 = kernel2(x1, x1)
		assert jnp.all(K2 >= K)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern12Kernel comparison with scikit-learn")
	@allure.description("Compare Matern 1/2 kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data, length_scale):
		from sklearn.gaussian_process.kernels import Matern

		kernel = Matern12Kernel(length_scale=length_scale)
		sklearn_kernel = Matern(length_scale=length_scale, nu=0.5)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern12Kernel comparison with GPyTorch")
	@allure.description("Compare Matern 1/2 kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data, length_scale):
		import torch
		from gpytorch.kernels import MaternKernel

		kernel = Matern12Kernel(length_scale=length_scale)
		gpytorch_kernel = MaternKernel(nu=0.5)
		gpytorch_kernel._set_lengthscale(length_scale)

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern12Kernel comparison with GPJax")
	@allure.description("Compare Matern 1/2 kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data, length_scale):
		from gpjax.kernels import Matern12

		kernel = Matern12Kernel(length_scale=length_scale)
		gpjax_kernel = Matern12(lengthscale=length_scale)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestMatern32Kernel:
	"""Tests for Matern 3/2 Kernel."""

	@allure.title("Matern32Kernel Instantiation")
	@allure.description("Test that Matern 3/2 kernel can be instantiated.")
	def test_instantiation(self):
		kernel = Matern32Kernel(length_scale=1.0)
		assert kernel.length_scale == 1.0

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
		kernel = Matern32Kernel(length_scale=1.)
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check diagonal elements are 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Test that higher length scale gives higher values
		kernel2 = Matern32Kernel(length_scale=2.)
		K2 = kernel2(x1, x1)
		assert jnp.all(K2 >= K)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern32Kernel comparison with scikit-learn")
	@allure.description("Compare Matern 3/2 kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data, length_scale):
		from sklearn.gaussian_process.kernels import Matern

		kernel = Matern32Kernel(length_scale=length_scale)
		sklearn_kernel = Matern(length_scale=length_scale, nu=1.5)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern32Kernel comparison with GPyTorch")
	@allure.description("Compare Matern 3/2 kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data, length_scale):
		import torch
		from gpytorch.kernels import MaternKernel

		kernel = Matern32Kernel(length_scale=length_scale)
		gpytorch_kernel = MaternKernel(nu=1.5)
		gpytorch_kernel._set_lengthscale(length_scale)

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern32Kernel comparison with GPJax")
	@allure.description("Compare Matern 3/2 kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data, length_scale):
		from gpjax.kernels import Matern32

		kernel = Matern32Kernel(length_scale=length_scale)
		gpjax_kernel = Matern32(lengthscale=length_scale)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestMatern52Kernel:
	"""Tests for Matern 5/2 Kernel."""

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern52Kernel Instantiation")
	@allure.description("Test that Matern 5/2 kernel can be instantiated.")
	def test_instantiation(self, length_scale):
		kernel = Matern52Kernel(length_scale=length_scale)
		assert kernel.length_scale == length_scale

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
		K = kernel(x1, x1)

		# Check diagonal elements are 1 (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check that bigger length scale gives higher values
		kernel2 = Matern52Kernel(length_scale=2.0)
		K2 = kernel2(x1, x1)
		assert jnp.all(K2 >= K)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern52Kernel comparison with scikit-learn")
	@allure.description("Compare Matern 5/2 kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data, length_scale):
		from sklearn.gaussian_process.kernels import Matern

		kernel = Matern52Kernel(length_scale=length_scale)
		sklearn_kernel = Matern(length_scale=length_scale, nu=2.5)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern52Kernel comparison with GPyTorch")
	@allure.description("Compare Matern 5/2 kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data, length_scale):
		import torch
		from gpytorch.kernels import MaternKernel

		kernel = Matern52Kernel(length_scale=length_scale)
		gpytorch_kernel = MaternKernel(nu=2.5)
		gpytorch_kernel._set_lengthscale(length_scale)

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
	@allure.title("Matern52Kernel comparison with GPJax")
	@allure.description("Compare Matern 5/2 kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data, length_scale):
		from gpjax.kernels import Matern52

		kernel = Matern52Kernel(length_scale=length_scale)
		gpjax_kernel = Matern52(lengthscale=length_scale)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestPeriodicKernel:
	"""Tests for Periodic Kernel."""

	@allure.title("PeriodicKernel Instantiation")
	@allure.description("Test that Periodic kernel can be instantiated.")
	def test_instantiation(self):
		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
		assert kernel.length_scale == 1.0
		assert kernel.variance == 1.0
		assert kernel.period == 2.0

	@allure.title("PeriodicKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
		x1 = jnp.array([0.5])
		x2 = jnp.array([1.5])
		# Periodic: k(x1, x2) = variance * exp(-2 * sin^2(pi * ||x1 - x2|| / period) / length_scale^2)
		# dist = 1.0, period = 2.0, length_scale = 1.0, variance = 1.0
		# sin(pi * 1.0 / 2.0) = sin(pi/2) = 1.0
		# exp(-2 * 1.0^2 / 1.0^2) = exp(-2) = 0.13533528
		result = kernel(x1, x2)
		expected = jnp.array(0.13533528)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@allure.title("PeriodicKernel cross-cov computation")
	@allure.description("Test cross-covariance computation between two batches of vectors.")
	def test_cross_cov_computation(self, sample_1d_data):
		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
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
		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=period)
		x = jnp.array([0.5])
		y1 = jnp.array([1.5])
		y2 = jnp.array([1.5 + period])
		# Values separated by one period should have same covariance
		assert jnp.allclose(kernel(x, y1), kernel(x, y2), atol=1e-5)

	@allure.title("PeriodicKernel mathematical properties")
	@allure.description("Test that mathematical properties of the kernel still hold.")
	def test_math_properties(self, sample_1d_data):
		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check diagonal elements equal variance (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Test that higher length scale gives higher values
		kernel2 = PeriodicKernel(length_scale=2.0, variance=1.0, period=2.0)
		K2 = kernel2(x1, x1)
		assert jnp.all(K2 >= K)

	@allure.title("PeriodicKernel comparison with scikit-learn")
	@allure.description("Compare Periodic kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data):
		from sklearn.gaussian_process.kernels import ExpSineSquared

		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
		# ExpSineSquared uses periodicity parameter instead of period
		sklearn_kernel = ExpSineSquared(length_scale=1.0, periodicity=2.0)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("PeriodicKernel comparison with GPyTorch")
	@allure.description("Compare Periodic kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data):
		import torch
		from gpytorch.kernels import PeriodicKernel as GPyTorchPeriodicKernel

		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
		gpytorch_kernel = GPyTorchPeriodicKernel()
		gpytorch_kernel._set_lengthscale(1.0)
		gpytorch_kernel.period_length = torch.tensor([2.0])

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@allure.title("PeriodicKernel properties verification")
	@allure.description("Verify Periodic kernel computation with exact values.")
	def test_periodic_exact_values(self):
		# Note: GPJax Periodic kernel uses different formula/parameterization
		# Testing with a simple case where we can verify the computation
		kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)

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

	@pytest.mark.parametrize("length_scale,alpha", [
		(0.5, 0.5),
		(1.0, 1.0),
		(2.0, 2.0),
		(1.0, 0.5),
		(1.0, 2.0),
	])
	@allure.title("RationalQuadraticKernel Instantiation")
	@allure.description("Test that RQ kernel can be instantiated.")
	def test_instantiation(self, length_scale, alpha):
		kernel = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		assert kernel.length_scale == length_scale
		assert kernel.alpha == alpha

	@allure.title("RationalQuadraticKernel scalar computation")
	@allure.description("Test covariance computation between two 1D vectors.")
	def test_scalar_computation(self):
		kernel = RationalQuadraticKernel(length_scale=1.0, alpha=1.0)
		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])
		# RQ: k(x1, x2) = variance * (1 + squared_dist / (2 * alpha * length_scale^2))^(-alpha)
		# squared_dist = 1.0, alpha = 1.0, length_scale = 1.0, variance = 1.0
		# (1 + 1.0 / (2 * 1.0 * 1.0))^(-1.0) = (1.5)^(-1.0) = 0.66666667
		result = kernel(x1, x2)
		expected = jnp.array(0.66666667)
		assert result.shape == ()
		assert jnp.isfinite(result)
		assert jnp.allclose(result, expected, atol=1e-5)

	@pytest.mark.parametrize("length_scale,alpha", [
		(0.5, 1.0),
		(1.0, 0.5),
		(1.0, 1.0),
		(2.0, 2.0),
	])
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
		K = kernel(x1, x1)

		# Check diagonal elements equal variance (same point)
		assert jnp.allclose(jnp.diag(K), 1.0)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Test that higher length scale gives higher values
		kernel2 = RationalQuadraticKernel(length_scale=2.0, alpha=1.0)
		K2 = kernel2(x1, x1)
		assert jnp.all(K2 >= K)

	@pytest.mark.parametrize("length_scale,alpha", [
		(0.5, 1.0),
		(1.0, 0.5),
		(1.0, 1.0),
		(2.0, 2.0),
	])
	@allure.title("RationalQuadraticKernel comparison with scikit-learn")
	@allure.description("Compare RQ kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data, length_scale, alpha):
		from sklearn.gaussian_process.kernels import RationalQuadratic

		kernel = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		sklearn_kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale,alpha", [
		(0.5, 1.0),
		(1.0, 0.5),
		(1.0, 1.0),
		(2.0, 2.0),
	])
	@allure.title("RationalQuadraticKernel comparison with GPyTorch")
	@allure.description("Compare RQ kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data, length_scale, alpha):
		import torch
		from gpytorch.kernels import RQKernel

		kernel = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		gpytorch_kernel = RQKernel()
		gpytorch_kernel._set_lengthscale(length_scale)
		gpytorch_kernel.alpha = alpha

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale,alpha", [
		(0.5, 1.0),
		(1.0, 0.5),
		(1.0, 1.0),
		(2.0, 2.0),
	])
	@allure.title("RationalQuadraticKernel comparison with GPJax")
	@allure.description("Compare RQ kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data, length_scale, alpha):
		from gpjax.kernels import RationalQuadratic

		kernel = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		gpjax_kernel = RationalQuadratic(lengthscale=length_scale, alpha=alpha)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestPolynomialKernel:
	"""Tests for Polynomial Kernel."""

	@allure.title("PolynomialKernel Instantiation")
	@allure.description("Test that Polynomial kernel can be instantiated.")
	def test_instantiation(self):
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=0.0)
		assert kernel.degree == 2
		assert kernel.gamma == 1.0
		assert kernel.constant == 0.0

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
		K = kernel(x1, x1)

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

	@allure.title("PolynomialKernel comparison with scikit-learn")
	@allure.description("Compare Polynomial kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data):
		from sklearn.gaussian_process.kernels import DotProduct

		# For polynomial degree 1 with constant=0, compare with DotProduct
		kernel = PolynomialKernel(degree=1, gamma=1.0, constant=0.0)
		sklearn_kernel = DotProduct(sigma_0=0.0)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("PolynomialKernel comparison with GPyTorch")
	@allure.description("Compare Polynomial kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data):
		import torch
		from gpytorch.kernels import PolynomialKernel as GPyTorchPolynomialKernel

		# GPyTorch polynomial kernel has similar parameters
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=1.0)
		gpytorch_kernel = GPyTorchPolynomialKernel(power=2)
		gpytorch_kernel.offset = torch.tensor(1.0)

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@allure.title("PolynomialKernel comparison with GPJax")
	@allure.description("Compare Polynomial kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data):
		from gpjax.kernels import Polynomial

		# GPJax polynomial kernel
		kernel = PolynomialKernel(degree=2, gamma=1.0, constant=1.0)
		gpjax_kernel = Polynomial(degree=2, shift=1.0, variance=1.0)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestConstantKernel:
	"""Tests for Constant Kernel."""

	@allure.title("ConstantKernel Instantiation")
	@allure.description("Test that Constant kernel can be instantiated.")
	def test_instantiation(self):
		kernel = ConstantKernel(value=2.0)
		assert kernel.value == 2.0

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
		K = kernel(x1, x1)

		# Check all elements are the constant value
		assert jnp.allclose(K, 1.5)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

	@allure.title("ConstantKernel comparison with scikit-learn")
	@allure.description("Compare Constant kernel results against scikit-learn implementation.")
	def test_against_scikitlearn(self, sample_1d_data):
		from sklearn.gaussian_process.kernels import ConstantKernel as SKConstantKernel

		kernel = ConstantKernel(value=2.0)
		sklearn_kernel = SKConstantKernel(constant_value=2.0)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = sklearn_kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("ConstantKernel comparison with GPyTorch")
	@allure.description("Compare Constant kernel results against GPyTorch implementation.")
	def test_against_gpytorch(self, sample_1d_data):
		import torch
		from gpytorch.kernels import ConstantKernel as GPyTorchConstantKernel

		kernel = ConstantKernel(value=2.0)
		gpytorch_kernel = GPyTorchConstantKernel()
		gpytorch_kernel.constant = torch.tensor([[2.0]])

		x1, x2 = sample_1d_data
		x1_torch = torch.tensor(x1)
		x2_torch = torch.tensor(x2)

		result = kernel(x1, x2)
		expected = gpytorch_kernel(x1_torch, x2_torch).detach().numpy()

		assert jnp.allclose(result, expected)

	@allure.title("ConstantKernel comparison with GPJax")
	@allure.description("Compare Constant kernel results against GPJax implementation.")
	def test_against_gpjax(self, sample_1d_data):
		from gpjax.kernels import Constant

		kernel = ConstantKernel(value=2.0)
		gpjax_kernel = Constant(constant=2.0)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)
		expected = gpjax_kernel.cross_covariance(x1, x2)

		assert jnp.allclose(result, expected)


class TestWhiteNoiseKernel:
	""" Tests for WhiteNoiseKernel class. """
	# As WhiteNoiseKernel is just a shortcut to a Diag(Constant()) kernel, we only test instantiation and equivalence

	@allure.title("WhiteNoiseKernel Instantiation")
	@allure.description("Test that WhiteNoise kernel can be instantiated.")
	def test_instantiation(self):
		kernel = WhiteNoiseKernel(noise=1.0)
		assert kernel.inner_kernel.value == 1.0

	@pytest.mark.parametrize("noise", [0.5, 1.0, 2.0])
	@allure.title("WhiteNoiseKernel comparison with Diag(Constant())")
	@allure.description("Compare WhiteNoiseKernel results against Diag(Constant()).")
	def test_against_diag(self, sample_1d_data, noise):
		white_noise_kernel = WhiteNoiseKernel(noise=noise)
		diag_const_kernel = DiagKernel(noise)

		x1, _ = sample_1d_data
		result = white_noise_kernel(x1)
		expected = diag_const_kernel(x1)

		assert jnp.allclose(result, expected)


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