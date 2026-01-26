"""
Tests for composite and wrapper kernels.
"""

import jax.numpy as jnp
import pytest
import allure

from kernax import (
	ConstantKernel,
	DiagKernel,
	ExpKernel,
	LogKernel,
	NegKernel,
	ProductKernel,
	SEKernel,
	SumKernel,
	RationalQuadraticKernel,
	PolynomialKernel,
)


class TestSumKernel:
	"""Tests for Sum Kernel."""

	@allure.title("SumKernel Instantiation with kernels")
	@allure.description("Test that Sum kernel can be instantiated with two kernels.")
	def test_instantiation_with_kernels(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = SumKernel(k1, k2)
		assert isinstance(kernel.left_kernel, SEKernel)
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@allure.title("SumKernel addition operator")
	@allure.description("Test that + operator creates SumKernel.")
	def test_addition_operator(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = k1 + k2
		assert isinstance(kernel, SumKernel)
		assert isinstance(kernel.left_kernel, SEKernel)
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@pytest.mark.parametrize("length_scale,constant_value", [
		(0.5, 0.5),
		(1.0, 1.0),
		(2.0, 0.1),
	])
	@allure.title("SumKernel computation SE + Constant")
	@allure.description("Test that sum kernel computes correctly with SE and Constant kernels.")
	def test_computation_SE_plus_const(self, sample_1d_data, length_scale, constant_value):
		k1 = SEKernel(length_scale=length_scale)
		k2 = ConstantKernel(value=constant_value)
		kernel = k1 + k2
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = k1(x1, x2) + k2(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("length_scale,alpha,degree", [
		(1.0, 1.0, 2),
		(0.5, 2.0, 3),
		(2.0, 0.5, 1),
	])
	@allure.title("SumKernel computation RQ + Polynomial")
	@allure.description("Test that sum kernel computes correctly with RationalQuadratic and Polynomial kernels.")
	def test_computation_RQ_plus_poly(self, sample_1d_data, length_scale, alpha, degree):
		k1 = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		k2 = PolynomialKernel(degree=degree, gamma=1.0, constant=1.0)
		kernel = k1 + k2
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = k1(x1, x2) + k2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("SumKernel auto-convert scalar to Constant")
	@allure.description("Test that scalars are auto-converted to ConstantKernel.")
	def test_auto_convert_scalar_to_constant(self):
		k1 = SEKernel(length_scale=1.0)
		# Constructor mode
		kernel = SumKernel(k1, 2.0)
		assert isinstance(kernel.right_kernel, ConstantKernel)

		# Operator mode
		kernel = k1 + 2.
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@allure.title("SumKernel mathematical properties")
	@allure.description("Test that sum kernel preserves mathematical properties.")
	def test_math_properties(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = k1 + k2
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K) > 0)


class TestProductKernel:
	"""Tests for Product Kernel."""

	@allure.title("ProductKernel Instantiation with kernels")
	@allure.description("Test that Product kernel can be instantiated with two kernels.")
	def test_instantiation_with_kernels(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = ProductKernel(k1, k2)
		assert isinstance(kernel.left_kernel, SEKernel)
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@allure.title("ProductKernel multiplication operator")
	@allure.description("Test that * operator creates ProductKernel.")
	def test_multiplication_operator(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = k1 * k2
		assert isinstance(kernel, ProductKernel)

	@pytest.mark.parametrize("length_scale,constant_value", [
		(0.5, 0.5),
		(1.0, 1.0),
		(2.0, 0.1),
	])
	@allure.title("ProductKernel computation")
	@allure.description("Test that product kernel computes correctly.")
	def test_computation(self, sample_1d_data, length_scale, constant_value):
		k1 = SEKernel(length_scale=length_scale)
		k2 = ConstantKernel(value=constant_value)
		kernel = k1 * k2
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = k1(x1, x2) * k2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("ProductKernel auto-convert scalar to Constant")
	@allure.description("Test that scalars are auto-converted to ConstantKernel.")
	def test_auto_convert_scalar_to_constant(self):
		k1 = SEKernel(length_scale=1.0)
		# Constructor mode
		kernel = ProductKernel(k1, 2.0)
		assert isinstance(kernel.right_kernel, ConstantKernel)

		# Operator mode
		kernel = k1 * 2.
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@allure.title("ProductKernel mathematical properties")
	@allure.description("Test that product kernel preserves mathematical properties.")
	def test_math_properties(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=2.0)
		kernel = k1 * k2
		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K) > 0)


class TestExpKernel:
	"""Tests for Exponential Wrapper Kernel."""

	@allure.title("ExpKernel Instantiation")
	@allure.description("Test that Exp kernel can be instantiated.")
	def test_instantiation(self):
		inner = ConstantKernel(value=1.0)
		kernel = ExpKernel(inner)
		assert isinstance(kernel.inner_kernel, ConstantKernel)

	@pytest.mark.parametrize("constant_value", [0.5, 1.0, 2.0])
	@allure.title("ExpKernel computation")
	@allure.description("Test that exp kernel applies exponential.")
	def test_computation(self, sample_1d_data, constant_value):
		inner = ConstantKernel(value=constant_value)
		kernel = ExpKernel(inner)
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = jnp.exp(inner(x1, x2))

		assert jnp.allclose(result, expected)

	@allure.title("ExpKernel auto-convert scalar to Constant")
	@allure.description("Test that scalars are auto-converted to ConstantKernel.")
	def test_auto_convert_scalar_to_constant(self):
		kernel = ExpKernel(2.0)
		assert isinstance(kernel.inner_kernel, ConstantKernel)
		assert kernel.inner_kernel.value == 2.0


class TestLogKernel:
	"""Tests for Logarithm Wrapper Kernel."""

	@allure.title("LogKernel Instantiation")
	@allure.description("Test that Log kernel can be instantiated.")
	def test_instantiation(self):
		inner = ConstantKernel(value=2.0)
		kernel = LogKernel(inner)
		assert isinstance(kernel.inner_kernel, ConstantKernel)

	@pytest.mark.parametrize("constant_value", [1.0, 2.0, 5.0])
	@allure.title("LogKernel computation")
	@allure.description("Test that log kernel applies logarithm.")
	def test_computation(self, sample_1d_data, constant_value):
		inner = ConstantKernel(value=constant_value)
		kernel = LogKernel(inner)
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = jnp.log(inner(x1, x2))

		assert jnp.allclose(result, expected)

	@allure.title("LogKernel auto-convert scalar to Constant")
	@allure.description("Test that scalars are auto-converted to ConstantKernel.")
	def test_auto_convert_scalar_to_constant(self):
		kernel = LogKernel(2.0)
		assert isinstance(kernel.inner_kernel, ConstantKernel)
		assert kernel.inner_kernel.value == 2.0


class TestNegKernel:
	"""Tests for Negation Wrapper Kernel."""

	@allure.title("NegKernel Instantiation")
	@allure.description("Test that Neg kernel can be instantiated.")
	def test_instantiation(self):
		inner = ConstantKernel(value=1.0)
		kernel = NegKernel(inner)
		assert isinstance(kernel.inner_kernel, ConstantKernel)

	@allure.title("NegKernel negation operator")
	@allure.description("Test that unary - operator creates NegKernel.")
	def test_negation_operator(self):
		inner = ConstantKernel(value=1.0)
		kernel = -inner
		assert isinstance(kernel, NegKernel)

	@pytest.mark.parametrize("constant_value", [0.5, 1.0, 2.0])
	@allure.title("NegKernel computation")
	@allure.description("Test that neg kernel negates output.")
	def test_computation(self, sample_1d_data, constant_value):
		inner = ConstantKernel(value=constant_value)
		kernel = -inner
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = -inner(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("NegKernel auto-convert scalar to Constant")
	@allure.description("Test that scalars are auto-converted to ConstantKernel.")
	def test_auto_convert_scalar_to_constant(self):
		kernel = NegKernel(2.0)
		assert isinstance(kernel.inner_kernel, ConstantKernel)
		assert kernel.inner_kernel.value == 2.0


class TestComplexComposition:
	"""Tests for complex kernel compositions."""

	@allure.title("ComplexComposition multiple operations")
	@allure.description("Test kernel with multiple composition operations.")
	def test_multiple_operations(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		k3 = DiagKernel(ExpKernel(0.1))

		# Create: (RBF + Constant) * DiagExp
		kernel = (k1 + k2) * k3

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)

		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))

		# Verify output value matches manual computation
		sum_kernel = k1 + k2
		expected = sum_kernel(x1, x2) * k3(x1, x2)
		assert jnp.allclose(result, expected)

	@allure.title("ComplexComposition realistic GP kernel")
	@allure.description("Test a realistic GP kernel: RBF + noise.")
	def test_realistic_gp_kernel(self, sample_1d_data):
		# Common pattern: signal kernel + noise on diagonal
		signal = SEKernel(length_scale=1.0)
		noise = DiagKernel(ExpKernel(0.1))
		kernel = signal + noise

		x1, _ = sample_1d_data
		K = kernel(x1, x1)

		# Should be symmetric
		assert jnp.allclose(K, K.T)
		# Diagonal should be larger due to noise
		signal_diag = jnp.diag(signal(x1, x1))
		full_diag = jnp.diag(K)
		assert jnp.all(full_diag >= signal_diag)

		# Verify output value matches manual computation
		expected = signal(x1, x1) + noise(x1, x1)
		assert jnp.allclose(K, expected)

	@allure.title("ComplexComposition nested operations")
	@allure.description("Test deeply nested kernel compositions.")
	def test_nested_operations(self, sample_1d_data):
		# Create: exp((k1 + k2) * k3)
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		k3 = ConstantKernel(value=2.0)

		kernel = ExpKernel((k1 + k2) * k3)

		x1, x2 = sample_1d_data
		result = kernel(x1, x2)

		# Verify shape and finiteness
		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.all(jnp.isfinite(result))

		# Verify output value
		inner = (k1(x1, x2) + k2(x1, x2)) * k3(x1, x2)
		expected = jnp.exp(inner)
		assert jnp.allclose(result, expected)

	@allure.title("ComplexComposition associativity")
	@allure.description("Test that kernel operations are associative.")
	def test_associativity(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		k3 = ConstantKernel(value=2.0)

		x1, x2 = sample_1d_data

		# Test addition associativity: (k1 + k2) + k3 = k1 + (k2 + k3)
		left_assoc_add = (k1 + k2) + k3
		right_assoc_add = k1 + (k2 + k3)
		assert jnp.allclose(left_assoc_add(x1, x2), right_assoc_add(x1, x2))

		# Test multiplication associativity: (k1 * k2) * k3 = k1 * (k2 * k3)
		left_assoc_mul = (k1 * k2) * k3
		right_assoc_mul = k1 * (k2 * k3)
		assert jnp.allclose(left_assoc_mul(x1, x2), right_assoc_mul(x1, x2))

	@allure.title("ComplexComposition distributivity")
	@allure.description("Test that multiplication distributes over addition.")
	def test_distributivity(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		k3 = ConstantKernel(value=2.0)

		x1, x2 = sample_1d_data

		# Test: k1 * (k2 + k3) = k1 * k2 + k1 * k3
		left_side = k1 * (k2 + k3)
		right_side = (k1 * k2) + (k1 * k3)
		assert jnp.allclose(left_side(x1, x2), right_side(x1, x2))
