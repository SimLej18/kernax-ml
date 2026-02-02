"""
Tests for kernel composition operations (add, subtract, multiply, negate).
"""

import allure
import jax.numpy as jnp
import pytest

from kernax import (
	ConstantKernel,
	ExpKernel,
	LogKernel,
	NegKernel,
	PolynomialKernel,
	ProductKernel,
	RationalQuadraticKernel,
	SEKernel,
	SumKernel,
)


class TestKernelAddition:
	"""Tests for kernel addition operations."""

	@allure.title("SumKernel instantiation with explicit constructor")
	@allure.description("Test that SumKernel can be instantiated explicitly.")
	def test_sum_kernel_instantiation(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = SumKernel(k1, k2)
		assert isinstance(kernel.left_kernel, SEKernel)
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@allure.title("Kernel addition with + operator")
	@allure.description("Test kernel addition using __add__.")
	def test_kernel_add(self):
		kernel1 = SEKernel(length_scale=1.0)
		kernel2 = SEKernel(length_scale=2.0)
		combined = kernel1 + kernel2
		assert isinstance(combined, SumKernel)

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = kernel1(x1, x2) + kernel2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel right addition with constant")
	@allure.description("Test kernel addition using __radd__ with a scalar.")
	def test_kernel_radd(self):
		kernel = SEKernel(length_scale=1.0)
		combined = 2.0 + kernel  # This triggers __radd__

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = 2.0 + kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel addition with matrix inputs")
	@allure.description("Test kernel addition with 2D inputs.")
	def test_kernel_add_matrix(self, sample_1d_data):
		kernel1 = SEKernel(length_scale=1.0)
		kernel2 = ConstantKernel(value=0.5)
		combined = kernel1 + kernel2

		x1, x2 = sample_1d_data
		result = combined(x1, x2)
		expected = kernel1(x1, x2) + kernel2(x1, x2)

		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize(
		"length_scale,constant_value",
		[
			(0.5, 0.5),
			(1.0, 1.0),
			(2.0, 0.1),
		],
	)
	@allure.title("SumKernel computation with different parameters")
	@allure.description("Test sum kernel with various parameter combinations.")
	def test_computation_parametrized(self, sample_1d_data, length_scale, constant_value):
		k1 = SEKernel(length_scale=length_scale)
		k2 = ConstantKernel(value=constant_value)
		kernel = k1 + k2
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = k1(x1, x2) + k2(x1, x2)

		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize(
		"length_scale,alpha,degree",
		[
			(1.0, 1.0, 2),
			(0.5, 2.0, 3),
			(2.0, 0.5, 1),
		],
	)
	@allure.title("SumKernel with RationalQuadratic and Polynomial")
	@allure.description("Test sum kernel with different kernel types.")
	def test_computation_rq_plus_poly(self, sample_1d_data, length_scale, alpha, degree):
		k1 = RationalQuadraticKernel(length_scale=length_scale, alpha=alpha)
		k2 = PolynomialKernel(degree=degree, gamma=1.0, constant=1.0)
		kernel = k1 + k2
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = k1(x1, x2) + k2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("SumKernel auto-conversion of scalars")
	@allure.description("Test that scalars are automatically converted to ConstantKernel.")
	def test_auto_convert_scalar(self):
		k1 = SEKernel(length_scale=1.0)
		# Constructor mode
		kernel1 = SumKernel(k1, 2.0)
		assert isinstance(kernel1.right_kernel, ConstantKernel)

		# Operator mode
		kernel2 = k1 + 2.0
		assert isinstance(kernel2.right_kernel, ConstantKernel)

	@allure.title("SumKernel mathematical properties")
	@allure.description("Test that sum kernel preserves mathematical properties.")
	def test_math_properties(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = k1 + k2
		x1, _ = sample_1d_data
		K = kernel(x1)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K) > 0)


class TestKernelSubtraction:
	"""Tests for kernel subtraction operations."""

	@allure.title("Kernel subtraction with - operator")
	@allure.description("Test kernel subtraction using __sub__.")
	def test_kernel_sub(self):
		kernel1 = ConstantKernel(value=3.0)
		kernel2 = ConstantKernel(value=1.0)
		combined = kernel1 - kernel2

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = kernel1(x1, x2) - kernel2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel right subtraction with constant")
	@allure.description("Test kernel subtraction using __rsub__ with a scalar.")
	def test_kernel_rsub(self):
		kernel = ConstantKernel(value=1.0)
		combined = 5.0 - kernel  # This triggers __rsub__

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = 5.0 - kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel subtraction with matrix inputs")
	@allure.description("Test kernel subtraction with 2D inputs.")
	def test_kernel_sub_matrix(self, sample_1d_data):
		kernel1 = SEKernel(length_scale=1.0)
		kernel2 = ConstantKernel(value=0.2)
		combined = kernel1 - kernel2

		x1, x2 = sample_1d_data
		result = combined(x1, x2)
		expected = kernel1(x1, x2) - kernel2(x1, x2)

		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.allclose(result, expected)


class TestKernelMultiplication:
	"""Tests for kernel multiplication operations."""

	@allure.title("ProductKernel instantiation with explicit constructor")
	@allure.description("Test that ProductKernel can be instantiated explicitly.")
	def test_product_kernel_instantiation(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		kernel = ProductKernel(k1, k2)
		assert isinstance(kernel.left_kernel, SEKernel)
		assert isinstance(kernel.right_kernel, ConstantKernel)

	@allure.title("Kernel multiplication with * operator")
	@allure.description("Test kernel multiplication using __mul__.")
	def test_kernel_mul(self):
		kernel1 = SEKernel(length_scale=1.0)
		kernel2 = ConstantKernel(value=2.0)
		combined = kernel1 * kernel2
		assert isinstance(combined, ProductKernel)

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = kernel1(x1, x2) * kernel2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel right multiplication with constant")
	@allure.description("Test kernel multiplication using __rmul__ with a scalar.")
	def test_kernel_rmul(self):
		kernel = SEKernel(length_scale=1.0)
		combined = 3.0 * kernel  # This triggers __rmul__

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = 3.0 * kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel multiplication with matrix inputs")
	@allure.description("Test kernel multiplication with 2D inputs.")
	def test_kernel_mul_matrix(self, sample_1d_data):
		kernel1 = SEKernel(length_scale=1.0)
		kernel2 = SEKernel(length_scale=2.0)
		combined = kernel1 * kernel2

		x1, x2 = sample_1d_data
		result = combined(x1, x2)
		expected = kernel1(x1, x2) * kernel2(x1, x2)

		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize(
		"length_scale,constant_value",
		[
			(0.5, 0.5),
			(1.0, 1.0),
			(2.0, 0.1),
		],
	)
	@allure.title("ProductKernel computation with different parameters")
	@allure.description("Test product kernel with various parameter combinations.")
	def test_computation_parametrized(self, sample_1d_data, length_scale, constant_value):
		k1 = SEKernel(length_scale=length_scale)
		k2 = ConstantKernel(value=constant_value)
		kernel = k1 * k2
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = k1(x1, x2) * k2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("ProductKernel auto-conversion of scalars")
	@allure.description("Test that scalars are automatically converted to ConstantKernel.")
	def test_auto_convert_scalar(self):
		k1 = SEKernel(length_scale=1.0)
		# Constructor mode
		kernel1 = ProductKernel(k1, 2.0)
		assert isinstance(kernel1.right_kernel, ConstantKernel)

		# Operator mode
		kernel2 = k1 * 2.0
		assert isinstance(kernel2.right_kernel, ConstantKernel)

	@allure.title("ProductKernel mathematical properties")
	@allure.description("Test that product kernel preserves mathematical properties.")
	def test_math_properties(self, sample_1d_data):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=2.0)
		kernel = k1 * k2
		x1, _ = sample_1d_data
		K = kernel(x1)

		# Check matrix is symmetric
		assert jnp.allclose(K, K.T)

		# Check diagonal elements are positive
		assert jnp.all(jnp.diag(K) > 0)


class TestKernelNegation:
	"""Tests for kernel negation operation."""

	@allure.title("NegKernel instantiation with explicit constructor")
	@allure.description("Test that NegKernel can be instantiated explicitly.")
	def test_neg_kernel_instantiation(self):
		inner = ConstantKernel(value=1.0)
		kernel = NegKernel(inner)
		assert isinstance(kernel.inner_kernel, ConstantKernel)

	@allure.title("Kernel negation with - operator")
	@allure.description("Test kernel negation using __neg__.")
	def test_kernel_neg(self):
		kernel = SEKernel(length_scale=1.0)
		negated = -kernel
		assert isinstance(negated, NegKernel)

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = negated(x1, x2)
		expected = -kernel(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Kernel negation with matrix inputs")
	@allure.description("Test kernel negation with 2D inputs.")
	def test_kernel_neg_matrix(self, sample_1d_data):
		kernel = SEKernel(length_scale=1.0)
		negated = -kernel

		x1, x2 = sample_1d_data
		result = negated(x1, x2)
		expected = -kernel(x1, x2)

		assert result.shape == (x1.shape[0], x2.shape[0])
		assert jnp.allclose(result, expected)

	@pytest.mark.parametrize("constant_value", [0.5, 1.0, 2.0])
	@allure.title("NegKernel computation with different parameters")
	@allure.description("Test negation with various parameter values.")
	def test_computation_parametrized(self, sample_1d_data, constant_value):
		inner = ConstantKernel(value=constant_value)
		kernel = -inner
		x1, x2 = sample_1d_data

		result = kernel(x1, x2)
		expected = -inner(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("NegKernel auto-conversion of scalars")
	@allure.description("Test that scalars are automatically converted to ConstantKernel.")
	def test_auto_convert_scalar(self):
		kernel = NegKernel(2.0)
		assert isinstance(kernel.inner_kernel, ConstantKernel)
		assert kernel.inner_kernel.value == 2.0


class TestExpKernel:
	"""Tests for exponential wrapper kernel."""

	@allure.title("ExpKernel instantiation")
	@allure.description("Test that ExpKernel can be instantiated.")
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

	@allure.title("ExpKernel auto-conversion of scalars")
	@allure.description("Test that scalars are automatically converted to ConstantKernel.")
	def test_auto_convert_scalar(self):
		kernel = ExpKernel(2.0)
		assert isinstance(kernel.inner_kernel, ConstantKernel)
		assert kernel.inner_kernel.value == 2.0


class TestLogKernel:
	"""Tests for logarithm wrapper kernel."""

	@allure.title("LogKernel instantiation")
	@allure.description("Test that LogKernel can be instantiated.")
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

	@allure.title("LogKernel auto-conversion of scalars")
	@allure.description("Test that scalars are automatically converted to ConstantKernel.")
	def test_auto_convert_scalar(self):
		kernel = LogKernel(2.0)
		assert isinstance(kernel.inner_kernel, ConstantKernel)
		assert kernel.inner_kernel.value == 2.0


class TestComplexCompositions:
	"""Tests for complex kernel compositions."""

	@allure.title("Complex composition chain")
	@allure.description("Test chaining multiple kernel operations.")
	def test_complex_chain(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = SEKernel(length_scale=2.0)
		k3 = ConstantKernel(value=0.5)

		# Test: (k1 + k2) * k3
		combined = (k1 + k2) * k3

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = (k1(x1, x2) + k2(x1, x2)) * k3(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("Complex composition with scalars")
	@allure.description("Test mixing kernels and scalars in complex expressions.")
	def test_complex_with_scalars(self):
		kernel = SEKernel(length_scale=1.0)

		# Test: 2.0 * kernel + 1.0
		combined = 2.0 * kernel + 1.0

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = 2.0 * kernel(x1, x2) + 1.0

		assert jnp.allclose(result, expected)

	@allure.title("Complex composition with negation")
	@allure.description("Test complex expressions involving negation.")
	def test_complex_with_negation(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.3)

		# Test: k1 - 2.0 * k2
		combined = k1 - 2.0 * k2

		x1 = jnp.array([1.0])
		x2 = jnp.array([2.0])

		result = combined(x1, x2)
		expected = k1(x1, x2) - 2.0 * k2(x1, x2)

		assert jnp.allclose(result, expected)

	@allure.title("String representation of composite kernels")
	@allure.description("Test that composite kernels have valid string representations.")
	def test_composite_str_representation(self):
		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=2.0)

		# Test addition
		sum_kernel = k1 + k2
		assert isinstance(str(sum_kernel), str)
		assert len(str(sum_kernel)) > 0

		# Test multiplication
		prod_kernel = k1 * k2
		assert isinstance(str(prod_kernel), str)
		assert len(str(prod_kernel)) > 0

		# Test negation
		neg_kernel = -k1
		assert isinstance(str(neg_kernel), str)
		assert len(str(neg_kernel)) > 0

		# Test subtraction (which creates SumKernel with NegKernel)
		diff_kernel = k1 - k2
		assert isinstance(str(diff_kernel), str)
		assert len(str(diff_kernel)) > 0

	@allure.title("Complex composition with diagonal computation engine")
	@allure.description("Test kernel with multiple composition operations including diagonal engine.")
	def test_multiple_operations_with_diag(self, sample_1d_data):
		from kernax.engines import SafeDiagonalEngine

		k1 = SEKernel(length_scale=1.0)
		k2 = ConstantKernel(value=0.5)
		k3 = ExpKernel(0.1, computation_engine=SafeDiagonalEngine)

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

	@allure.title("Realistic GP kernel composition")
	@allure.description("Test a realistic GP kernel: RBF + noise.")
	def test_realistic_gp_kernel(self, sample_1d_data):
		from kernax.engines import SafeDiagonalEngine

		# Common pattern: signal kernel + noise on diagonal
		signal = SEKernel(length_scale=1.0)
		noise = ExpKernel(0.1, computation_engine=SafeDiagonalEngine)
		kernel = signal + noise

		x1, _ = sample_1d_data
		K = kernel(x1)

		# Should be symmetric
		assert jnp.allclose(K, K.T)
		# Diagonal should be larger due to noise
		signal_diag = jnp.diag(signal(x1))
		full_diag = jnp.diag(K)
		assert jnp.all(full_diag >= signal_diag)

		# Verify output value matches manual computation
		expected = signal(x1) + noise(x1)
		assert jnp.allclose(K, expected)

	@allure.title("Deeply nested kernel operations")
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

	@allure.title("Kernel operation associativity")
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

	@allure.title("Kernel operation distributivity")
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
