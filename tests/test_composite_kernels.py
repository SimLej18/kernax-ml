"""
Tests for composite and wrapper kernels.
"""

import pytest
import jax.numpy as jnp
from kernax import (
    SEKernel,
    ConstantKernel,
    SumKernel,
    ProductKernel,
    DiagKernel,
    ExpKernel,
    LogKernel,
    NegKernel,
)


class TestSumKernel:
    """Tests for Sum Kernel."""

    def test_instantiation_with_kernels(self):
        """Test that Sum kernel can be instantiated with two kernels."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        kernel = SumKernel(k1, k2)
        assert isinstance(kernel.left_kernel, SEKernel)
        assert isinstance(kernel.right_kernel, ConstantKernel)

    def test_addition_operator(self):
        """Test that + operator creates SumKernel."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        kernel = k1 + k2
        assert isinstance(kernel, SumKernel)

    def test_computation(self, sample_1d_data):
        """Test that sum kernel computes correctly."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        kernel = k1 + k2
        x1, x2 = sample_1d_data

        result = kernel(x1, x2)
        expected = k1(x1, x2) + k2(x1, x2)

        assert jnp.allclose(result, expected)

    def test_auto_convert_scalar_to_constant(self):
        """Test that scalars are auto-converted to ConstantKernel."""
        k1 = SEKernel(length_scale=1.0)
        kernel = SumKernel(k1, 2.0)
        assert isinstance(kernel.right_kernel, ConstantKernel)


class TestProductKernel:
    """Tests for Product Kernel."""

    def test_instantiation_with_kernels(self):
        """Test that Product kernel can be instantiated with two kernels."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        kernel = ProductKernel(k1, k2)
        assert isinstance(kernel.left_kernel, SEKernel)
        assert isinstance(kernel.right_kernel, ConstantKernel)

    def test_multiplication_operator(self):
        """Test that * operator creates ProductKernel."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        kernel = k1 * k2
        assert isinstance(kernel, ProductKernel)

    def test_computation(self, sample_1d_data):
        """Test that product kernel computes correctly."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        kernel = k1 * k2
        x1, x2 = sample_1d_data

        result = kernel(x1, x2)
        expected = k1(x1, x2) * k2(x1, x2)

        assert jnp.allclose(result, expected)


class TestDiagKernel:
    """Tests for Diagonal Kernel."""

    def test_instantiation(self):
        """Test that Diag kernel can be instantiated."""
        inner = ConstantKernel(value=1.0)
        kernel = DiagKernel(inner)
        assert isinstance(kernel.inner_kernel, ConstantKernel)

    def test_diagonal_behavior(self):
        """Test that kernel returns 0 for non-equal inputs."""
        kernel = DiagKernel(ConstantKernel(value=1.0))
        x1 = jnp.array([1.0])
        x2 = jnp.array([2.0])
        result = kernel(x1, x2)
        assert jnp.allclose(result, 0.0)

    def test_diagonal_equal_inputs(self):
        """Test that kernel returns inner kernel value for equal inputs."""
        value = 1.5
        kernel = DiagKernel(ConstantKernel(value=value))
        x = jnp.array([1.0])
        result = kernel(x, x)
        assert jnp.allclose(result, value)

    def test_creates_diagonal_matrix(self, sample_1d_data):
        """Test that kernel creates diagonal covariance matrix."""
        kernel = DiagKernel(ConstantKernel(value=1.0))
        x1, _ = sample_1d_data
        K = kernel(x1, x1)
        # Off-diagonal elements should be zero
        n = x1.shape[0]
        mask = ~jnp.eye(n, dtype=bool)
        assert jnp.allclose(K[mask], 0.0)


class TestExpKernel:
    """Tests for Exponential Wrapper Kernel."""

    def test_instantiation(self):
        """Test that Exp kernel can be instantiated."""
        inner = ConstantKernel(value=1.0)
        kernel = ExpKernel(inner)
        assert isinstance(kernel.inner_kernel, ConstantKernel)

    def test_computation(self, sample_1d_data):
        """Test that exp kernel applies exponential."""
        inner = ConstantKernel(value=1.0)
        kernel = ExpKernel(inner)
        x1, x2 = sample_1d_data

        result = kernel(x1, x2)
        expected = jnp.exp(inner(x1, x2))

        assert jnp.allclose(result, expected)


class TestLogKernel:
    """Tests for Logarithm Wrapper Kernel."""

    def test_instantiation(self):
        """Test that Log kernel can be instantiated."""
        inner = ConstantKernel(value=2.0)
        kernel = LogKernel(inner)
        assert isinstance(kernel.inner_kernel, ConstantKernel)

    def test_computation(self, sample_1d_data):
        """Test that log kernel applies logarithm."""
        inner = ConstantKernel(value=2.0)
        kernel = LogKernel(inner)
        x1, x2 = sample_1d_data

        result = kernel(x1, x2)
        expected = jnp.log(inner(x1, x2))

        assert jnp.allclose(result, expected)


class TestNegKernel:
    """Tests for Negation Wrapper Kernel."""

    def test_instantiation(self):
        """Test that Neg kernel can be instantiated."""
        inner = ConstantKernel(value=1.0)
        kernel = NegKernel(inner)
        assert isinstance(kernel.inner_kernel, ConstantKernel)

    def test_negation_operator(self):
        """Test that unary - operator creates NegKernel."""
        inner = ConstantKernel(value=1.0)
        kernel = -inner
        assert isinstance(kernel, NegKernel)

    def test_computation(self, sample_1d_data):
        """Test that neg kernel negates output."""
        inner = ConstantKernel(value=1.0)
        kernel = -inner
        x1, x2 = sample_1d_data

        result = kernel(x1, x2)
        expected = -inner(x1, x2)

        assert jnp.allclose(result, expected)


class TestComplexComposition:
    """Tests for complex kernel compositions."""

    def test_multiple_operations(self, sample_1d_data):
        """Test kernel with multiple composition operations."""
        k1 = SEKernel(length_scale=1.0)
        k2 = ConstantKernel(value=0.5)
        k3 = DiagKernel(ExpKernel(0.1))

        # Create: (RBF + Constant) * DiagExp
        kernel = (k1 + k2) * k3

        x1, x2 = sample_1d_data
        result = kernel(x1, x2)

        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))

    def test_realistic_gp_kernel(self, sample_1d_data):
        """Test a realistic GP kernel: RBF + noise."""
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