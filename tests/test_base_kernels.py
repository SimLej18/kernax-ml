"""
Tests for base kernel implementations.
"""

import pytest
import jax.numpy as jnp
from Kernax import (
    RBFKernel,
    LinearKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    PeriodicKernel,
    RationalQuadraticKernel,
    ConstantKernel,
    SEMagmaKernel,
)


class TestRBFKernel:
    """Tests for RBF (Squared Exponential) Kernel."""

    def test_instantiation(self):
        """Test that RBF kernel can be instantiated."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        assert kernel.length_scale == 1.0
        assert kernel.variance == 1.0

    def test_scalar_computation(self):
        """Test covariance computation between two scalars."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1 = jnp.array([1.0])
        x2 = jnp.array([2.0])
        result = kernel(x1, x2)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_vector_computation(self, sample_1d_data):
        """Test covariance computation between vectors."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1, x2 = sample_1d_data
        result = kernel(x1, x2)
        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))

    def test_self_covariance_positive(self, sample_1d_data):
        """Test that self-covariance is positive definite."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        x1, _ = sample_1d_data
        K = kernel(x1, x1)
        # Check diagonal elements are positive
        assert jnp.all(jnp.diag(K) > 0)
        # Check matrix is symmetric
        assert jnp.allclose(K, K.T)


class TestLinearKernel:
    """Tests for Linear Kernel."""

    def test_instantiation(self):
        """Test that Linear kernel can be instantiated."""
        kernel = LinearKernel(variance_b=0.5, variance_v=1.0, offset_c=0.0)
        assert kernel.variance_b == 0.5
        assert kernel.variance_v == 1.0
        assert kernel.offset_c == 0.0

    def test_computation(self, sample_1d_data):
        """Test covariance computation."""
        kernel = LinearKernel(variance_b=0.5, variance_v=1.0, offset_c=0.0)
        x1, x2 = sample_1d_data
        result = kernel(x1, x2)
        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))


class TestMaternKernels:
    """Tests for Matern kernel family."""

    @pytest.mark.parametrize(
        "KernelClass", [Matern12Kernel, Matern32Kernel, Matern52Kernel]
    )
    def test_instantiation(self, KernelClass):
        """Test that Matern kernels can be instantiated."""
        kernel = KernelClass(length_scale=1.0)
        assert kernel.length_scale == 1.0

    @pytest.mark.parametrize(
        "KernelClass", [Matern12Kernel, Matern32Kernel, Matern52Kernel]
    )
    def test_computation(self, KernelClass, sample_1d_data):
        """Test covariance computation for Matern kernels."""
        kernel = KernelClass(length_scale=1.0)
        x1, x2 = sample_1d_data
        result = kernel(x1, x2)
        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))


class TestPeriodicKernel:
    """Tests for Periodic Kernel."""

    def test_instantiation(self):
        """Test that Periodic kernel can be instantiated."""
        kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
        assert kernel.length_scale == 1.0
        assert kernel.variance == 1.0
        assert kernel.period == 2.0

    def test_computation(self, sample_1d_data):
        """Test covariance computation."""
        kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=2.0)
        x1, x2 = sample_1d_data
        result = kernel(x1, x2)
        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))

    def test_periodicity(self):
        """Test that kernel respects periodicity."""
        period = 2.0
        kernel = PeriodicKernel(length_scale=1.0, variance=1.0, period=period)
        x = jnp.array([0.5])
        y1 = jnp.array([1.5])
        y2 = jnp.array([1.5 + period])
        # Values separated by one period should have same covariance
        assert jnp.allclose(kernel(x, y1), kernel(x, y2), atol=1e-5)


class TestRationalQuadraticKernel:
    """Tests for Rational Quadratic Kernel."""

    def test_instantiation(self):
        """Test that RQ kernel can be instantiated."""
        kernel = RationalQuadraticKernel(length_scale=1.0, variance=1.0, alpha=1.0)
        assert kernel.length_scale == 1.0
        assert kernel.variance == 1.0
        assert kernel.alpha == 1.0

    def test_computation(self, sample_1d_data):
        """Test covariance computation."""
        kernel = RationalQuadraticKernel(length_scale=1.0, variance=1.0, alpha=1.0)
        x1, x2 = sample_1d_data
        result = kernel(x1, x2)
        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))


class TestConstantKernel:
    """Tests for Constant Kernel."""

    def test_instantiation(self):
        """Test that Constant kernel can be instantiated."""
        kernel = ConstantKernel(value=2.0)
        assert kernel.value == 2.0

    def test_returns_constant(self):
        """Test that kernel returns constant value."""
        value = 2.0
        kernel = ConstantKernel(value=value)
        x1 = jnp.array([1.0])
        x2 = jnp.array([5.0])
        result = kernel(x1, x2)
        assert jnp.allclose(result, value)


class TestSEMagmaKernel:
    """Tests for SEMagma Kernel."""

    def test_instantiation(self):
        """Test that SEMagma kernel can be instantiated."""
        kernel = SEMagmaKernel(length_scale=1.0, variance=1.0)
        assert kernel.length_scale == 1.0
        assert kernel.variance == 1.0

    def test_computation(self, sample_1d_data):
        """Test covariance computation."""
        kernel = SEMagmaKernel(length_scale=1.0, variance=1.0)
        x1, x2 = sample_1d_data
        result = kernel(x1, x2)
        assert result.shape == (x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))