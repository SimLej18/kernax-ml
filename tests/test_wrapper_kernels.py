"""
Tests for wrapper kernels (BatchKernel, ActiveDimsKernel, ARDKernel).
"""

import pytest
import jax.numpy as jnp
from kernax import (
    SEKernel,
    ConstantKernel,
    BatchKernel,
    ActiveDimsKernel,
    ARDKernel,
)


class TestBatchKernel:
    """Tests for BatchKernel wrapper."""

    def test_instantiation(self):
        """Test that BatchKernel can be instantiated."""
        base_kernel = SEKernel(length_scale=1.0)
        batch_kernel = BatchKernel(
            base_kernel,
            batch_size=5,
            batch_in_axes=0,
            batch_over_inputs=True
        )
        assert batch_kernel.inner_kernel is not None
        assert batch_kernel.batch_over_inputs == 0

    def test_batch_over_hyperparameters(self):
        """Test batching with distinct hyperparameters per batch element."""
        # Create base kernel with single length_scale
        base_kernel = SEKernel(length_scale=1.0)
        batch_size = 3

        # Wrap in BatchKernel to handle batched hyperparameters
        batch_kernel = BatchKernel(
            base_kernel,
            batch_size=batch_size,
            batch_in_axes=0,  # Batch over all hyperparameters
            batch_over_inputs=False  # Same inputs for all batches
        )

        # Create non-batched inputs
        x1 = jnp.array([[1.0], [2.0], [3.0]])
        x2 = jnp.array([[1.5], [2.5], [3.5]])

        # Compute covariance - should produce batched output
        result = batch_kernel(x1, x2)

        # Result should have batch dimension
        assert result.shape == (batch_size, x1.shape[0], x2.shape[0])
        assert jnp.all(jnp.isfinite(result))

    def test_batch_over_inputs_and_hyperparameters(self):
        """Test batching over both inputs and hyperparameters."""
        base_kernel = SEKernel(length_scale=1.0)
        batch_size = 4

        batch_kernel = BatchKernel(
            base_kernel,
            batch_size=batch_size,
            batch_in_axes=0,
            batch_over_inputs=True
        )

        # Create batched inputs
        x1_batched = jnp.array([
            [[1.0], [2.0]],
            [[1.5], [2.5]],
            [[2.0], [3.0]],
            [[2.5], [3.5]]
        ])  # Shape: (4, 2, 1)

        result = batch_kernel(x1_batched, x1_batched)

        # Should produce batch_size covariance matrices
        assert result.shape == (batch_size, 2, 2)
        assert jnp.all(jnp.isfinite(result))

        # Each batch element should be symmetric
        for i in range(batch_size):
            assert jnp.allclose(result[i], result[i].T)

    def test_batch_over_inputs_only(self):
        """Test batching over inputs with shared hyperparameters."""
        base_kernel = SEKernel(length_scale=1.0)
        batch_size = 3

        # Batch over inputs but share hyperparameters
        batch_kernel = BatchKernel(
            base_kernel,
            batch_size=batch_size,
            batch_in_axes=None,  # Shared hyperparameters
            batch_over_inputs=True
        )

        x_batched = jnp.array([
            [[1.0], [2.0]],
            [[1.5], [2.5]],
            [[2.0], [3.0]]
        ])

        result = batch_kernel(x_batched, x_batched)

        assert result.shape == (batch_size, 2, 2)
        assert jnp.all(jnp.isfinite(result))


class TestActiveDimsKernel:
    """Tests for ActiveDimsKernel wrapper."""

    def test_instantiation(self):
        """Test that ActiveDimsKernel can be instantiated."""
        base_kernel = SEKernel(length_scale=1.0)
        active_dims = jnp.array([0, 2])
        kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

        assert kernel.inner_kernel is not None
        assert jnp.array_equal(kernel.active_dims, active_dims)

    def test_dimension_selection(self):
        """Test that kernel only uses specified dimensions."""
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

    def test_with_matrix_inputs(self):
        """Test ActiveDimsKernel with matrix inputs."""
        base_kernel = SEKernel(length_scale=1.0)
        active_dims = jnp.array([1, 3])
        kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

        # Create inputs with 5 dimensions
        n_points = 4
        n_dims = 5
        x1 = jnp.ones((n_points, n_dims))
        x2 = jnp.ones((n_points, n_dims)) * 2.0

        result = kernel(x1, x2)

        # Should produce covariance matrix
        assert result.shape == (n_points, n_points)
        assert jnp.all(jnp.isfinite(result))

    def test_single_dimension(self):
        """Test ActiveDimsKernel with single active dimension."""
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

    def test_instantiation(self):
        """Test that ARDKernel can be instantiated."""
        base_kernel = SEKernel(length_scale=1.0)
        length_scales = jnp.array([1.0, 2.0, 0.5])
        kernel = ARDKernel(base_kernel, length_scales=length_scales)

        assert kernel.inner_kernel is not None
        assert jnp.array_equal(kernel.length_scales, length_scales)

    def test_different_scales_per_dimension(self):
        """Test that ARD applies different length scales per dimension."""
        base_kernel = SEKernel(length_scale=1.0)

        # Different relevance for each dimension
        length_scales = jnp.array([1.0, 0.1, 10.0])  # middle dim most relevant
        kernel = ARDKernel(base_kernel, length_scales=length_scales)

        # Create inputs
        x1 = jnp.array([[0.0, 0.0, 0.0]])
        x2 = jnp.array([[1.0, 1.0, 1.0]])

        result = kernel(x1, x2)

        # Manually compute ARD result
        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        base_kernel_unit = SEKernel(length_scale=1.0)
        expected = base_kernel_unit(scaled_x1, scaled_x2)

        assert jnp.allclose(result, expected, rtol=1e-5)
        assert jnp.isfinite(result)

    def test_isotropic_equivalence(self):
        """Test that uniform length scales give isotropic kernel."""
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

    def test_matrix_inputs(self):
        """Test ARDKernel with matrix inputs."""
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

    def test_relevance_interpretation(self):
        """Test that smaller length scales indicate higher relevance."""
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

    def test_ard_with_active_dims(self):
        """Test combining ARD and ActiveDims wrappers."""
        base_kernel = SEKernel(length_scale=1.0)

        # First select dimensions, then apply ARD
        active_dims = jnp.array([0, 2, 4])
        active_kernel = ActiveDimsKernel(base_kernel, active_dims=active_dims)

        length_scales = jnp.array([1.0, 0.5, 2.0])  # For the 3 active dims
        ard_kernel = ARDKernel(active_kernel, length_scales=length_scales)

        # Create 5D input
        x1 = jnp.ones((1, 5))
        x2 = jnp.ones((1, 5)) * 2.0

        result = ard_kernel(x1, x2)

        assert jnp.isfinite(result)
        assert result.shape == ()  # Scalar output

    def test_batch_with_ard(self):
        """Test combining Batch and ARD wrappers."""
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
            batch_over_inputs=True
        )

        x_batched = jnp.array([
            [[1.0, 2.0]],
            [[1.5, 2.5]],
            [[2.0, 3.0]]
        ])

        result = batch_kernel(x_batched, x_batched)

        assert result.shape == (batch_size, 1, 1)
        assert jnp.all(jnp.isfinite(result))