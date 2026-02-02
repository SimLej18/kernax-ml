"""
Tests for wrapper kernels (BatchKernel, ActiveDimsKernel, ARDKernel).
"""

import allure
import jax.numpy as jnp

from kernax import (
	ActiveDimsKernel,
	ARDKernel,
	BatchKernel,
	BlockDiagKernel,
	BlockKernel,
	SEKernel,
)


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

	@allure.title("BatchKernel with shared hyperparameters and shared inputs")
	@allure.description(
		"Test BatchKernel with batch_in_axes=None and batch_over_inputs=False. "
		"All batch matrices should be identical since same HPs and inputs are used."
	)
	def test_shared_hyperparameters_shared_inputs(self):
		base_kernel = SEKernel(length_scale=1.0)
		batch_size = 4

		# Shared hyperparameters AND shared inputs
		batch_kernel = BatchKernel(
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


class TestBlockKernel:
	"""Tests for BlockKernel wrapper."""

	@allure.title("BlockKernel Instantiation")
	@allure.description("Test that BlockKernel can be instantiated.")
	def test_instantiation(self):
		import jax.tree_util as jtu

		base_kernel = SEKernel(length_scale=1.0)
		# Create a pytree with block_in_axes=0 for all hyperparameters
		block_in_axes = jtu.tree_map(lambda _: 0, base_kernel)
		block_kernel = BlockKernel(
			base_kernel, nb_blocks=3, block_in_axes=block_in_axes, block_over_inputs=True
		)
		assert block_kernel.inner_kernel is not None
		assert block_kernel.nb_blocks == 3

	@allure.title("BlockKernel block over hyperparameters")
	@allure.description("Test blocking with distinct hyperparameters per block.")
	def test_block_over_hyperparameters(self):
		import jax.tree_util as jtu

		# Create base kernel with single length_scale
		base_kernel = SEKernel(length_scale=1.0)
		nb_blocks = 3

		# Create a pytree where hyperparameters vary across rows (0) and columns (1)
		# For a valid block covariance matrix, we need different HPs for rows and cols
		block_in_axes = jtu.tree_map(lambda _: 0, base_kernel)  # Vary across rows

		# Wrap in BlockKernel to handle blocked hyperparameters
		block_kernel = BlockKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=block_in_axes,
			block_over_inputs=False,  # Same inputs for all blocks
		)

		# Create non-blocked inputs
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5]])

		# Compute covariance - should produce block matrix
		result = block_kernel(x1, x2)

		# Result should be a block matrix of shape (nb_blocks*N, nb_blocks*M)
		expected_shape = (nb_blocks * x1.shape[0], nb_blocks * x2.shape[0])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BlockKernel block over inputs and hyperparameters")
	@allure.description("Test blocking over both inputs and hyperparameters.")
	def test_block_over_inputs_and_hyperparameters(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x1_batched, x2_batched = sample_batched_data
		nb_blocks = x1_batched.shape[0]

		# Use a pytree to specify different axes for different hyperparameters
		import jax.tree_util as jtu

		block_in_axes = jtu.tree_map(lambda _: 0, base_kernel)

		block_kernel = BlockKernel(
			base_kernel, nb_blocks=nb_blocks, block_in_axes=block_in_axes, block_over_inputs=True
		)

		result = block_kernel(x1_batched, x1_batched)

		# Should produce block matrix of shape (nb_blocks*N, nb_blocks*N)
		expected_shape = (nb_blocks * x1_batched.shape[1], nb_blocks * x1_batched.shape[1])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BlockKernel block over inputs only")
	@allure.description("Test blocking over inputs with shared hyperparameters.")
	def test_block_over_inputs_only(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		nb_blocks = x_batched.shape[0]

		# Block over inputs but share hyperparameters
		block_kernel = BlockKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,  # Shared hyperparameters
			block_over_inputs=True,
		)

		result = block_kernel(x_batched, x_batched)

		expected_shape = (nb_blocks * x_batched.shape[1], nb_blocks * x_batched.shape[1])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BlockKernel mathematical properties")
	@allure.description("Test that block matrix is symmetric.")
	def test_math_properties(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		nb_blocks = x_batched.shape[0]

		block_kernel = BlockKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,
			block_over_inputs=True,
		)

		result = block_kernel(x_batched, x_batched)

		# Check matrix is symmetric (with numerical tolerance)
		assert jnp.allclose(result, result.T, rtol=1e-5)

		# Check all values are positive (for SE kernel with identical x1=x2)
		assert jnp.all(result >= 0)

	@allure.title("BlockKernel block structure verification")
	@allure.description("Test that result has correct block structure.")
	def test_block_structure(self):
		base_kernel = SEKernel(length_scale=1.0)
		nb_blocks = 2
		n_points = 3

		# Create batched inputs with distinct patterns
		x_batched = jnp.array([[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]])

		block_kernel = BlockKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,
			block_over_inputs=True,
		)

		result = block_kernel(x_batched, x_batched)

		# Extract blocks manually
		block_00 = result[:n_points, :n_points]
		block_01 = result[:n_points, n_points:]
		block_10 = result[n_points:, :n_points]
		block_11 = result[n_points:, n_points:]

		# Verify blocks are computed correctly
		expected_00 = base_kernel(x_batched[0], x_batched[0])
		expected_11 = base_kernel(x_batched[1], x_batched[1])
		expected_01 = base_kernel(x_batched[0], x_batched[1])
		expected_10 = base_kernel(x_batched[1], x_batched[0])

		assert jnp.allclose(block_00, expected_00)
		assert jnp.allclose(block_11, expected_11)
		assert jnp.allclose(block_01, expected_01)
		assert jnp.allclose(block_10, expected_10)

	@allure.title("BlockKernel with shared hyperparameters and shared inputs")
	@allure.description(
		"Test BlockKernel with block_in_axes=None and block_over_inputs=False. "
		"All blocks should be identical since same HPs and inputs are used."
	)
	def test_shared_hyperparameters_shared_inputs(self):
		base_kernel = SEKernel(length_scale=1.0)
		nb_blocks = 3
		n_points = 4

		# Shared hyperparameters AND shared inputs
		block_kernel = BlockKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,  # Shared hyperparameters
			block_over_inputs=False,  # Shared inputs
		)

		# Non-blocked inputs
		x1 = jnp.array([[1.0], [2.0], [3.0], [4.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5], [4.5]])

		result = block_kernel(x1, x2)

		# Result should be a block matrix of shape (nb_blocks*N, nb_blocks*M)
		expected_shape = (nb_blocks * x1.shape[0], nb_blocks * x2.shape[0])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

		# Compute the expected block (same for all blocks)
		expected_block = base_kernel(x1, x2)

		# All blocks should be identical (same HPs + same inputs)
		for i in range(nb_blocks):
			for j in range(nb_blocks):
				start_i = i * n_points
				end_i = (i + 1) * n_points
				start_j = j * n_points
				end_j = (j + 1) * n_points

				block_ij = result[start_i:end_i, start_j:end_j]
				assert jnp.allclose(
					block_ij, expected_block, rtol=1e-6
				), f"Block ({i},{j}) differs from expected"

		# Verify that all blocks are identical to block (0,0)
		block_00 = result[:n_points, :n_points]
		for i in range(nb_blocks):
			for j in range(nb_blocks):
				start_i = i * n_points
				end_i = (i + 1) * n_points
				start_j = j * n_points
				end_j = (j + 1) * n_points

				block_ij = result[start_i:end_i, start_j:end_j]
				assert jnp.allclose(
					block_ij, block_00, rtol=1e-6
				), f"Block ({i},{j}) differs from block (0,0)"


class TestBlockDiagKernel:
	"""Tests for BlockDiagKernel wrapper."""

	@allure.title("BlockDiagKernel Instantiation")
	@allure.description("Test that BlockDiagKernel can be instantiated.")
	def test_instantiation(self):
		import jax.tree_util as jtu

		base_kernel = SEKernel(length_scale=1.0)
		# Create a pytree with block_in_axes=0 for all hyperparameters
		block_in_axes = jtu.tree_map(lambda _: 0, base_kernel)
		block_diag_kernel = BlockDiagKernel(
			base_kernel, nb_blocks=3, block_in_axes=block_in_axes, block_over_inputs=True
		)
		assert block_diag_kernel.inner_kernel is not None
		assert block_diag_kernel.batch_over_inputs == 0

	@allure.title("BlockDiagKernel block over hyperparameters")
	@allure.description("Test block-diagonal with distinct hyperparameters per block.")
	def test_block_over_hyperparameters(self):
		import jax.tree_util as jtu

		# Create base kernel with single length_scale
		base_kernel = SEKernel(length_scale=1.0)
		nb_blocks = 3

		# Create a pytree where hyperparameters are batched (0 for all)
		block_in_axes = jtu.tree_map(lambda _: 0, base_kernel)

		# Wrap in BlockDiagKernel to handle blocked hyperparameters
		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=block_in_axes,
			block_over_inputs=False,  # Same inputs for all blocks
		)

		# Create non-blocked inputs
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5]])

		# Compute covariance - should produce block-diagonal matrix
		result = block_diag_kernel(x1, x2)

		# Result should be a block-diagonal matrix of shape (nb_blocks*N, nb_blocks*M)
		expected_shape = (nb_blocks * x1.shape[0], nb_blocks * x2.shape[0])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BlockDiagKernel block over inputs and hyperparameters")
	@allure.description("Test block-diagonal over both inputs and hyperparameters.")
	def test_block_over_inputs_and_hyperparameters(self, sample_batched_data):
		import jax.tree_util as jtu

		base_kernel = SEKernel(length_scale=1.0)
		x1_batched, x2_batched = sample_batched_data
		nb_blocks = x1_batched.shape[0]

		# Create a pytree where hyperparameters are batched
		block_in_axes = jtu.tree_map(lambda _: 0, base_kernel)

		block_diag_kernel = BlockDiagKernel(
			base_kernel, nb_blocks=nb_blocks, block_in_axes=block_in_axes, block_over_inputs=True
		)

		result = block_diag_kernel(x1_batched, x1_batched)

		# Should produce block-diagonal matrix
		expected_shape = (nb_blocks * x1_batched.shape[1], nb_blocks * x1_batched.shape[1])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BlockDiagKernel block over inputs only")
	@allure.description("Test block-diagonal over inputs with shared hyperparameters.")
	def test_block_over_inputs_only(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		nb_blocks = x_batched.shape[0]

		# Block over inputs but share hyperparameters
		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,  # Shared hyperparameters
			block_over_inputs=True,
		)

		result = block_diag_kernel(x_batched, x_batched)

		expected_shape = (nb_blocks * x_batched.shape[1], nb_blocks * x_batched.shape[1])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

	@allure.title("BlockDiagKernel diagonal structure")
	@allure.description("Test that matrix is truly block-diagonal (zeros off blocks).")
	def test_diagonal_structure(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		nb_blocks = x_batched.shape[0]
		n_points = x_batched.shape[1]

		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,
			block_over_inputs=True,
		)

		result = block_diag_kernel(x_batched, x_batched)

		# Extract diagonal blocks
		for i in range(nb_blocks):
			start_i = i * n_points
			end_i = (i + 1) * n_points
			diag_block = result[start_i:end_i, start_i:end_i]

			# Diagonal blocks should not be all zeros
			assert not jnp.allclose(diag_block, 0.0)

			# Off-diagonal blocks should be all zeros
			for j in range(nb_blocks):
				if i != j:
					start_j = j * n_points
					end_j = (j + 1) * n_points
					off_diag_block = result[start_i:end_i, start_j:end_j]
					assert jnp.allclose(off_diag_block, 0.0, atol=1e-6)

	@allure.title("BlockDiagKernel mathematical properties")
	@allure.description("Test that block-diagonal matrix is symmetric.")
	def test_math_properties(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		nb_blocks = x_batched.shape[0]

		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,
			block_over_inputs=True,
		)

		result = block_diag_kernel(x_batched, x_batched)

		# Check matrix is symmetric
		assert jnp.allclose(result, result.T)

		# Check all diagonal values are positive
		assert jnp.all(jnp.diag(result) >= 0)

	@allure.title("BlockDiagKernel individual blocks verification")
	@allure.description("Test that each diagonal block is computed correctly.")
	def test_individual_blocks(self):
		base_kernel = SEKernel(length_scale=1.0)
		nb_blocks = 3
		n_points = 4

		# Create batched inputs with distinct patterns
		x_batched = jnp.array(
			[
				[[1.0], [2.0], [3.0], [4.0]],
				[[10.0], [20.0], [30.0], [40.0]],
				[[100.0], [200.0], [300.0], [400.0]],
			]
		)

		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,
			block_over_inputs=True,
		)

		result = block_diag_kernel(x_batched, x_batched)

		# Verify each diagonal block matches the expected kernel computation
		for i in range(nb_blocks):
			start_i = i * n_points
			end_i = (i + 1) * n_points
			diag_block = result[start_i:end_i, start_i:end_i]

			expected_block = base_kernel(x_batched[i], x_batched[i])
			assert jnp.allclose(diag_block, expected_block)

	@allure.title("BlockDiagKernel comparison with BatchKernel")
	@allure.description("Test that BlockDiagKernel produces same diagonal blocks as BatchKernel.")
	def test_comparison_with_batch_kernel(self, sample_batched_data):
		base_kernel = SEKernel(length_scale=1.0)
		x_batched, _ = sample_batched_data
		nb_blocks = x_batched.shape[0]
		n_points = x_batched.shape[1]

		# Create both kernels with same configuration
		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,
			block_over_inputs=True,
		)

		batch_kernel = BatchKernel(
			base_kernel,
			batch_size=nb_blocks,
			batch_in_axes=None,
			batch_over_inputs=True,
		)

		block_diag_result = block_diag_kernel(x_batched, x_batched)
		batch_result = batch_kernel(x_batched, x_batched)

		# Extract diagonal blocks from BlockDiagKernel and compare with BatchKernel output
		for i in range(nb_blocks):
			start_i = i * n_points
			end_i = (i + 1) * n_points
			diag_block = block_diag_result[start_i:end_i, start_i:end_i]

			assert jnp.allclose(diag_block, batch_result[i])

	@allure.title("BlockDiagKernel with shared hyperparameters and shared inputs")
	@allure.description(
		"Test BlockDiagKernel with block_in_axes=None and block_over_inputs=False. "
		"All diagonal blocks should be identical since same HPs and inputs are used."
	)
	def test_shared_hyperparameters_shared_inputs(self):
		base_kernel = SEKernel(length_scale=1.0)
		nb_blocks = 3
		n_points = 4

		# Shared hyperparameters AND shared inputs
		block_diag_kernel = BlockDiagKernel(
			base_kernel,
			nb_blocks=nb_blocks,
			block_in_axes=None,  # Shared hyperparameters
			block_over_inputs=False,  # Shared inputs
		)

		# Non-blocked inputs
		x1 = jnp.array([[1.0], [2.0], [3.0], [4.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5], [4.5]])

		result = block_diag_kernel(x1, x2)

		# Result should be a block-diagonal matrix of shape (nb_blocks*N, nb_blocks*M)
		expected_shape = (nb_blocks * x1.shape[0], nb_blocks * x2.shape[0])
		assert result.shape == expected_shape
		assert jnp.all(jnp.isfinite(result))

		# Compute the expected diagonal block (same for all diagonal blocks)
		expected_block = base_kernel(x1, x2)

		# Verify all diagonal blocks are identical
		for i in range(nb_blocks):
			start_i = i * n_points
			end_i = (i + 1) * n_points

			diag_block = result[start_i:end_i, start_i:end_i]
			assert jnp.allclose(
				diag_block, expected_block, rtol=1e-6
			), f"Diagonal block {i} differs from expected"

		# Verify all diagonal blocks are identical to the first one
		first_diag_block = result[:n_points, :n_points]
		for i in range(1, nb_blocks):
			start_i = i * n_points
			end_i = (i + 1) * n_points
			diag_block = result[start_i:end_i, start_i:end_i]

			assert jnp.allclose(
				diag_block, first_diag_block, rtol=1e-6
			), f"Diagonal block {i} differs from first diagonal block"

		# Verify off-diagonal blocks are all zeros
		for i in range(nb_blocks):
			for j in range(nb_blocks):
				if i != j:
					start_i = i * n_points
					end_i = (i + 1) * n_points
					start_j = j * n_points
					end_j = (j + 1) * n_points

					off_diag_block = result[start_i:end_i, start_j:end_j]
					assert jnp.allclose(
						off_diag_block, 0.0, atol=1e-6
					), f"Off-diagonal block ({i},{j}) is not zero"


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

	@allure.title("Commutativity of BatchKernel and BlockKernel composition")
	@allure.description(
		"Test that BlockKernel(BatchKernel(inner)) and BatchKernel(BlockKernel(inner)) "
		"produce the same result when both use shared hyperparameters and shared inputs."
	)
	def test_batch_block_commutativity(self):
		base_kernel = SEKernel(length_scale=1.0)
		batch_size = 3
		nb_blocks = 2

		# Create test inputs
		x1 = jnp.array([[1.0], [2.0], [3.0]])
		x2 = jnp.array([[1.5], [2.5], [3.5]])

		# Composition 1: BlockKernel(BatchKernel(inner))
		batch_then_block = BlockKernel(
			BatchKernel(
				base_kernel,
				batch_size=batch_size,
				batch_in_axes=None,  # Shared HPs
				batch_over_inputs=False,  # Shared inputs
			),
			nb_blocks=nb_blocks,
			block_in_axes=None,  # Shared HPs
			block_over_inputs=False,  # Shared inputs
		)

		# Composition 2: BatchKernel(BlockKernel(inner))
		block_then_batch = BatchKernel(
			BlockKernel(
				base_kernel,
				nb_blocks=nb_blocks,
				block_in_axes=None,  # Shared HPs
				block_over_inputs=False,  # Shared inputs
			),
			batch_size=batch_size,
			batch_in_axes=None,  # Shared HPs
			batch_over_inputs=False,  # Shared inputs
		)

		# Compute results
		result1 = batch_then_block(x1, x2)
		result2 = block_then_batch(x1, x2)

		# Check shapes are identical
		assert result1.shape == result2.shape, (
			f"Shapes differ: {result1.shape} vs {result2.shape}"
		)
		expected_shape = (batch_size, nb_blocks * x1.shape[0], nb_blocks * x2.shape[0])
		assert result1.shape == expected_shape, (
			f"Shape {result1.shape} doesn't match expected {expected_shape}"
		)

		# Check values are identical (commutativity)
		assert jnp.allclose(result1, result2, rtol=1e-6), (
			"BlockKernel(BatchKernel) and BatchKernel(BlockKernel) should commute "
			"when both use shared hyperparameters and shared inputs"
		)

		# Verify the structure: all batch elements should be identical
		for i in range(1, batch_size):
			assert jnp.allclose(result1[i], result1[0], rtol=1e-6), (
				f"Batch element {i} differs from batch element 0"
			)

		# Verify the block structure: all blocks should be identical
		base_result = base_kernel(x1, x2)
		n_points = x1.shape[0]
		for i in range(nb_blocks):
			for j in range(nb_blocks):
				start_i = i * n_points
				end_i = (i + 1) * n_points
				start_j = j * n_points
				end_j = (j + 1) * n_points

				block_ij = result1[0, start_i:end_i, start_j:end_j]
				assert jnp.allclose(block_ij, base_result, rtol=1e-6), (
					f"Block ({i},{j}) differs from base kernel result"
				)

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
