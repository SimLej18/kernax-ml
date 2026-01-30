import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import filter_jit
from jax import Array, vmap

from ..AbstractKernel import AbstractKernel
from .WrapperKernel import WrapperKernel


class BlockKernel(WrapperKernel):
	"""
	Wrapper kernel to build block covariance matrices using any kernel.

	A basic kernel usually works on inputs of shape (N, I), and produces covariance matrices of shape (N, N).

	Wrapped inside a block kernel, they can either:
	- still work on inputs of shape (N, I), but produce covariance matrices of shape (B*N, B*N), where B is the number of blocks. This is useful when the hyperparameters are distinct to blocks, i.e. each sub-matrix has its own set of hyperparameters.
	- or work on inputs of shape (B, N, I), producing covariance matrices of shape (B*N, B*N). This is useful when inputs are different for each block, regardless of whether the hyperparameters are shared between blocks or not.

	This class uses vmap to vectorize the kernel computation of each block, then resize the result into a block matrix.
	"""

	inner_kernel: AbstractKernel = eqx.field()
	nb_blocks: int = eqx.field(static=True)
	block_in_axes: bool = eqx.field(static=True)
	block_over_inputs: int | None = eqx.field(static=True)

	def __init__(self, inner_kernel, nb_blocks, block_in_axes=None, block_over_inputs=True):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param nb_blocks: the number of blocks
		:param block_in_axes: a pytree indicating which hyperparameters change across blocks.
								If 0, the hyperparameter changes across the columns of the block matrix.
								If 1, the hyperparameter changes across the rows of the block matrix.
								If None, the hyperparameter is shared across all blocks.
								To compute the block matrix, the kernel needs to have at least one of its hyperparameters changing across rows and one across columns.
		:param block_over_inputs: whether to expect inputs of shape (B, N, I) (True) or (N, I) (False)

		N.b: the result of this kernel is not always a valid covariance matrix! For example, an RBF kernel with a varying lengthscale across rows and a varying amplitude across column will not produce a symmetric matrix, hence giving an invalid covariance matrix.
		Usually, you want to use this kernel with an appropriate inner_kernel, calculating a function where two hyper-parameters have symmetric roles.
		A good example is a multi-output (convolutional) kernel in GPs, which usually have two distinct lengthscales (and variances) depending on which output dimension is considered.
		"""
		# Initialize the WrapperKernel
		super().__init__(inner_kernel=inner_kernel)

		# TODO: explicit error message when nb_blocks is 1, as vmap is not needed then
		# TODO: check that at least one hyperparameter varies across rows and one across columns

		self.nb_blocks = nb_blocks

		# Default: all array hyperparameters are shared (None for all array leaves)
		if block_in_axes is None:
			# Extract only array leaves and map them to None
			self.block_in_axes = jtu.tree_map(lambda _: None, inner_kernel)
		else:
			self.block_in_axes = block_in_axes

		self.block_over_inputs = 0 if block_over_inputs else None

		# Add block dimension to parameters where batch_in_axes is 0
		self.inner_kernel = jtu.tree_map(
			lambda param, block_in_ax: (
				param if block_in_ax is None else jnp.repeat(param[None, ...], nb_blocks, axis=0)
			),
			self.inner_kernel,
			self.block_in_axes,
		)

	@filter_jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		"""
		Compute the kernel over batched inputs using vmap.

		Args:
				x1: Input of shape (B, ..., N, I)
				x2: Optional second input of shape (B, ..., M, I)

		Returns:
				Kernel block-matrix of appropriate shape
		"""
		x2 = x1 if x2 is None else x2

		# Check if we can use vmap (at least one axis is not None)
		can_use_vmap = (
			not jtu.tree_all(jtu.tree_map(lambda k: k is None, self.block_in_axes))
			or self.block_over_inputs is not None
		)

		if can_use_vmap:
			# Use vmap when we have blocked hyperparameters or blocked inputs
			rows, cols = jnp.triu_indices(self.nb_blocks)

			full_kernel = jtu.tree_map(
				lambda param, block_in_ax: param[rows]
				if block_in_ax == 0
				else param[cols]
				if block_in_ax == 1
				else param,
				self.inner_kernel,
				self.block_in_axes,
			)

			x1_indexed = x1[rows] if self.block_over_inputs == 0 else x1
			x2_indexed = x2[cols] if self.block_over_inputs == 0 else x2

			# vmap over the block dimension of inner_kernel and inputs
			# Each block element gets its own version of inner_kernel with corresponding hyperparameters
			return self.symmetric_blocks_to_matrix(  # type: ignore[no-any-return]
				vmap(
					lambda kernel, x1, x2: kernel(x1, x2),
					in_axes=(
						jtu.tree_map(lambda x: None if x is None else 0, self.block_in_axes),
						self.block_over_inputs,
						self.block_over_inputs,
					),
				)(full_kernel, x1_indexed, x2_indexed)
			)
		else:
			# All hyperparameters and inputs are shared: create a block matrix with identical blocks
			single_block = self.inner_kernel(x1, x2)
			# Tile the block to create the full block matrix
			return jnp.tile(single_block, (self.nb_blocks, self.nb_blocks))

	@filter_jit
	def symmetric_blocks_to_matrix(self, flat_blocks):
		"""
		Rebuilds a symmetric matrix from its unique blocks (upper triangle).

		Args:
			flat_blocks: Tensor (T, H, W) where T is a triangular number.
						 Expected order: (0,0), (0,1), (0,2)... (row-major upper)
		"""
		# 2. Créer la grille de blocs vide (B, B, H, W)
		grid = jnp.zeros(
			(self.nb_blocks, self.nb_blocks, flat_blocks.shape[-2], flat_blocks.shape[-1]),
			dtype=flat_blocks.dtype,
		)

		# 3. Récupérer les indices du triangle supérieur
		# ex: rows=[0,0,1], cols=[0,1,1] pour B=2
		rows, cols = jnp.triu_indices(self.nb_blocks)

		# 4. Remplir le triangle supérieur
		grid = grid.at[rows, cols].set(flat_blocks)

		# 5. Remplir le triangle inférieur par symétrie
		# On prend le bloc en (rows, cols), on le transpose (swapaxes -1, -2)
		# et on le place en (cols, rows).
		# Note : Sur la diagonale (rows==cols), cela transpose le bloc sur lui-même,
		# ce qui est correct car un bloc diagonal d'une matrice symétrique est lui-même symétrique.
		grid = grid.at[cols, rows].set(grid[rows, cols].swapaxes(-1, -2))

		# 6. Assemblage final (Même logique que précédemment)
		return grid.swapaxes(1, 2).reshape(  # (RowBlock, H, ColBlock, W)
			self.nb_blocks * flat_blocks.shape[-2], self.nb_blocks * flat_blocks.shape[-1]
		)

	def __str__(self):
		return f"Block{self.inner_kernel}"
