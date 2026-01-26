from functools import partial

import equinox as eqx
from equinox import filter_jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import jit, vmap
from jax.lax import cond

from kernax import AbstractKernel, ConstantKernel, StaticAbstractKernel


class WrapperKernel(AbstractKernel):
	"""Class for kernels that perform some operation on the output of another "inner" kernel."""

	inner_kernel: AbstractKernel = eqx.field()

	def __init__(self, inner_kernel=None):
		"""
		Instantiates a wrapper kernel with the given inner kernel.

		:param inner_kernel: the inner kernel to wrap
		"""
		# If the inner kernel is not a kernel, we try to convert it to a ConstantKernel
		if not isinstance(inner_kernel, AbstractKernel):
			inner_kernel = ConstantKernel(value=inner_kernel)

		self.inner_kernel = inner_kernel


class StaticDiagKernel(StaticAbstractKernel):
	"""
	Static kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""

	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		return cond(
			jnp.all(x1 == x2), lambda _: kern.inner_kernel(x1, x2), lambda _: jnp.array(0.0), None
		)


class DiagKernel(WrapperKernel):
	"""
	Kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""

	static_class = StaticDiagKernel

	def __init__(self, inner_kernel=None):
		super().__init__(inner_kernel=inner_kernel)

	def __str__(self):
		return f"Diag({self.inner_kernel})"


class ExpKernel(WrapperKernel):
	"""
	Kernel that applies the exponential operator to the output of another kernel.
	"""

	@jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return jnp.exp(self.inner_kernel(x1, x2))

	def __str__(self):
		return f"Exp({self.inner_kernel})"


class LogKernel(WrapperKernel):
	"""
	Kernel that applies the logarithm operator to the output of another kernel.
	"""

	@jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return jnp.log(self.inner_kernel(x1, x2))

	def __str__(self):
		return f"Log({self.inner_kernel})"


class NegKernel(WrapperKernel):
	@jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return -self.inner_kernel(x1, x2)

	def __str__(self):
		return f"- {self.inner_kernel}"


class BatchKernel(WrapperKernel):
	"""
	Wrapper kernel to add batch handling to any kernel.

	A basic kernel usually works on inputs of shape (N, I), and produces covariance matrices of shape (N, N).

	Wrapped inside a batch kernel, they can either:
	- still work on inputs of shape (N, I), but produce covariance matrices of shape (B, N, N), where B is the batch size. This is useful when the hyperparameters are batched, i.e. each batch element has its own set of hyperparameters.
	- or work on inputs of shape (B, N, I), producing covariance matrices of shape (B, N, N). This is useful when the inputs are batched, regardless of whether the hyperparameters are batched or not.

	A batch kernel can itself be wrapped inside another batch kernel, to handle multiple batch dimensions/hyperparameter sets.

	This class uses vmap to vectorize the kernel computation over the batch dimension.
	"""

	inner_kernel: AbstractKernel = eqx.field()
	batch_in_axes: bool = eqx.field(static=True)
	batch_over_inputs: int | None = eqx.field(static=True)

	def __init__(self, inner_kernel, batch_size, batch_in_axes=None, batch_over_inputs=True):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param batch_size: the size of the batch (int)
		:param batch_in_axes: a value or pytree indicating which hyperparameters are batched (0)
												   or shared (None) across the batch.
												   If None, all hyperparameters are assumed to be shared across the batch.
												   If 0, all hyperparameters are assumed to be batched across the batch.
												   If a pytree, it must have the same structure as inner_kernel, with hyperparameter
												   leaves being either 0 (batched) or None (shared).
		:param batch_over_inputs: whether to expect inputs of shape (B, N, I) (True) or (N, I) (False)
		"""
		# Initialize the WrapperKernel
		super().__init__(inner_kernel=inner_kernel)

		# TODO: batch_size isn't needed if hyperparameters are shared
		# TODO: explicit error message when batch_in_axes is all None and batch_over_inputs is False, as that makes vmap (and a Batch Kernel) useless
		# Default: all array hyperparameters are shared (None for all array leaves)
		if batch_in_axes is None:
			# Extract only array leaves and map them to None
			self.batch_in_axes = jtu.tree_map(lambda _: None, inner_kernel)
		elif batch_in_axes == 0:
			# All hyperparameters are batched
			self.batch_in_axes = jtu.tree_map(lambda _: 0, inner_kernel)
		else:
			self.batch_in_axes = batch_in_axes

		self.batch_over_inputs = 0 if batch_over_inputs else None

		# Add batch dimension to parameters where batch_in_axes is 0
		self.inner_kernel = jtu.tree_map(
			lambda param, batch_in_ax: (
				param if batch_in_ax is None else jnp.repeat(param[None, ...], batch_size, axis=0)
			),
			self.inner_kernel,
			self.batch_in_axes,
		)

	@jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		"""
		Compute the kernel over batched inputs using vmap.

		Args:
				x1: Input of shape (B, ..., N, I)
				x2: Optional second input of shape (B, ..., M, I)

		Returns:
				Kernel matrix of appropriate shape with batch dimension
		"""
		# vmap over the batch dimension of inner_kernel and inputs
		# Each batch element gets its own version of inner_kernel with corresponding hyperparameters
		return vmap(
			lambda kernel, x1, x2: kernel(x1, x2),
			in_axes=(
				self.batch_in_axes,
				self.batch_over_inputs,
				self.batch_over_inputs if x2 is not None else None,
			),
		)(self.inner_kernel, x1, x2)

	def __str__(self):
		# just str of the inner kernel, as the batch info is in the parameters of the inner kernel
		return f"{self.inner_kernel}"


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
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		"""
		Compute the kernel over batched inputs using vmap.

		Args:
				x1: Input of shape (B, ..., N, I)
				x2: Optional second input of shape (B, ..., M, I)

		Returns:
				Kernel block-matrix of appropriate shape
		"""
		x2 = x1 if x2 is None else x2

		rows, cols = jnp.triu_indices(self.nb_blocks)

		full_kernel = jtu.tree_map(
			lambda param, block_in_ax:
			param[rows] if block_in_ax == 0 else param[cols] if block_in_ax == 1 else param,
			self.inner_kernel,
			self.block_in_axes,
		)

		x1 = x1[rows] if self.block_over_inputs == 0 else x1
		x2 = x2[cols] if self.block_over_inputs == 0 else x2

		# vmap over the block dimension of inner_kernel and inputs
		# Each block element gets its own version of inner_kernel with corresponding hyperparameters
		return self.symmetric_blocks_to_matrix(vmap(
			lambda kernel, x1, x2: kernel(x1, x2),
			in_axes=(
				jtu.tree_map(lambda x: None if x is None else 0, self.block_in_axes),
				self.block_over_inputs,
				self.block_over_inputs,
			),
		)(full_kernel, x1, x2))

	@filter_jit
	def symmetric_blocks_to_matrix(self, flat_blocks):
		"""
		Rebuilds a symmetric matrix from its unique blocks (upper triangle).

		Args:
			flat_blocks: Tensor (T, H, W) where T is a triangular number.
						 Expected order: (0,0), (0,1), (0,2)... (row-major upper)
		"""
		# 2. Créer la grille de blocs vide (B, B, H, W)
		grid = jnp.zeros((self.nb_blocks, self.nb_blocks, flat_blocks.shape[-2], flat_blocks.shape[-1]), dtype=flat_blocks.dtype)

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
		return (grid
		        .swapaxes(1, 2)  # (RowBlock, H, ColBlock, W)
		        .reshape(self.nb_blocks * flat_blocks.shape[-2], self.nb_blocks * flat_blocks.shape[-1]))

	def __str__(self):
		return f"Block{self.inner_kernel}"


class BlockDiagKernel(BatchKernel):
	"""
	Wrapper kernel to build block-diagonal covariance matrices using any kernel.

	A basic kernel usually works on inputs of shape (N, I), and produces covariance matrices of shape (N, N).

	Wrapped inside a block-diagonal kernel, they can either:
	- still work on inputs of shape (N, I), but produce covariance matrices of shape (B*N, B*N), where B is the number of blocks. This is useful when the hyperparameters are distinct to blocks, i.e. each sub-matrix has its own set of hyperparameters.
	- or work on inputs of shape (B, N, I), producing covariance matrices of shape (B*N, B*N). This is useful when inputs are different for each block, regardless of whether the hyperparameters are shared between blocks or not.

	This class uses vmap to vectorize the kernel computation of each block, then resize the result into a block matrix.
	"""
	def __init__(self, inner_kernel, nb_blocks, block_in_axes=None, block_over_inputs=True):
		super().__init__(inner_kernel, nb_blocks, block_in_axes, block_over_inputs)


	@filter_jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		"""
		Compute the kernel over batched inputs using vmap.

		Args:
				x1: Input of shape (B, ..., N, I)
				x2: Optional second input of shape (B, ..., M, I)

		Returns:
				Kernel block-matrix of appropriate shape
		"""
		return jsp.linalg.block_diag(*super().__call__(x1, x2))

	def __str__(self):
		return f"BlockDiag{self.inner_kernel}"


class ActiveDimsKernel(WrapperKernel):
	"""
	Wrapper kernel to select active dimensions from the inputs before passing them to the inner kernel.

	NOTE: This kernel *must* be the outer-most kernel (aka it shouldn't be wrapped inside another one)
	If you use a kernel that has HPs specific to *input dimensions* (like an ARDKernel), make sure you instantiate it
	with HPs only for the active dimensions. For example, on inputs of dimension 5 with 3 active dimensions:

	```
	# First, define ARD
	length_scales = jnp.array([1.0, 0.5, 2.0])  # Defined only on 3 dims, as we later use ARD!
	ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

	# ActiveDims must always be the outer-most kernel
	active_dims = jnp.array([0, 2, 4])
	active_kernel = ActiveDimsKernel(ard_kernel, active_dims=active_dims)
	```
	"""

	active_dims: jnp.ndarray = eqx.field(static=True, converter=jnp.asarray)

	def __init__(self, inner_kernel, active_dims):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param active_dims: the indices of the active dimensions to select from the inputs (1D array of integers)
		"""
		super().__init__(inner_kernel=inner_kernel)
		self.active_dims = active_dims

	@jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		# TODO: add runtime error if active_dims doesn't match input dimensions
		if x2 is None:
			x2 = x1

		return self.inner_kernel(x1[..., self.active_dims], x2[..., self.active_dims])


class ARDKernel(WrapperKernel):
	"""
	Wrapper kernel to apply Automatic Relevance Determination (ARD) to the inputs before passing them to the inner kernel.
	Each input dimension is scaled by a separate length scale hyperparameter.
	"""

	length_scales: jnp.ndarray = eqx.field(converter=jnp.asarray)

	def _freeze_inner_lengthscales(self, kernel):
		def map_func(path, leaf):
			if len(path) > 0:
				last_node = path[-1] # To retrieve the attribute name
				if isinstance(last_node, jtu.GetAttrKey) and last_node.name == "length_scale":
					# Force length scale of 1
					return jnp.ones_like(leaf)
			return leaf

		return jtu.tree_map_with_path(map_func, kernel)

	def __init__(self, inner_kernel, length_scales):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param length_scales: the length scales for each input dimension (1D array of floats)
		"""
		super().__init__(inner_kernel=inner_kernel)
		self.length_scales = length_scales

	@jit
	def __call__(self, x1: jnp.ndarray, x2: None | jnp.ndarray = None) -> jnp.ndarray:
		# TODO: add runtime error if length_scales doesn't match input dimensions
		if x2 is None:
			x2 = x1

		# FIXME: if used in an optimisation setting, inner length_scales can still have other values.
		#  Freezing the inner at every call may be costly/slow down learning.
		#  We should find a proper way to freeze these parameters.
		return self._freeze_inner_lengthscales(self.inner_kernel)(x1 / self.length_scales, x2 / self.length_scales)
