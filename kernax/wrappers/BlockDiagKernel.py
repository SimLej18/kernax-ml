import jax.scipy as jsp
from equinox import filter_jit
from jax import Array

from .BatchKernel import BatchKernel


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
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:  # type: ignore[override]
		"""
		Compute the kernel over batched inputs using vmap.

		Args:
				x1: Input of shape (B, ..., N, I)
				x2: Optional second input of shape (B, ..., M, I)

		Returns:
				Kernel block-matrix of appropriate shape
		"""
		return jsp.linalg.block_diag(*super().__call__(x1, x2))  # type: ignore[no-any-return]

	def __str__(self):
		return f"BlockDiag{self.inner_kernel}"
