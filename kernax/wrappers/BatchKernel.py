import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, jit, vmap

from ..AbstractKernel import AbstractKernel
from .WrapperKernel import WrapperKernel


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
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
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
		return vmap(  # type: ignore[no-any-return]
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
