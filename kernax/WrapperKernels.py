from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond
import jax.tree_util as jtu
import equinox as eqx

from kernax import StaticAbstractKernel, AbstractKernel, ConstantKernel


class WrapperKernel(AbstractKernel):
	""" Class for kernels that perform some operation on the output of another "inner" kernel."""
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
		return cond(jnp.all(x1 == x2),
		            lambda _: kern.inner_kernel(x1, x2),
		            lambda _: jnp.array(0.0),
		            None)


class DiagKernel(WrapperKernel):
	"""
	Kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""
	def __init__(self, inner_kernel=None):
		super().__init__(inner_kernel=inner_kernel)
		self.static_class = StaticDiagKernel


class ExpKernel(WrapperKernel):
	"""
	Kernel that applies the exponential operator to the output of another kernel.
	"""
	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return jnp.exp(self.inner_kernel(x1, x2))


class LogKernel(WrapperKernel):
	"""
	Kernel that applies the logarithm operator to the output of another kernel.
	"""
	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return jnp.log(self.inner_kernel(x1, x2))


class NegKernel(WrapperKernel):
	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return - self.inner_kernel(x1, x2)


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
	batch_over_inputs: int|None = eqx.field(static=True)

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
			lambda param, batch_in_ax: param if batch_in_ax is None else jnp.repeat(param[None, ...], batch_size, axis=0),
			self.inner_kernel,
			self.batch_in_axes
		)

	def __call__(self, x1, x2=None):
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
			in_axes=(self.batch_in_axes, self.batch_over_inputs, self.batch_over_inputs if x2 is not None else None)
		)(self.inner_kernel, x1, x2)
