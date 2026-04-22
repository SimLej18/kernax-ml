from jax import Array, vmap
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from ..module import AbstractModule
from .WrapperModule import AbstractWrapperModule


class BatchModule(AbstractWrapperModule):
	"""
	Wrapper module to add batch handling to any module.

	A basic kernel usually works on inputs of shape (N, I), and produces covariance matrices of shape (N, N).

	Wrapped inside a batch module, they can either:
	- still work on inputs of shape (N, I), but produce covariance matrices of shape (B, N, N), where B is the batch size. This is useful when the hyperparameters are batched, i.e. each batch element has its own set of hyperparameters.
	- or work on inputs of shape (B, N, I), producing covariance matrices of shape (B, N, N). This is useful when the inputs are batched, regardless of whether the hyperparameters are batched or not.

	A batch module can itself be wrapped inside another batch module, to handle multiple batch dimensions/hyperparameter sets.

	This class uses vmap to vectorize the module computation over the batch dimension.
	"""
	inner: AbstractModule
	batch_size: int = eqx.field(static=True)
	batch_in_axes: bool = eqx.field(static=True)
	batch_over_inputs: int | None = eqx.field(static=True)

	@property
	def can_use_vmap(self):
		return not (self.batch_over_inputs is None and jtu.tree_all(
			jtu.tree_map(lambda k: k is None, self.batch_in_axes)))

	def __init__(self, inner, batch_size, batch_in_axes=None, batch_over_inputs=True):
		"""
		:param inner: the kernel to wrap, must be an instance of AbstractKernel
		:param batch_size: the size of the batch (int)
		:param batch_in_axes: a value or pytree indicating which hyperparameters are batched (0)
											   or shared (None) across the batch.
											   If None, all hyperparameters are assumed to be shared across the batch.
											   If 0, all hyperparameters are assumed to be batched across the batch.
											   If a pytree, it must have the same structure as inner, with hyperparameter
											   leaves being either 0 (batched) or None (shared).
		:param batch_over_inputs: whether to expect inputs of shape (B, N, I) (True) or (N, I) (False)
		"""
		self.batch_size = batch_size

		# Default: all array hyperparameters are shared (None for all array leaves)
		if batch_in_axes is None:
			# Extract only array leaves and map them to None
			self.batch_in_axes = jtu.tree_map(lambda _: None, inner)
		elif batch_in_axes == 0:
			# All hyperparameters are batched
			self.batch_in_axes = jtu.tree_map(lambda _: 0, inner)
		else:
			self.batch_in_axes = batch_in_axes

		self.batch_over_inputs = 0 if batch_over_inputs else None

		# Add batch dimension to parameters where batch_in_axes is 0
		self.inner = jtu.tree_map(
			lambda param, batch_in_ax: (
				param if batch_in_ax is None else jnp.repeat(param[None, ...], batch_size, axis=0)
			),
			inner,
			self.batch_in_axes,
		)

	def __call__(self, x1: Array, x2: Array | None = None, *args, **kwargs) -> Array:
		if x2 is None:
			# As BatchModule can either wrap a Kernel (might expect x2) or a Mean (doesn't expect
			# x2), we have to adapt calls to inner module depending on whether x2 was provided or
			# not.
			if self.can_use_vmap:
				return vmap(
					lambda module, x1: module(x1, *args, **kwargs),
					in_axes=(self.batch_in_axes, self.batch_over_inputs)
				)(self.inner, x1)

			# We can't use vmap
			if self.batch_size == 1:
				return self.inner(x1, *args, **kwargs)[
					None, ...]  # Add batch dimension

			# We can't use vmap but have to repeat cov n times
			return jnp.repeat(
				self.inner(x1, *args, **kwargs)[None, ...],
				self.batch_size,
				axis=0
			)

		if self.can_use_vmap:
			return vmap(
				lambda module, x1, x2: module(x1, x2, *args, **kwargs),
				in_axes=(self.batch_in_axes, self.batch_over_inputs, self.batch_over_inputs)
			)(self.inner, x1, x2)

		# We can't use vmap
		if self.batch_size == 1:
			return self.inner(x1, x2, *args, **kwargs)[
				None, ...]  # Add batch dimension

		# We can't use vmap but have to repeat cov n times
		return jnp.repeat(
			self.inner(x1, x2, *args, **kwargs)[None, ...],
			self.batch_size,
			axis=0
		)

	def __str__(self):
		# just str of the inner kernel, as the batch info is in the parameters of the inner kernel
		return f"{self.inner}"

	def replace(self,
	            inner: AbstractModule | None = None,
				batch_size: int | None = None,
				batch_in_axes: bool | None = None,
				batch_over_inputs: bool | None = None,
				**kwargs):
		# NOTE: replacing batch_in_axes to None wouldn't throw an exception, as `replace()`
		# interprets None not as a new value but as the info that the parameter doesn't have to change

		if batch_size is not None:
			raise ValueError(
				"`batch_size` is a static field and cannot be mutated for BatchModule. "
				"Initialise a new module instance instead.")
		if batch_in_axes is not None:
			raise ValueError(
				"`batch_in_axes` is a static field and cannot be mutated for BatchModule. "
				"Initialise a new module instance instead.")
		if batch_over_inputs is not None:
			raise ValueError(
				"`batch_over_inputs` is a static field and cannot be mutated for BatchModule. "
				"Initialise a new module instance instead.")

		return super().replace(inner=inner, **kwargs)
