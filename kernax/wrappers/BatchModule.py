import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import filter_jit
from jax import Array, vmap

from .WrapperModule import WrapperModule


class BatchModule(WrapperModule):
	"""
	Wrapper module to add batch handling to any module.

	A basic kernel usually works on inputs of shape (N, I), and produces covariance matrices of shape (N, N).

	Wrapped inside a batch module, they can either:
	- still work on inputs of shape (N, I), but produce covariance matrices of shape (B, N, N), where B is the batch size. This is useful when the hyperparameters are batched, i.e. each batch element has its own set of hyperparameters.
	- or work on inputs of shape (B, N, I), producing covariance matrices of shape (B, N, N). This is useful when the inputs are batched, regardless of whether the hyperparameters are batched or not.

	A batch module can itself be wrapped inside another batch module, to handle multiple batch dimensions/hyperparameter sets.

	This class uses vmap to vectorize the module computation over the batch dimension.
	"""
	batch_size: int = eqx.field(static=True)
	batch_in_axes: bool = eqx.field(static=True)
	batch_over_inputs: int | None = eqx.field(static=True)

	def __init__(self, inner, batch_size, batch_in_axes=None, batch_over_inputs=True, **kwargs):
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
		# Initialize the WrapperKernel
		super().__init__(inner=inner, **kwargs)
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
			self.inner,
			self.batch_in_axes,
		)

	@filter_jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		"""
		Compute the kernel over batched inputs using vmap.

		Args:
				x1: Input of shape (B, ..., N, I)
				x2: Optional second input of shape (B, ..., M, I)

		Returns:
				Kernel matrix of appropriate shape with batch dimension
		"""
		# Check if we can use vmap (at least one axis is not None)
		can_use_vmap = (
			not jtu.tree_all(jtu.tree_map(lambda k: k is None, self.batch_in_axes))
			or self.batch_over_inputs is not None
		)

		if can_use_vmap:
			# Use vmap when we have batched hyperparameters or batched inputs

			# As inner can either be a Mean or a Kernel, we need to check if x2 is None to call the right function signature in vmap
			if x2 is None:
				return vmap(  # type: ignore[no-any-return]
					lambda module, x1: module(x1),
				in_axes=(
					self.batch_in_axes,
					self.batch_over_inputs))(self.inner, x1)
			else:
				return vmap(  # type: ignore[no-any-return]
					lambda module, x1, x2: module(x1, x2),
					in_axes=(
						self.batch_in_axes,
						self.batch_over_inputs,
						self.batch_over_inputs,
					))(self.inner, x1, x2)
		else:
			if self.batch_size == 1:
				if x2 is None:
					return self.inner(x1)[None, ...]  # Add batch dimension

				# If batch size is 1, we can just call the inner kernel without repeating
				return self.inner(x1, x2)[None, ...]  # Add batch dimension

			# Repeat the same matrix when all hyperparameters and inputs are shared
			if x2 is None:
				return jnp.repeat(
					jnp.expand_dims(self.inner(x1), 0),
					self.batch_size,
					axis=0
				)
			return jnp.repeat(
				jnp.expand_dims(self.inner(x1, x2), 0),
				self.batch_size,
				axis=0
			)

	def replace(self, **kwargs):
		_STATIC_FIELDS = {"batch_size", "batch_in_axes", "batch_over_inputs"}
		illegal = _STATIC_FIELDS & kwargs.keys()
		if illegal:
			names = ", ".join(f"'{f}'" for f in sorted(illegal))
			raise ValueError(
				f"{names} {'is' if len(illegal) == 1 else 'are'} structural "
				f"parameter(s) of BatchModule and cannot be modified via replace(). "
				f"Create a new BatchModule with the desired configuration."
			)
		return super().replace(**kwargs)

	def __str__(self):
		# just str of the inner kernel, as the batch info is in the parameters of the inner kernel
		return f"{self.inner}"
