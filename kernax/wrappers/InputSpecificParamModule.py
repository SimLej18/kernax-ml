from jax import Array, vmap
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from ..module import AbstractModule
from .WrapperModule import AbstractWrapperModule


class InputSpecificParamModule(AbstractWrapperModule):
	"""
	Wrapper module to add input-specific parameters to an inner module

	This is useful when you know the shape of x1 in advance. You can then use this module
	so that each point in x1 gets its own parameter.

	A typical usage example is with Noise kernel when you want distinct noise values along
	the x1 dimension.

	The parameters are vmaped along the x1 dimension only ; you should be mindful when you use
	this wrapper using a different x1 and x2.

	If x1 doesn't math the input_size provided at init, we throw an error, as inner parameters
	won't have an appropriate shape for vmap.
	"""
	inner: AbstractModule
	input_size: int = eqx.field(static=True)
	vmap_in_axes: bool = eqx.field(static=True)

	def __init__(self, inner, input_size, vmap_in_axes=0):
		"""
		:param inner: the kernel to wrap, must be an instance of AbstractKernel
		:param input_size: the size of the expected inputs (int)
		:param vmap_in_axes: a value or pytree indicating which hyperparameters are specific to input (0)
											   or shared (None) across all inputs.
											   If None, all hyperparameters are assumed to be shared across inputs, leading to the default behavior of the inner kernel.
											   If 0, all hyperparameters are assumed to be specific to input.
											   If a pytree, it must have the same structure as inner, with hyperparameter
											   leaves being either 0 (specific) or None (shared).
		"""
		self.input_size = input_size

		# Default: all array hyperparameters are shared (None for all array leaves)
		if vmap_in_axes is None:
			# Extract only array leaves and map them to None
			self.vmap_in_axes = jtu.tree_map(lambda _: None, inner)
		elif vmap_in_axes == 0:
			# All hyperparameters are batched
			self.vmap_in_axes = jtu.tree_map(lambda _: 0, inner)
		else:
			self.vmap_in_axes = vmap_in_axes

		# Add batch dimension to parameters where batch_in_axes is 0
		self.inner = jtu.tree_map(
			lambda param, batch_in_ax: (
				param if batch_in_ax is None else jnp.repeat(param[None, ...], input_size, axis=0)
			),
			inner,
			self.vmap_in_axes,
		)

	def __call__(self, x1: Array, x2: Array | None = None, *args, **kwargs) -> Array:
		if len(x1) != self.input_size:
			raise ValueError(f"Size of x1 ({len(x1)}) does not match input_size ({self.input_size})")

		if x2 is None:
			x2 = x1

		return vmap(
			lambda module, x: module(x, x2, *args, **kwargs),
			in_axes=(self.vmap_in_axes, 0)
		)(self.inner, x1)

	def __str__(self):
		# just str of the inner kernel, as the batch info is in the parameters of the inner kernel
		return f"{self.inner}"

	def replace(self,
	            inner: AbstractModule | None = None,
				input_size: int | None = None,
				vmap_in_axes: bool | None = None,
				**kwargs):
		# NOTE: replacing vmap_in_axes to None wouldn't throw an exception, as `replace()`
		# interprets None not as a new value but as the info that the parameter doesn't have to change

		if input_size is not None:
			raise ValueError(
				"`batch_size` is a static field and cannot be mutated for BatchModule. "
				"Initialise a new module instance instead.")
		if vmap_in_axes is not None:
			raise ValueError(
				"`batch_in_axes` is a static field and cannot be mutated for BatchModule. "
				"Initialise a new module instance instead.")

		return super().replace(inner=inner, **kwargs)
