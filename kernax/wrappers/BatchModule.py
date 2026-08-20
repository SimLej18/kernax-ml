"""
Batching wrapper: evaluate a module for several hyperparameter sets and/or input sets at once.
"""

from __future__ import annotations

from typing import Any, Generic

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from eqxbatch import Batched, broadcast

from ..module import ModuleT
from .WrapperModule import AbstractWrapperModule


class BatchModule(Batched, AbstractWrapperModule[ModuleT], Generic[ModuleT]):
	"""
	Wrapper module to add batch handling to any module.

	A basic kernel usually works on inputs of shape ``(N, I)`` and produces covariance
	matrices of shape ``(N, N)``. Wrapped inside a batch module, it can either:

	- still work on inputs of shape ``(N, I)``, but produce covariance matrices of shape
	  ``(B, N, N)``, where ``B`` is the batch size. This is useful when the hyperparameters
	  are batched, i.e. each batch element has its own set of hyperparameters.
	- or work on inputs of shape ``(B, N, I)``, producing covariance matrices of shape
	  ``(B, N, N)``. This is useful when the inputs are batched, regardless of whether the
	  hyperparameters are batched or not.

	A batch module can itself be wrapped inside another batch module, to handle multiple
	batch dimensions/hyperparameter sets.

	This is kernax's typed view of :class:`eqxbatch.Batched`: it keeps kernax's constructor
	signature and its ``replace()`` contract, and it participates in the module algebra
	(``+``, ``*``, ``-``) that :class:`~kernax.module.AbstractModule` provides. The batching
	itself -- stacking the leaves, re-applying ``eqx.filter_vmap`` to ``__call__`` and to
	every forwarded attribute -- lives in ``eqxbatch`` and is not kernel-specific.

	Batched hyperparameters are stored with a leading batch axis, and are only meaningful
	under the vmap this module applies. Reaching into ``batched.inner`` and calling it
	directly therefore yields wrong results -- silently so, for kernels whose maths happens
	to broadcast. Use the forwarding instead: any public attribute this class does not
	define itself is looked up on the wrapped module and re-evaluated under the same vmap
	as ``__call__``::

	    batched = BatchModule(SEKernel(1.0), batch_size=4, batch_in_axes=0)
	    batched.length_scale            # (4,)    -- property, evaluated per batch element
	    batched.spectral_density(w)     # (4, M)  -- method, called per batch element

	Unlike the other wrappers, this one forwards *methods* as well as hyperparameters: it
	does not transform what ``inner`` computes, it only evaluates it once per batch element,
	so the inner module's answer stays the right answer.

	See :meth:`eqxbatch.Batched.map` for computations spanning several attributes at once,
	and ``batched[i]`` to recover one batch element as an ordinary kernel.

	:param inner: the module to wrap, must be an instance of ``AbstractModule``
	:param batch_size: the size of the batch
	:param batch_in_axes: a value or pytree indicating which hyperparameters are batched
	                      (``0``) or shared (``None``) across the batch.
	                      If ``None``, all hyperparameters are assumed to be shared.
	                      If ``0``, all hyperparameters are assumed to be batched.
	                      If a pytree, it must have the same structure as ``inner``, with
	                      hyperparameter leaves being either ``0`` (batched) or ``None``
	                      (shared).
	:param batch_over_inputs: whether ``__call__`` expects inputs of shape ``(B, N, I)``
	                          (``True``) or ``(N, I)`` (``False``)
	:param batch_over_kwargs: whether keyword arguments passed to ``__call__`` (e.g.
	                          ``output_ids``) carry a leading batch axis and should be
	                          sliced per batch element (``True``), or are shared across the
	                          batch (``False``). Defaults to ``batch_over_inputs``.
	"""

	def __init__(self,
	             inner: ModuleT,
	             batch_size: int,
	             batch_in_axes: Any = None,
	             batch_over_inputs: bool = True,
	             batch_over_kwargs: bool | None = None):
		if batch_over_kwargs is None:
			batch_over_kwargs = batch_over_inputs
		if batch_in_axes is None:
			# All hyperparameters are shared
			in_axes = None
		elif batch_in_axes == 0:
			# All hyperparameters are batched
			in_axes = eqx.if_array(0)
			inner = broadcast(inner, batch_size)
		else:
			# Per-leaf spec, e.g. the output of `kernax.mask.create_mask`
			in_axes = batch_in_axes
			inner = jtu.tree_map(
				lambda leaf, ax: (
					leaf if ax is None
					else jnp.broadcast_to(leaf, (batch_size, *jnp.shape(leaf)))
				),
				inner,
				batch_in_axes,
			)

		Batched.__init__(
			self,
			inner,
			in_axes=in_axes,
			arg_axes=eqx.if_array(0) if batch_over_inputs else None,
			kwarg_axes=eqx.if_array(0) if batch_over_kwargs else None,
			axis_size=batch_size,
		)

	@property
	def batch_size(self) -> int | None:
		return self.axis_size

	@property
	def batch_in_axes(self) -> Any:
		return self.in_axes

	@property
	def batch_over_inputs(self) -> int | None:
		return None if self.arg_axes is None else 0

	@property
	def batch_over_kwargs(self) -> int | None:
		return None if self.kwarg_axes is None else 0

	def __str__(self):
		# just str of the inner kernel, as the batch info is in the parameters of the inner kernel
		return f"{self.inner}"

	def replace(self,
	            inner: ModuleT | None = None,
	            batch_size: int | None = None,
	            batch_in_axes: Any = None,
	            batch_over_inputs: bool | None = None,
	            batch_over_kwargs: bool | None = None,
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
		if batch_over_kwargs is not None:
			raise ValueError(
				"`batch_over_kwargs` is a static field and cannot be mutated for BatchModule. "
				"Initialise a new module instance instead.")

		return super().replace(inner=inner, **kwargs)

