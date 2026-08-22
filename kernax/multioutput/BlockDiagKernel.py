from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
from eqxbatch import Batched
from jax import Array

from ..types import KernelLike
from ..wrappers.BatchModule import BatchModule
from ._gather import gather_by_output


class BlockDiagKernel(BatchModule[KernelLike]):
	"""Block-diagonal multi-output kernel: independent kernel per output, no cross-output
	correlation -- the off-diagonal blocks are exactly zero.

	Composes with :class:`~kernax.multioutput.ICMKernel.ICMKernel` and friends via ``+``,
	e.g. to add independent per-output noise on top of a correlated signal.

	Same two input regimes as ``ICMKernel``/``LCMKernel``, selected by whether
	``output_ids`` is passed to ``__call__`` -- a property of the data, not of the kernel:

	- Omitted: ``x1`` is the grid *shared* by every output, shape ``(N, I)``. The result is
	  the dense block-diagonal matrix, shape ``(P*N, P*N)``, in output-major order.
	- Given: ``x1`` holds points from every output, in any order, shape ``(N, I)``;
	  ``output_ids`` (and ``output_ids2`` for cross-covariances) is an integer array of
	  shape ``(N,)`` giving, for each row, which of the ``P`` outputs it belongs to. The
	  result has shape ``(N, M)``, entry ``(i, j)`` zero unless
	  ``output_ids[i] == output_ids2[j]``.

	:param inner: kernel used as a template; per-output hyperparameters are read from
	    ``output_hps_in_axes``
	:param n_outputs: number of outputs ``P``
	:param output_hps_in_axes: pytree mirror of ``inner``, leaf ``0`` where the hyperparameter
	    varies per output or ``None`` where it is shared across outputs -- see
	    :func:`kernax.mask.create_mask`
	"""

	def __init__(self, inner: KernelLike, n_outputs: int, output_hps_in_axes=None):
		super().__init__(inner, n_outputs, output_hps_in_axes, batch_over_inputs=False)

	@property
	def n_outputs(self) -> int:
		return self.axis_size

	@property
	def output_hps_in_axes(self):
		return self.in_axes

	def __call__(self, x1: Array, x2: Array | None = None, *,
	             output_ids: Array | None = None,
	             output_ids2: Array | None = None) -> Array:
		if output_ids is None:
			if output_ids2 is not None:
				raise ValueError("`output_ids2` requires `output_ids`.")
			return jsp.linalg.block_diag(*super().__call__(x1, x2))  # type: ignore[no-any-return]

		output_ids = jnp.asarray(output_ids)
		if output_ids.shape[0] != x1.shape[0]:
			raise ValueError(
				f"`output_ids` must have length {x1.shape[0]} (= len(x1)), got {output_ids.shape[0]}.")
		x2 = x1 if x2 is None else x2
		if output_ids2 is None:
			output_ids2 = output_ids
		else:
			output_ids2 = jnp.asarray(output_ids2)
			if output_ids2.shape[0] != x2.shape[0]:
				raise ValueError(
					f"`output_ids2` must have length {x2.shape[0]} (= len(x2)), got {output_ids2.shape[0]}.")

		per_point = Batched(
			gather_by_output(self.inner, self.in_axes, output_ids),
			in_axes=self.in_axes,
			axis_size=output_ids.shape[0],
		)
		K = per_point.map(lambda k, xi: k(xi, x2), x1, arg_axes=0)
		return jnp.where(output_ids[:, None] == output_ids2[None, :], K, 0.0)

	def spectral_density(self, w: Array) -> Array:
		raise NotImplementedError(
			"`BlockDiagKernel` has no scalar spectral density: it is the matrix-valued "
			"`diag(S_1(w), ..., S_P(w))`, which this class does not expose. Use "
			"`self.map(lambda k: k.spectral_density(w))` for the per-output densities."
		)

	def replace(self, **kwargs) -> BlockDiagKernel:
		if "n_outputs" in kwargs:
			raise ValueError(
				"`n_outputs` is a structural parameter of BlockDiagKernel and cannot be "
				"modified via replace(). Create a new BlockDiagKernel with the desired configuration."
			)
		return super().replace(**kwargs)  # BatchModule.replace() handles output_hps_in_axes etc.

	def __str__(self):
		return f"BlockDiag{self.inner}"
