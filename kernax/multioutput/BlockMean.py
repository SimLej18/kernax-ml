from __future__ import annotations

import jax.numpy as jnp
from eqxbatch import Batched
from jax import Array

from ..AbstractMean import AbstractMean
from ..types import MeanLike
from ..wrappers.BatchModule import BatchModule
from ._gather import gather_by_output


class BlockMean(BatchModule[MeanLike]):
	"""Block multi-output mean: independent mean function per output.

	Same two input regimes as :class:`~kernax.multioutput.BlockDiagKernel.BlockDiagKernel`,
	selected by whether ``output_ids`` is passed to ``__call__``:

	- Omitted: ``x`` is the grid *shared* by every output, shape ``(N, I)``. The result is
	  ``(P*N,)``, the per-output means flattened output-major -- aligned with the rows of a
	  ``BlockDiagKernel``/``ICMKernel`` built the same way.
	- Given: ``x`` holds points from every output, in any order, shape ``(N, I)``;
	  ``output_ids`` is an integer array of shape ``(N,)`` giving, for each row, which of
	  the ``P`` outputs it belongs to. The result has shape ``(N,)``, one value per point.

	:param inner: mean used as a template; per-output hyperparameters are read from
	    ``output_hps_in_axes``
	:param n_outputs: number of outputs ``P``
	:param output_hps_in_axes: pytree mirror of ``inner``, leaf ``0`` where the hyperparameter
	    varies per output or ``None`` where it is shared across outputs -- see
	    :func:`kernax.mask.create_mask`
	"""

	def __init__(self, inner: AbstractMean, n_outputs: int, output_hps_in_axes=None):
		super().__init__(inner, n_outputs, output_hps_in_axes, batch_over_inputs=False)

	@property
	def n_outputs(self) -> int:
		return self.axis_size

	@property
	def output_hps_in_axes(self):
		return self.in_axes

	def __call__(self, x: Array, *, output_ids: Array | None = None) -> Array:
		if output_ids is None:
			return super().__call__(x).reshape(-1)  # type: ignore[no-any-return]

		output_ids = jnp.asarray(output_ids)
		if output_ids.shape[0] != x.shape[0]:
			raise ValueError(
				f"`output_ids` must have length {x.shape[0]} (= len(x)), got {output_ids.shape[0]}.")

		per_point = Batched(
			gather_by_output(self.inner, self.in_axes, output_ids),
			in_axes=self.in_axes,
			axis_size=output_ids.shape[0],
		)
		return per_point.map(lambda m, xi: m(xi), x, arg_axes=0)  # type: ignore[no-any-return]

	def replace(self, **kwargs) -> BlockMean:
		if "n_outputs" in kwargs:
			raise ValueError(
				"`n_outputs` is a structural parameter of BlockMean and cannot be "
				"modified via replace(). Create a new BlockMean with the desired configuration."
			)
		return super().replace(**kwargs)  # BatchModule.replace() handles output_hps_in_axes etc.

	def __str__(self):
		return f"Block{self.inner}"
