"""Per-point hyperparameter gather shared by `BlockDiagKernel` and `BlockMean`."""
from __future__ import annotations

from typing import Any

import jax.tree_util as jtu
from jax import Array


def _resolve(spec: Any, tree: Any) -> Any:
	"""Turn an axis spec (int, None, callable, or pytree prefix) into one axis per leaf.

	Mirrors the private resolution `eqxbatch.Batched` applies internally before vmapping --
	needed here too since gathering indexes leaves directly, outside of any vmap that would
	otherwise interpret the shorthand spec on its own.
	"""
	if spec is None or isinstance(spec, int):
		return jtu.tree_map(lambda _: spec, tree)
	if callable(spec):
		return jtu.tree_map(spec, tree)
	return jtu.tree_map(_resolve, spec, tree, is_leaf=lambda x: x is None)


def gather_by_output(inner: Any, in_axes: Any, output_ids: Array) -> Any:
	"""Index every per-output leaf of `inner` (marked `0` in `in_axes`) by `output_ids`.

	Turns a `(P, ...)` pytree of per-output hyperparameters into an `(N, ...)` pytree of
	per-point hyperparameters, leaving shared (`None`-marked) leaves untouched.
	"""
	resolved = _resolve(in_axes, inner)
	return jtu.tree_map(
		lambda leaf, ax: leaf if ax is None else leaf[output_ids],
		inner, resolved, is_leaf=lambda a: a is None,
	)
