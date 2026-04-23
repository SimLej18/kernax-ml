"""
Goal of the script: provide `create_mask`, a utility to build pytree-compatible masks
over any Equinox module (kernel or mean).
"""
from __future__ import annotations
import equinox as eqx
from jax import Array
from kernax.parametrisations import AbstractParametrisation, IdentityParametrisation


def create_mask(module: eqx.Module, default=None, **kwargs) -> eqx.Module:
	# FIXME: create_mask doesn't work on BatchModules
	"""
	Return a copy of `module` with the same pytree structure, where Array fields are
	replaced by mask values instead of their original JAX arrays.

	- Fields whose name matches a kwarg receive that kwarg's value.
	- All other Array fields receive `default`.
	- Non-Array, non-Module fields (static fields, ints, …) are left unchanged.

	Args:
		module:   an Equinox module (kernel or mean)
		default:  value assigned to Array fields not listed in kwargs (e.g. None, 0, False)
		**kwargs: field_name=mask_value pairs

	Examples:
		# Build a batch_in_axes for BatchModule — only length_scale is batched
		mask = create_mask(kernel, default=None, length_scale=0)
		batched = BatchModule(kernel, batch_size=4, batch_in_axes=mask)

		# Freeze everything except length_scale for training (eqx.partition)
		mask = create_mask(kernel, default=False, length_scale=True)
		trainable, frozen = eqx.partition(kernel, mask)
	"""
	return _recurse(module, kwargs, default)


def _recurse(module: eqx.Module, masks: dict, default) -> eqx.Module:
	"""Recursively replace Array fields with their mask values."""

	for field_name, field_value in vars(module).items():
		if isinstance(field_value, Array):
			if field_name in masks:
				module = eqx.tree_at(
					lambda m: m.__getattribute__(field_name),
					module,
					masks[field_name],
				)
			elif field_name.startswith("_") and field_name[1:] in masks:
				module = eqx.tree_at(
					lambda m: m.__getattribute__(field_name),
					module,
					masks[field_name[1:]],
				)
			else:
				module = eqx.tree_at(
					lambda m: m.__getattribute__(field_name),
					module,
					default,
				)

		elif isinstance(field_value, eqx.Module) and field_name in ("inner", "left", "right"):
			module = eqx.tree_at(
				lambda m: m.__getattribute__(field_name),
				module,
				_recurse(field_value, masks, default),
			)

	return module
