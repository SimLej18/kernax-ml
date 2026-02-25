"""
Goal of the script: contain functions about sampling HPs randomly to initialise kernels/means.
"""

from __future__ import annotations

import jax
import equinox as eqx
from jax import Array

from .transforms import to_unconstrained


def sample_hps_from_uniform_priors(key, module, priors):
	"""
	Sample hyperparameters from uniform priors.

	Args:
		key: JAX PRNG key
		module: Kernel or Mean module to initialize.
		priors: Dictionary of prior bounds for hyperparameters. If an HP is not specified in the dictionary, it will be left unchanged.

	N.b:
	If batched (aka HPs are arrays), each scalar HP value will be sampled independently, and the structure will be left unchanged.
	If you want HP values to be the same at initialisation, you should sample HPs *before* batching the module.

	ex:
	```
	# Priors with lower and upper bounds for each hyperparameter
	mean_priors = {
	"slope": (-1., 1.),
	"intercept": (-5., 5.)
	}

	mean_kernel_priors = {
		"variance": (5, 20.),
		"length_scale": (2.5, 10.)
	}

	task_kernel_priors = {
		"variance": (0.25, 2.5),
		"length_scale": (2., 8.),
		"noise": (0.01, 0.1)
	}

	mean = AffineMean(slope=0., intercept=0.)
	mean_kernel = VarianceKernel(20.) * SEKernel(length_scale=10.)
	task_kernel = VarianceKernel(.2) * SEKernel(length_scale=9.) + WhiteNoiseKernel(noise=.01)

	# The function returns new instances of the modules with sampled HPs.
	sampled_mean = sample_hps_from_uniform_priors(key, mean, mean_priors)
	sampled_mean_kernel = sample_hps_from_uniform_priors(key, mean_kernel, mean_kernel_priors)
	sampled_task_kernel = sample_hps_from_uniform_priors(key, task_kernel, task_kernel_priors)

	# ``sampled_mean``, ``sampled_mean_kernel``, and ``sampled_task_kernel`` are now initialized with hyperparameters sampled from the specified uniform priors.
	```
	"""
	return _sample_recursive(key, module, priors)


def _sample_recursive(key, module, priors):
	"""Recursively traverse the module tree and sample matching HP fields from uniform priors."""
	# --- Step 1: sample direct Array fields of this module ---
	direct_fields, direct_values = [], []

	for field_name, field_value in vars(module).items():
		if not isinstance(field_value, Array):
			continue

		hp_name = field_name[5:] if field_name.startswith("_raw_") else field_name
		if hp_name not in priors:
			continue

		low, high = priors[hp_name]
		key, subkey = jax.random.split(key)
		sampled = jax.random.uniform(subkey, shape=field_value.shape, minval=low, maxval=high)

		if field_name.startswith("_raw_"):
			sampled = to_unconstrained(sampled)

		direct_fields.append(field_name)
		direct_values.append(sampled)

	if direct_fields:
		where = lambda m: [getattr(m, k) for k in direct_fields]
		module = eqx.tree_at(where, module, direct_values)

	# --- Step 2: recurse into nested eqx.Module fields ---
	for field_name, field_value in vars(module).items():
		if not isinstance(field_value, eqx.Module):
			continue

		key, subkey = jax.random.split(key)
		new_sub = _sample_recursive(subkey, field_value, priors)

		if new_sub is not field_value:
			module = eqx.tree_at(lambda m, fn=field_name: getattr(m, fn), module, new_sub)

	return module