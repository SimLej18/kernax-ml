"""
Goal of the script: contain functions about sampling HPs randomly to initialise kernels/means.
"""

from __future__ import annotations

import jax.random as jr

from kernax import AbstractModule, AbstractWrapperModule
from kernax.module import has_own_attr
from kernax.operators import AbstractOperatorModule


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
	if isinstance(module, AbstractWrapperModule):
		key, subkey = jr.split(key)
		new_module = module.replace(inner=sample_hps_from_uniform_priors(subkey, module.inner, priors))
		return _sample_own_hps_uniform(key, new_module, priors)

	if isinstance(module, AbstractOperatorModule):
		key, subkey1, subkey2 = jr.split(key, 3)
		new_operator_module = module.replace(
			left=sample_hps_from_uniform_priors(subkey1, module.left, priors),
			right=sample_hps_from_uniform_priors(subkey2, module.right, priors),
		)
		return _sample_own_hps_uniform(key, new_operator_module, priors)

	if isinstance(module, AbstractModule):
		return _sample_own_hps_uniform(key, module, priors)

	raise ValueError("Module must be an instance of AbstractModule, AbstractWrapperModule, or AbstractOperatorModule.")


def _sample_own_hps_uniform(key, module, priors):
	"""Sample the hyperparameters `module` itself carries (as opposed to ones nested in
	`inner`/`left`/`right`), for every `param` in `priors` the module has."""
	new_module = module
	for param in priors.keys():
		key, subkey = jr.split(key)
		if has_own_attr(module, param):
			current = getattr(new_module, param)
			new_module = new_module.replace(**{param: jr.uniform(
				subkey,
				shape=current.shape,
				dtype=current.dtype,
				minval=priors[param][0],
				maxval=priors[param][1],
			)})

	return new_module


def sample_hps_from_normal_priors(key, module, priors):
	"""
	Sample hyperparameters from normal priors.

	Args:
		key: JAX PRNG key
		module: Kernel or Mean module to initialize.
		priors: Dictionary of prior bounds for hyperparameters. If an HP is not specified in the dictionary, it will be left unchanged.

	N.b:
	If batched (aka HPs are arrays), each scalar HP value will be sampled independently, and the structure will be left unchanged.
	If you want HP values to be the same at initialisation, you should sample HPs *before* batching the module.

	ex:
	```
	# Priors with mean and std for each hyperparameter
	mean_priors = {
	"slope": (0., 1.),
	"intercept": (0., 5.)
	}

	mean_kernel_priors = {
		"variance": (5, 20.),
		"length_scale": (2.5, 10.)
	}

	task_kernel_priors = {
		"variance": (1.5, .5),
		"length_scale": (4., 1.),
		"noise": (0.05, 0.1)
	}

	mean = AffineMean(slope=0., intercept=0.)
	mean_kernel = VarianceKernel(20.) * SEKernel(length_scale=10.)
	task_kernel = VarianceKernel(.2) * SEKernel(length_scale=9.) + WhiteNoiseKernel(noise=.01)

	# The function returns new instances of the modules with sampled HPs.
	sampled_mean = sample_hps_from_normal_priors(key, mean, mean_priors)
	sampled_mean_kernel = sample_hps_from_normal_priors(key, mean_kernel, mean_kernel_priors)
	sampled_task_kernel = sample_hps_from_normal_priors(key, task_kernel, task_kernel_priors)

	# ``sampled_mean``, ``sampled_mean_kernel``, and ``sampled_task_kernel`` are now initialized with hyperparameters sampled from the specified normal priors.
	```
	"""
	if isinstance(module, AbstractWrapperModule):
		key, subkey = jr.split(key)
		new_module = module.replace(inner=sample_hps_from_normal_priors(subkey, module.inner, priors))
		return _sample_own_hps_normal(key, new_module, priors)

	if isinstance(module, AbstractOperatorModule):
		key, subkey1, subkey2 = jr.split(key, 3)
		new_operator_module = module.replace(
			left=sample_hps_from_normal_priors(subkey1, module.left, priors),
			right=sample_hps_from_normal_priors(subkey2, module.right, priors),
		)
		return _sample_own_hps_normal(key, new_operator_module, priors)

	if isinstance(module, AbstractModule):
		return _sample_own_hps_normal(key, module, priors)

	raise ValueError(
		"Module must be an instance of AbstractModule, AbstractWrapperModule, or AbstractOperatorModule.")


def _sample_own_hps_normal(key, module, priors):
	"""Sample the hyperparameters `module` itself carries (as opposed to ones nested in
	`inner`/`left`/`right`), for every `param` in `priors` the module has."""
	new_module = module
	for param in priors.keys():
		key, subkey = jr.split(key)
		if has_own_attr(module, param):
			current = getattr(new_module, param)
			new_module = new_module.replace(**{param: jr.normal(
				subkey,
				shape=current.shape,
				dtype=current.dtype,
			) * priors[param][1] + priors[param][0]})

	return new_module
