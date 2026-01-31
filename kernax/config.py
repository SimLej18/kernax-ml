"""
Configuration system for kernax.

This module provides a global configuration object that can be accessed via `kernax.config`.
Configuration options can be set globally or temporarily within a context manager.

IMPORTANT: parameter_transform must be set BEFORE creating any kernels, as JIT compilation
caches prevent dynamic changes. It cannot be used with the set_config() context manager.

Example:
	>>> import kernax
	>>> # Set parameter transform before creating kernels
	>>> kernax.config.parameter_transform = "softplus"
	>>> kernel = kernax.SEKernel(length_scale=1.0)
"""

from __future__ import annotations

import warnings
from contextvars import ContextVar
from typing import Any, Literal

# Valid parameter transform modes
ParameterTransform = Literal["identity", "exp", "softplus"]

# Context variable to store thread-local configuration
_config_context: ContextVar[dict[str, Any]] = ContextVar("kernax_config", default={})


class Config:
	"""
	Global configuration object for kernax.

	This class manages global and thread-local configuration options for the library.
	Configuration can be modified globally or temporarily within a context manager.

	IMPORTANT: parameter_transform must be set BEFORE creating any kernels, as changing it
	after kernel instantiation would cause inconsistencies with JIT-compiled functions.

	Attributes:
		parameter_transform: Transform to apply to positive-constrained parameters.
			- "identity": No transform (parameters stay in constrained space)
			- "exp": Exponential transform (log-exp trick)
			- "softplus": Softplus transform (log(1 + exp(x)))
			Default: "identity"
			NOTE: Must be set before creating any kernels!
	"""

	def __init__(self):
		# Default configuration values
		self._global_config: dict[str, Any] = {
			"parameter_transform": "identity",
		}
		# Track whether any kernels have been instantiated
		self._kernels_instantiated = False
		# Allow unsafe changes (for testing purposes only)
		self._allow_unsafe_changes = False

	@property
	def parameter_transform(self) -> ParameterTransform:
		"""
		Get the current parameter transform mode.

		Returns:
			The current parameter transform: "identity", "exp", or "softplus"
		"""
		return self._get_value("parameter_transform")

	@parameter_transform.setter
	def parameter_transform(self, value: ParameterTransform) -> None:
		"""
		Set the parameter transform mode globally.

		IMPORTANT: This must be set BEFORE creating any kernels. Once a kernel has been
		instantiated, changing this value would cause inconsistencies with JIT-compiled
		functions.

		Args:
			value: Parameter transform mode ("identity", "exp", or "softplus")

		Raises:
			ValueError: If value is not a valid transform mode
			RuntimeError: If trying to change after kernels have been instantiated

		Example:
			>>> import kernax
			>>> kernax.config.parameter_transform = "softplus"  # Before kernels
			>>> kernel = kernax.SEKernel(length_scale=1.0)
			>>> # kernax.config.parameter_transform = "exp"  # Would raise RuntimeError!
		"""
		# Check if kernels have been instantiated
		if self._kernels_instantiated and not self._allow_unsafe_changes:
			raise RuntimeError(
				"Cannot change parameter_transform after kernels have been instantiated.\n"
				"Reason: JIT-compiled functions cache the transform value, causing "
				"inconsistent behavior.\n"
				"Solution: Set parameter_transform before creating kernels, or call\n"
				"  kernax.config.unsafe_reset() to reset (invalidates existing kernels)."
			)

		# Validate value
		valid_transforms: tuple[ParameterTransform, ...] = ("identity", "exp", "softplus")
		if value not in valid_transforms:
			raise ValueError(
				f"Invalid parameter_transform: {value}. "
				f"Must be one of {valid_transforms}"
			)

		self._global_config["parameter_transform"] = value

	def _get_value(self, key: str) -> Any:
		"""
		Get a configuration value, checking thread-local context first, then global.

		Args:
			key: Configuration key to retrieve

		Returns:
			The configuration value
		"""
		context_config = _config_context.get()
		if key in context_config:
			return context_config[key]
		return self._global_config[key]

	def get_all(self) -> dict[str, Any]:
		"""
		Get all current configuration values (global + thread-local overrides).

		Returns:
			Dictionary of all configuration values
		"""
		config = self._global_config.copy()
		config.update(_config_context.get())
		return config

	def set_config(self, **kwargs: Any) -> ConfigContext:
		"""
		Create a context manager for temporary configuration changes.

		This allows you to temporarily override configuration values within a
		specific scope. Changes are automatically reverted when exiting the context.

		NOTE: parameter_transform CANNOT be used with this context manager due to
		JIT compilation caching. It must be set globally before creating kernels.

		Args:
			**kwargs: Configuration values to set temporarily

		Returns:
			A context manager that applies the configuration changes

		Example:
			>>> # This would work if we had other config options:
			>>> # with kernax.config.set_config(some_other_option=value):
			>>> #     ...

		Raises:
			ValueError: If any configuration key or value is invalid, or if
				parameter_transform is specified
		"""
		# Block parameter_transform in context manager
		if "parameter_transform" in kwargs:
			raise ValueError(
				"parameter_transform cannot be used with set_config() context manager.\n"
				"Reason: JIT compilation caches prevent dynamic changes.\n"
				"Solution: Set it globally before creating kernels:\n"
				"  kernax.config.parameter_transform = 'exp'"
			)

		# Validate all keys and values before creating context
		for key, value in kwargs.items():
			if key not in self._global_config:
				raise ValueError(
					f"Unknown configuration key: {key}. "
					f"Valid keys: {list(self._global_config.keys())}"
				)

		return ConfigContext(kwargs)

	def _mark_kernel_instantiated(self) -> None:
		"""
		Mark that a kernel has been instantiated.

		This is called automatically by AbstractKernel.__init__() and locks the
		parameter_transform setting to prevent inconsistencies with JIT-compiled code.

		Internal use only - should not be called by users.
		"""
		self._kernels_instantiated = True

	def reset(self) -> None:
		"""
		Reset all global configuration values to their defaults.

		This does not affect thread-local context overrides currently in effect.

		NOTE: This does NOT reset the kernel instantiation lock. If kernels have been
		created, you cannot change parameter_transform even after reset(). Use
		unsafe_reset() if you need to reset everything (use with caution).
		"""
		self._global_config = {
			"parameter_transform": "identity",
		}

	def unsafe_reset(self) -> None:
		"""
		Reset configuration AND clear the kernel instantiation lock.

		WARNING: This is unsafe! After calling this, any existing kernels will behave
		inconsistently because their JIT-compiled code may use different transform
		settings than newly created kernels.

		Use this only when:
		- You are certain no existing kernels will be used
		- You are in an interactive session and want to start fresh
		- You are writing tests

		After calling this, you should delete all existing kernel instances before
		creating new ones.
		"""
		warnings.warn(
			"unsafe_reset() clears the kernel instantiation lock. "
			"Any existing kernels will behave inconsistently if used after this call. "
			"Delete all kernel instances before creating new ones.",
			RuntimeWarning,
			stacklevel=2,
		)
		self._kernels_instantiated = False
		self._allow_unsafe_changes = False
		self.reset()

	def __repr__(self) -> str:
		"""String representation showing current configuration."""
		config = self.get_all()
		config_str = ", ".join(f"{k}={v!r}" for k, v in sorted(config.items()))
		return f"Config({config_str})"


class ConfigContext:
	"""
	Context manager for temporary configuration changes.

	This class should not be instantiated directly. Use `config.set_config()` instead.
	"""

	def __init__(self, overrides: dict[str, Any]):
		"""
		Initialize the context manager.

		Args:
			overrides: Configuration values to override temporarily
		"""
		self.overrides = overrides
		self.token = None

	def __enter__(self) -> ConfigContext:
		"""Enter the context and apply configuration overrides."""
		# Get current context config and merge with new overrides
		current_context = _config_context.get().copy()
		current_context.update(self.overrides)
		self.token = _config_context.set(current_context)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
		"""Exit the context and restore previous configuration."""
		if self.token is not None:
			_config_context.reset(self.token)


# Global configuration instance
config = Config()