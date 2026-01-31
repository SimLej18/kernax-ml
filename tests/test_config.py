"""
Tests for the kernax configuration system.
"""

import threading
import warnings

import allure
import pytest

import kernax


@allure.title("Test default configuration values")
@allure.description("Verify that config has correct default values on initialization")
def test_default_config():
	"""Test that default configuration values are set correctly."""
	# Reset to ensure clean state
	kernax.config.reset()

	assert kernax.config.parameter_transform == "identity"
	assert kernax.config.get_all() == {"parameter_transform": "identity"}


@allure.title("Test global configuration setting")
@allure.description("Verify that global configuration can be set and persists")
def test_global_config_setting():
	"""Test setting global configuration values."""
	# Use unsafe_reset to ensure clean state without kernel lock
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Set parameter_transform (before any kernel instantiation)
	kernax.config.parameter_transform = "exp"
	assert kernax.config.parameter_transform == "exp"

	# Change to softplus (still no kernels)
	kernax.config.parameter_transform = "softplus"
	assert kernax.config.parameter_transform == "softplus"

	# Reset for other tests
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test invalid parameter transform raises error")
@allure.description("Verify that setting invalid parameter_transform raises ValueError")
def test_invalid_parameter_transform():
	"""Test that invalid parameter_transform values raise errors."""
	# Use unsafe_reset to ensure we can change config
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	with pytest.raises(ValueError, match="Invalid parameter_transform"):
		kernax.config.parameter_transform = "invalid"  # type: ignore[assignment]

	with pytest.raises(ValueError, match="Invalid parameter_transform"):
		kernax.config.parameter_transform = "log"  # type: ignore[assignment]

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test context manager blocked for parameter_transform")
@allure.description("Verify that parameter_transform cannot be used in set_config")
def test_context_manager_blocks_parameter_transform():
	"""Test that parameter_transform is blocked in context manager."""
	with pytest.raises(ValueError, match="parameter_transform cannot be used with set_config"):
		with kernax.config.set_config(parameter_transform="exp"):
			pass


@allure.title("Test context manager with unknown key")
@allure.description("Verify that set_config raises error for unknown keys")
def test_context_manager_unknown_key():
	"""Test that context manager works for future config options."""
	# Since parameter_transform is blocked, we test with an unknown key
	# to ensure the validation works
	with pytest.raises(ValueError, match="Unknown configuration key"):
		with kernax.config.set_config(unknown_key="value"):
			pass


@allure.title("Test kernel instantiation locks parameter_transform")
@allure.description("Verify that creating a kernel locks parameter_transform")
def test_kernel_instantiation_locks_config():
	"""Test that creating a kernel locks parameter_transform."""
	# Start fresh
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Before kernel: should be able to change
	kernax.config.parameter_transform = "exp"
	assert kernax.config.parameter_transform == "exp"

	# Create a kernel
	kernel = kernax.SEKernel(length_scale=1.0)

	# After kernel: should NOT be able to change
	with pytest.raises(RuntimeError, match="Cannot change parameter_transform after kernels"):
		kernax.config.parameter_transform = "softplus"

	# Config should still be "exp"
	assert kernax.config.parameter_transform == "exp"

	# Clean up
	del kernel
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test unsafe_reset clears lock")
@allure.description("Verify that unsafe_reset allows changing parameter_transform again")
def test_unsafe_reset():
	"""Test that unsafe_reset clears the kernel instantiation lock."""
	# Start fresh
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Create a kernel
	kernel = kernax.SEKernel(length_scale=1.0)

	# Should be locked
	with pytest.raises(RuntimeError, match="Cannot change parameter_transform"):
		kernax.config.parameter_transform = "softplus"

	# Use unsafe_reset (should warn)
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		kernax.config.unsafe_reset()
		assert len(w) == 1
		assert issubclass(w[0].category, RuntimeWarning)
		assert "unsafe_reset" in str(w[0].message)

	# Now should be able to change again
	kernax.config.parameter_transform = "softplus"
	assert kernax.config.parameter_transform == "softplus"

	# Clean up
	del kernel
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test config.reset() method")
@allure.description("Verify that reset() restores default configuration but keeps lock")
def test_config_reset():
	"""Test that reset() restores default values but doesn't clear lock."""
	# Start fresh
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Modify config (before kernels)
	kernax.config.parameter_transform = "exp"
	assert kernax.config.parameter_transform == "exp"

	# Create a kernel (locks config)
	kernel = kernax.SEKernel(length_scale=1.0)

	# Reset (resets values but NOT lock)
	kernax.config.reset()
	assert kernax.config.parameter_transform == "identity"

	# Should still be locked
	with pytest.raises(RuntimeError, match="Cannot change parameter_transform"):
		kernax.config.parameter_transform = "softplus"

	# Clean up
	del kernel
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test config.get_all() method")
@allure.description("Verify that get_all() returns complete configuration")
def test_config_get_all():
	"""Test that get_all() returns all configuration values."""
	# Start fresh
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Test with default values
	config_dict = kernax.config.get_all()
	assert isinstance(config_dict, dict)
	assert "parameter_transform" in config_dict
	assert config_dict["parameter_transform"] == "identity"

	# Test with modified global value (before kernel)
	kernax.config.parameter_transform = "softplus"
	config_dict = kernax.config.get_all()
	assert config_dict["parameter_transform"] == "softplus"

	# Reset
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test config repr")
@allure.description("Verify that config has readable string representation")
def test_config_repr():
	"""Test that config has a useful string representation."""
	kernax.config.reset()

	repr_str = repr(kernax.config)
	assert "Config" in repr_str
	assert "parameter_transform" in repr_str
	assert "identity" in repr_str


@allure.title("Test thread-local behavior")
@allure.description("Verify that config is thread-safe")
def test_thread_local_behavior():
	"""Test that config access is thread-safe."""
	# Start fresh
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Track results from threads
	results = {}

	def thread_func(thread_id):
		"""Function to run in thread."""
		# Each thread should see the same global config
		results[f"thread{thread_id}"] = kernax.config.parameter_transform
		import time

		time.sleep(0.05)

	# Run threads
	t1 = threading.Thread(target=thread_func, args=(1,))
	t2 = threading.Thread(target=thread_func, args=(2,))

	t1.start()
	t2.start()

	t1.join()
	t2.join()

	# Both threads should see the same value
	assert results["thread1"] == "identity"
	assert results["thread2"] == "identity"

	# Main thread should see same value
	assert kernax.config.parameter_transform == "identity"

	# Clean up
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()


@allure.title("Test all valid parameter transforms")
@allure.description("Verify that all valid parameter transforms can be set")
@pytest.mark.parametrize("transform", ["identity", "exp", "softplus"])
def test_all_valid_transforms(transform):
	"""Test that all valid parameter transforms work."""
	# Start fresh for each transform
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()

	# Test global setting (before kernels)
	kernax.config.parameter_transform = transform
	assert kernax.config.parameter_transform == transform

	# Clean up
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", RuntimeWarning)
		kernax.config.unsafe_reset()