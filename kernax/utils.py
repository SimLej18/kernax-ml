import jax.numpy as jnp


def to_superscript(text):
	"""Converts text to unicode superscript characters."""
	superscript_table = str.maketrans("0123456789x×", "⁰¹²³⁴⁵⁶⁷⁸⁹ˣˣ")
	return text.translate(superscript_table)


def to_subscript(text):
	"""Converts text to unicode subscript characters."""
	subscript_table = str.maketrans("0123456789x×", "₀₁₂₃₄₅₆₇₈₉ₓₓ")
	return text.translate(subscript_table)


def format_jax_array(arr, decimals=2, color=None):
	"""Formats a JAX array as [mean ± std]^shape.

	Args:
		arr: JAX array to format
		decimals: Number of decimals to display (default: 4)
		color: Text color - 'green', 'red', 'blue', 'yellow', etc. (default: None)
	"""
	# ANSI codes for colors
	color_codes = {
		"red": "\033[91m",
		"green": "\033[92m",
		"yellow": "\033[93m",
		"blue": "\033[94m",
		"magenta": "\033[95m",
		"cyan": "\033[96m",
		"reset": "\033[0m",
	}

	if arr.size == 1:
		result = f"{arr.item():.{decimals}f}"

	else:
		mean = float(jnp.mean(arr))
		std = float(jnp.std(arr))

		# Format the shape with × between dimensions
		shape_str = "×".join(str(d) for d in arr.shape)
		shape_str = to_subscript(shape_str)

		result = f"[{mean:.{decimals}f} ± {std:.{decimals}f}]{shape_str}"

	# Apply color if specified
	if color and color.lower() in color_codes:
		result = f"{color_codes[color.lower()]}{result}{color_codes['reset']}"

	return result
