"""
Pytest configuration for kernax-ml benchmarks.
"""


def pytest_addoption(parser):
	"""Add command-line options for benchmarks."""
	parser.addoption(
		"--bench-rounds",
		action="store",
		default="20",
		help="Number of rounds for each benchmark (default: 20)",
	)
