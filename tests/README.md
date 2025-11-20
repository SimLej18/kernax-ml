# Kernax Test Suite

This directory contains the test suite for Kernax.

## Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=kernax --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_base_kernels.py
```

Run specific test class:
```bash
pytest tests/test_base_kernels.py::TestRBFKernel
```

Run specific test:
```bash
pytest tests/test_base_kernels.py::TestRBFKernel::test_instantiation
```

## Test Structure

- `conftest.py`: Pytest configuration and shared fixtures
- `test_base_kernels.py`: Tests for base kernel implementations
- `test_composite_kernels.py`: Tests for composite and wrapper kernels
- `test_batched_operations.py`: Tests for batched operations, NaN handling, and dimension handling

## Writing Tests

When adding new tests:

1. Use the fixtures defined in `conftest.py` for sample data
2. Follow the existing test structure and naming conventions
3. Test both functionality and edge cases
4. Include docstrings describing what each test validates

Example:
```python
def test_new_feature(self, sample_1d_data):
    """Test that new feature works correctly."""
    kernel = SomeKernel(param=1.0)
    x1, x2 = sample_1d_data
    result = kernel(x1, x2)
    assert result.shape == (x1.shape[0], x2.shape[0])
    assert jnp.all(jnp.isfinite(result))
```

## Test Coverage

Aim for high test coverage, especially for:
- Core kernel computations
- Dimension handling
- NaN propagation
- Batched operations with distinct hyperparameters
- Kernel composition