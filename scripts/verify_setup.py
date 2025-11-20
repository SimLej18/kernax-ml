#!/usr/bin/env python3
"""
Verification script to check that all package files are in place
and properly configured.
"""

import os
import sys
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_file(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"{GREEN}✓{RESET} {description}")
        return True
    else:
        print(f"{RED}✗{RESET} {description} - NOT FOUND: {filepath}")
        return False

def check_placeholder(filepath, description):
    """Check if a file contains placeholders that need to be filled."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if '[PLACEHOLDER' in content:
                print(f"{YELLOW}⚠{RESET} {description} - Contains placeholders to fill")
                return False
            else:
                print(f"{GREEN}✓{RESET} {description} - No placeholders")
                return True
    except FileNotFoundError:
        print(f"{RED}✗{RESET} {description} - File not found: {filepath}")
        return False

def main():
    print("=" * 60)
    print("Kernax Package Setup Verification")
    print("=" * 60)
    print()

    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)

    all_good = True

    # Core package files
    print("Core Package Files:")
    print("-" * 60)
    all_good &= check_file("pyproject.toml", "Package configuration")
    all_good &= check_file("setup.py", "Setup script")
    all_good &= check_file("MANIFEST.in", "Distribution manifest")
    all_good &= check_file("requirements.txt", "Core dependencies")
    all_good &= check_file("requirements-dev.txt", "Dev dependencies")
    all_good &= check_file("LICENSE", "License file")
    print()

    # Documentation
    print("Documentation:")
    print("-" * 60)
    all_good &= check_file("README.md", "Main README")
    all_good &= check_file("CLAUDE.md", "Architecture guide")
    all_good &= check_file("CONTRIBUTING.md", "Contributing guide")
    all_good &= check_file("CHANGELOG.md", "Version history")
    all_good &= check_file("SETUP_CHECKLIST.md", "Setup checklist")
    print()

    # Kernax package
    print("Kernax Package:")
    print("-" * 60)
    all_good &= check_file("kernax/__init__.py", "Package init")
    all_good &= check_file("kernax/py.typed", "Type hints marker")
    all_good &= check_file("kernax/AbstractKernel.py", "Abstract kernel")
    all_good &= check_file("kernax/RBFKernel.py", "RBF kernel")
    all_good &= check_file("kernax/OperatorKernels.py", "Operator kernels")
    all_good &= check_file("kernax/WrapperKernels.py", "Wrapper kernels")
    print()

    # Tests
    print("Test Suite:")
    print("-" * 60)
    all_good &= check_file("tests/__init__.py", "Tests init")
    all_good &= check_file("tests/conftest.py", "Pytest config")
    all_good &= check_file("tests/test_base_kernels.py", "Base kernel tests")
    all_good &= check_file("tests/test_composite_kernels.py", "Composite kernel tests")
    all_good &= check_file("tests/test_batched_operations.py", "Batched ops tests")
    print()

    # Development tools
    print("Development Tools:")
    print("-" * 60)
    all_good &= check_file(".gitignore", "Git ignore file")
    all_good &= check_file(".editorconfig", "Editor config")
    all_good &= check_file(".python-version", "Python version")
    all_good &= check_file("Makefile", "Make commands")
    all_good &= check_file(".github/workflows/tests.yml", "Test workflow")
    all_good &= check_file(".github/workflows/lint.yml", "Lint workflow")
    print()

    # Check for placeholders
    print("Placeholder Check:")
    print("-" * 60)
    placeholders_ok = True
    placeholders_ok &= check_placeholder("pyproject.toml", "pyproject.toml")
    placeholders_ok &= check_placeholder("kernax/__init__.py", "kernax/__init__.py")
    placeholders_ok &= check_placeholder("LICENSE", "LICENSE")
    placeholders_ok &= check_placeholder("CHANGELOG.md", "CHANGELOG.md")
    print()

    # Try to import the package
    print("Package Import:")
    print("-" * 60)
    try:
        import kernax
        print(f"{GREEN}✓{RESET} Successfully imported kernax")
        print(f"  Version: {kernax.__version__}")
        print(f"  Author: {kernax.__author__}")
        print(f"  License: {kernax.__license__}")

        # Try to instantiate a kernel
        kernel = kernax.RBFKernel(length_scale=1.0, variance=1.0)
        print(f"{GREEN}✓{RESET} Successfully created RBFKernel instance")

    except ImportError as e:
        print(f"{RED}✗{RESET} Failed to import kernax: {e}")
        all_good = False
    except Exception as e:
        print(f"{RED}✗{RESET} Error testing kernax: {e}")
        all_good = False
    print()

    # Summary
    print("=" * 60)
    if all_good and placeholders_ok:
        print(f"{GREEN}✓ All checks passed! Package is ready.{RESET}")
        print()
        print("Next steps:")
        print("  1. Run tests: make test")
        print("  2. Check code style: make lint")
        print("  3. See SETUP_CHECKLIST.md for publishing steps")
        return 0
    elif all_good and not placeholders_ok:
        print(f"{YELLOW}⚠ Package structure is complete but has placeholders.{RESET}")
        print()
        print("Action required:")
        print("  1. Fill in all [PLACEHOLDER] values")
        print("  2. See SETUP_CHECKLIST.md for details")
        return 1
    else:
        print(f"{RED}✗ Some checks failed. Please review errors above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())