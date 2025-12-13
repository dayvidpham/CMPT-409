#!/usr/bin/env python
"""
Check Python syntax compilation and imports for all engine modules.
Catches syntax errors and import errors before runtime.
"""

import sys
from pathlib import Path
import py_compile
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file(filepath):
    """Check if a Python file compiles and imports without errors."""
    # First check syntax
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        return False, f"Syntax Error: {str(e)}"

    # Then try to import it
    try:
        relative_path = filepath.relative_to(project_root)
        module_name = ".".join(relative_path.with_suffix("").parts)

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        return True, None
    except Exception as e:
        return False, f"Import Error: {str(e)}"


def main():
    """Check compilation of all Python files in the engine package."""
    print("=" * 70)
    print("CHECKING PYTHON SYNTAX COMPILATION")
    print("=" * 70)

    # Find all Python files in engine/
    engine_dir = project_root / "engine"
    python_files = sorted(engine_dir.rglob("*.py"))

    if not python_files:
        print("⚠️  No Python files found in engine/")
        return False

    failed = []
    passed = []

    for filepath in python_files:
        relative_path = filepath.relative_to(project_root)
        success, error = check_file(filepath)

        if success:
            print(f"  ✓ {relative_path}")
            passed.append(relative_path)
        else:
            print(f"  ✗ {relative_path}")
            print(f"     {error}")
            failed.append((relative_path, error))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {len(passed)}/{len(python_files)}")
    print(f"Failed: {len(failed)}/{len(python_files)}")

    if failed:
        print("\n❌ COMPILATION ERRORS:")
        for filepath, error in failed:
            print(f"  {filepath}")
            print(f"    {error}")
        print("=" * 70)
        return False
    else:
        print("\n✅ All files compile successfully!")
        print("=" * 70)
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
