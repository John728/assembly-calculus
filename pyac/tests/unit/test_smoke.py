"""
Smoke test: verify pyac package is importable and has correct version.

TDD: Smoke test written first to verify package structure.
"""

import pyac


def test_version():
    """Test that pyac package exports __version__ correctly."""
    assert pyac.__version__ == "0.1.0"
