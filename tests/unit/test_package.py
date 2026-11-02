"""Packaging canary: the installed distribution and the package agree on the version."""

from importlib.metadata import version

import nomad


def test_package_version_matches_distribution() -> None:
    assert nomad.__version__ == version("nomad")
