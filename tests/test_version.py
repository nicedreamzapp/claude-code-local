"""The release version lives in VERSION; everything that prints it must agree."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = (ROOT / "VERSION").read_text().strip()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION)


def test_server_version_matches():
    src = (ROOT / "proxy" / "server.py").read_text()
    m = re.search(r'^__version__ = "([^"]+)"', src, re.M)
    assert m and m.group(1) == VERSION


def test_installer_banners_match():
    assert f"Claude Code Local v{VERSION}" in (ROOT / "install.sh").read_text()
    assert f"Claude Code Local v{VERSION}" in (ROOT / "install.ps1").read_text()
