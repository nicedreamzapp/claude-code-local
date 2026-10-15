"""Run the existing standalone checks in scripts/ as part of the suite.

Each one stubs mlx at import and asserts at module level, so each runs in its
own interpreter. scripts/test_mlx_server.py is left out on purpose: it needs a
real model served on localhost:4000.
"""
import subprocess
import sys

import pytest

from conftest import ROOT

SCRIPTS = sorted(p for p in (ROOT / "scripts").glob("test_*.py") if p.name != "test_mlx_server.py")


def test_scripts_found():
    assert len(SCRIPTS) >= 5


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_standalone_script_passes(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "all checks passed" in r.stdout


@pytest.mark.skip(reason="needs a model served by proxy/server.py on localhost:4000 "
                         "(Apple Silicon + MLX); run scripts/test_mlx_server.py by hand")
def test_live_model_harness():
    pass
