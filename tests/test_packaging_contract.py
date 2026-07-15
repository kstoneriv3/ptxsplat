import subprocess
import sys
from pathlib import Path

import ptxsplat


ROOT = Path(__file__).resolve().parents[1]


def test_version_contract():
    assert ptxsplat.__version__ == "0.1.0"
    assert ptxsplat.__gsplat_version__ == "1.5.3"


def test_gsplat_overload_is_fresh():
    subprocess.run(
        [sys.executable, "compat/gsplat_overload/generate.py", "--check"],
        cwd=ROOT,
        check=True,
    )
