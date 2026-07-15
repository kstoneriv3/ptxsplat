import os
import subprocess
import sys
from pathlib import Path

import ptxsplat


ROOT = Path(__file__).resolve().parents[1]
OVERLOAD_ROOT = ROOT / "compat" / "gsplat_overload"


def test_version_contract():
    assert ptxsplat.__version__ == "0.1.0"
    assert ptxsplat.__gsplat_version__ == "1.5.3"


def test_gsplat_overload_is_fresh():
    subprocess.run(
        [sys.executable, "compat/gsplat_overload/generate.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_gsplat_overload_forwards_public_imports():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(OVERLOAD_ROOT), env.get("PYTHONPATH")))
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import gsplat, ptxsplat; "
                "from gsplat.cuda._wrapper import rasterize_to_pixels; "
                "from gsplat.rendering import rasterization; "
                "assert gsplat.rasterization is ptxsplat.rasterization; "
                "assert rasterization is ptxsplat.rasterization; "
                "assert rasterize_to_pixels is ptxsplat.rasterize_to_pixels"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
