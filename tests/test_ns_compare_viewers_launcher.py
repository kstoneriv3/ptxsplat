import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ns-compare-viewers.sh"


def test_ns_compare_viewer_launcher_contract() -> None:
    text = SCRIPT.read_text()

    assert "ptxsplat-ns-compare-upstream" in text
    assert "ptxsplat-ns-compare-sm120" in text
    assert 'DEFAULT_SCENE="tiny-synthetic"' in text
    assert "ptxsplat.ns-compare.scene=${scene}" in text
    assert "--network host" in text
    assert "--gpus 'device=0'" in text
    assert "USER=ptxsplat" in text
    assert "env TORCHDYNAMO_DISABLE=1" in text
    assert "ns-viewer --load-config" in text

    assert "/workspace/.bcodex/gsplat-1.5.3" in text
    assert "/workspace/compat/gsplat_overload:/workspace" in text
    assert "PTXSPLAT_BACKEND=${backend}" in text
    assert '"sm120"' in text

    assert (
        'UPSTREAM_CONFIG="/workspace/results/ns-compare/upstream/${SELECTED_SCENE}/splatfacto/matched-1000/config.yml"'
        in text
    )
    assert (
        'PTXSPLAT_CONFIG="/workspace/results/ns-compare/ptxsplat/${SELECTED_SCENE}/splatfacto/matched-1000/config.yml"'
        in text
    )
    assert '--viewer.websocket-port "${port}"' in text
    assert '"7007"' in text
    assert '"7008"' in text


def test_ns_compare_viewer_launcher_documents_scene_selection() -> None:
    result = subprocess.run(
        [SCRIPT, "help"], check=True, capture_output=True, text=True
    )

    assert "up [tiny-synthetic|bonsai]" in result.stdout
    assert "default scene for `up` is tiny-synthetic" in result.stdout
