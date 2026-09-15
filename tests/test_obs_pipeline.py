#!/usr/bin/env python
"""D2 — unit tests for the token -> camera-observation resolver, on shard 0.

CPU-only, read-only, no GPU, no renderer. Uses the real dataset metadata (which
is small) plus SyntheticFrameSource for pixels.

Run serially:
    ~/alpasim/.venv/bin/python -m pytest tests/test_obs_pipeline.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rl"))

obs = pytest.importorskip("obs_pipeline")

LABELS = str(REPO / "rl" / "teacher_labels")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(obs.DATASET_ROOT, "navtest", "assets")),
    reason="dataset not available")


@pytest.fixture(scope="module")
def split():
    ok, no = obs.renderable_tokens(LABELS)
    if not ok:
        pytest.skip("no renderable tokens in shard 0")
    return ok, no


# ---------------------------------------------------------------------------
# renderability
# ---------------------------------------------------------------------------

def test_renderable_split_is_consistent(split):
    ok, no = split
    assert len(ok) + len(no) == len(list(Path(LABELS).glob("*.npz")))
    # documents the actual coverage: most labels have no possible observation
    frac = len(ok) / (len(ok) + len(no))
    assert 0.0 < frac < 1.0
    assert all(obs.is_renderable(t.rpartition("_")[0]) for t in ok[:25])
    assert not any(obs.is_renderable(t.rpartition("_")[0]) for t in no[:25])


def test_non_renderable_raises_not_silently_degrades(split):
    """A missing splat must be a loud error, never an empty/black frame."""
    _, no = split
    with pytest.raises(obs.SceneNotRenderable):
        obs.resolve(no[0])


# ---------------------------------------------------------------------------
# alignment — the part that had to be derived rather than read off
# ---------------------------------------------------------------------------

def test_alignment_residual_is_subsize_of_a_car(split):
    """label ts -> frame must land within a fraction of a vehicle length."""
    ok, _ = split
    res = [obs.resolve(t).align_residual_m for t in ok[:40]]
    assert max(res) < 1.0, f"worst alignment residual {max(res):.3f} m"
    assert float(np.median(res)) < 0.5


def test_alignment_is_monotonic_within_a_scene(split):
    """Increasing label ts must map to non-decreasing frame_idx."""
    ok, _ = split
    by_scene: dict[str, list[str]] = {}
    for t in ok:
        by_scene.setdefault(t.rpartition("_")[0], []).append(t)
    scene = next(s for s, v in by_scene.items() if len(v) >= 4)
    toks = sorted(by_scene[scene], key=lambda t: int(t.rpartition("_")[2]))
    idx = [obs.resolve(t).frame_idx for t in toks]
    assert idx == sorted(idx), f"frame_idx not monotonic in ts: {idx}"
    assert len(set(idx)) == len(idx), f"distinct ts collapsed to one frame: {idx}"


def test_resolve_populates_camera_geometry(split):
    ok, _ = split
    s = obs.resolve(ok[0], cams=obs.ALL_CAMS)
    assert set(s.cams) == set(obs.ALL_CAMS), "not all 8 eval cameras resolved"
    c = s.cams["CAM_F0"]
    assert c.intrinsic.shape == (3, 3)
    assert c.distortion.shape == (5,)
    assert c.sensor2ego_translation.shape == (3,)
    assert c.sensor2ego_rotation.shape == (4,)
    assert np.isfinite(c.intrinsic).all()
    # focal lengths positive and plausible for a 1920x1080 sensor
    assert 100.0 < c.intrinsic[0, 0] < 5000.0


def test_splat_checkpoint_exists_for_renderable(split):
    ok, _ = split
    s = obs.resolve(ok[0])
    assert os.path.exists(s.splat_ckpt)
    assert s.splat_ckpt.endswith(".ckpt")


def test_privileged_gt_is_opt_in_only(split):
    """gt_boxes must be absent unless explicitly requested (never fed to student)."""
    ok, _ = split
    assert obs.resolve(ok[0]).gt_boxes is None
    s = obs.resolve(ok[0], want_gt=True)
    assert s.gt_boxes is not None and s.gt_boxes.ndim == 2 and s.gt_boxes.shape[1] == 7
    assert s.gt_names is not None and len(s.gt_names) == len(s.gt_boxes)


# ---------------------------------------------------------------------------
# frame sources — the swappable half
# ---------------------------------------------------------------------------

def test_synthetic_source_shape_dtype_range(split):
    ok, _ = split
    s = obs.resolve(ok[0])
    f = obs.SyntheticFrameSource().frame(s, "CAM_F0")
    assert f.shape == (3, 180, 320)
    assert f.dtype == np.float32
    assert 0.0 <= float(f.min()) and float(f.max()) <= 1.0


def test_synthetic_source_is_deterministic_and_token_dependent(split):
    ok, _ = split
    src = obs.SyntheticFrameSource()
    a = obs.resolve(ok[0])
    b = obs.resolve(ok[1])
    assert np.array_equal(src.frame(a, "CAM_F0"), src.frame(a, "CAM_F0")), \
        "synthetic frames must be reproducible"
    assert not np.array_equal(src.frame(a, "CAM_F0"), src.frame(b, "CAM_F0")), \
        "different tokens must yield different observations"
    assert not np.array_equal(src.frame(a, "CAM_F0"), src.frame(a, "CAM_L0")), \
        "different cameras must yield different observations"


def test_real_source_fails_loudly_with_an_explanation(split):
    """The fork must stay visible: REAL frames are absent, not silently empty."""
    ok, _ = split
    s = obs.resolve(ok[0])
    with pytest.raises(obs.SceneNotRenderable, match="not staged"):
        obs.RealFrameSource().frame(s, "CAM_F0")


def test_real_source_data_path_is_recorded_but_absent(split):
    """Provenance kept; bytes confirmed missing (this is why RENDERED is default)."""
    ok, _ = split
    s = obs.resolve(ok[0])
    dp = s.cams["CAM_F0"].data_path
    assert dp.endswith(".jpg") and "CAM_F0" in dp
    for base in (obs.DATASET_ROOT,
                 os.path.join(obs.DATASET_ROOT, "navtest"),
                 obs.asset_dir(s.scene)):
        assert not os.path.exists(os.path.join(base, dp)), (
            f"a real frame unexpectedly exists at {base}/{dp} -- the "
            f"observation-source analysis in the D2 report needs revisiting")


def test_rendered_source_missing_cache_is_explicit(split):
    ok, _ = split
    s = obs.resolve(ok[0])
    src = obs.RenderedFrameSource(cache_dir="/nonexistent-render-cache")
    with pytest.raises(FileNotFoundError, match="render pass"):
        src.frame(s, "CAM_F0")
    assert src.path(s, "CAM_F0").endswith(f"{s.frame_idx:04d}_CAM_F0.jpg")


def test_default_camera_set_is_forward_only():
    """Documents the budget-driven default; 8-cam decode is far over budget."""
    assert obs.DEFAULT_CAMS == ("CAM_F0",)
    assert len(obs.ALL_CAMS) == 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
