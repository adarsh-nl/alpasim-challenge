#!/usr/bin/env python3
"""D2 — token -> camera-observation resolver for the distillation student.

The teacher labels are targets only (ego_future / neighbor_future / token /
map_name). This module resolves a label token to the *observation side*: which
scene, which frame, which cameras, and with what geometry — so a frame source
can produce pixels for it.

THE OBSERVATION FORK, stated not hidden
---------------------------------------
There are no real camera frames in the staged dataset. Every
`frame_info['cams'][C]['data_path']` points at a raw nuPlan JPEG
(`<log>/CAM_F0/<token>.jpg`) and **none of them are staged** — checked under the
dataset root, navtest/, and the asset dir. What IS staged per renderable scene:

  background/<scene>.ckpt      Gaussian splat  (~415 MB)
  road_height_map/*.npy
  video_scene_dict.pkl         camera intrinsics/extrinsics/distortion, 8 cams,
                               per-frame ego pose, and privileged gt_boxes

So the three candidate frame sources are:

  RENDERED  render the splat with the same MTGS path the evaluator uses.
            Matches eval exactly -> NO domain gap. Requires a GPU.
  REAL      raw nuPlan sensor JPEGs. NOT AVAILABLE locally; staging them is a
            multi-TB decision AND would introduce the domain gap that RENDERED
            avoids. Implemented as an explicit error, not a silent fallback.
  SYNTHETIC deterministic procedural frames. For CPU unit tests only.

This module does not choose. `resolve()` is source-agnostic; the choice is the
`FrameSource` you pass in. See reports/d2_observation_pipeline.md.

Alignment (validated): label `ts` is a trajdata timestep (dt=0.10) while
`frame_infos` are video frames over a different, longer span. They are matched
by nearest ego position in a common global frame
(`global = ego2global_translation + global2world_translation`). Measured
residuals on a real scene: median 0.091 m, max 0.178 m, monotonic in ts.

CPU-only. Read-only w.r.t. the dataset.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np

CANONICAL_ROOT = "/datasets/eemcs/ps/alpasim-nuplan-2026"


def _resolve_root() -> str:
    """Prefer $ALPASIM_NUPLAN_ROOT, but only if it actually holds the dataset.

    Some shells still export the README's older `~/data/alpasim-nuplan`, which
    does not exist. Silently trusting it produces a confusing FileNotFoundError
    deep in a loader, so validate up front and fall back to the canonical
    shared path.
    """
    env = os.environ.get("ALPASIM_NUPLAN_ROOT", "")
    if env and os.path.isdir(os.path.join(env, "navtest", "assets")):
        return env
    return CANONICAL_ROOT


DATASET_ROOT = _resolve_root()
# The eval spec's camera set, in proto order.
ALL_CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2",
            "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
# Default to the forward camera alone: see the decode-cost four-axis in the
# report -- 8-camera decode is 6-14x over the aggregate per-call budget.
DEFAULT_CAMS = ("CAM_F0",)


class SceneNotRenderable(FileNotFoundError):
    """No Gaussian splat staged for this scene, so no observation is possible."""


@dataclass
class CameraSpec:
    name: str
    intrinsic: np.ndarray          # (3,3)
    distortion: np.ndarray         # (5,)
    sensor2ego_translation: np.ndarray   # (3,)
    sensor2ego_rotation: np.ndarray      # (4,) quaternion wxyz
    data_path: str                 # raw-JPEG path; NOT staged, kept for provenance
    valid: bool


@dataclass
class ObservationSpec:
    """Everything a frame source needs to produce pixels for one label token."""
    token: str
    scene: str
    label_ts: int
    frame_idx: int
    align_residual_m: float
    splat_ckpt: str
    ego_global: np.ndarray                 # (2,) matched ego position
    cams: dict[str, CameraSpec] = field(default_factory=dict)
    # privileged per-frame agent boxes -- used by the D1 far-horizon audit,
    # never fed to the student
    gt_boxes: np.ndarray | None = None
    gt_names: np.ndarray | None = None


# ---------------------------------------------------------------------------
# dataset access
# ---------------------------------------------------------------------------

def asset_dir(scene: str, root: str = DATASET_ROOT) -> str:
    return os.path.join(root, "navtest", "assets", scene)


def is_renderable(scene: str, root: str = DATASET_ROOT) -> bool:
    """A scene is renderable iff its Gaussian splat is staged."""
    d = asset_dir(scene, root)
    return os.path.isdir(d) and os.path.isdir(os.path.join(d, "background"))


def splat_path(scene: str, root: str = DATASET_ROOT) -> str:
    bg = os.path.join(asset_dir(scene, root), "background")
    if not os.path.isdir(bg):
        raise SceneNotRenderable(scene)
    cks = [f for f in os.listdir(bg) if f.endswith(".ckpt")]
    if not cks:
        raise SceneNotRenderable(f"{scene}: background/ has no .ckpt")
    return os.path.join(bg, cks[0])


@lru_cache(maxsize=64)
def _load_video_dict(scene: str, root: str = DATASET_ROOT):
    p = os.path.join(asset_dir(scene, root), "video_scene_dict.pkl")
    if not os.path.exists(p):
        raise SceneNotRenderable(f"{scene}: no video_scene_dict.pkl")
    with open(p, "rb") as f:
        d = pickle.load(f)
    # one entry, keyed "<scene>-<n>"
    return d[next(iter(d))]


@lru_cache(maxsize=64)
def _ego_track_global(scene: str, root: str = DATASET_ROOT) -> np.ndarray:
    """Trajdata ego (x,y) per timestep, in global coords. Shape (T,2)."""
    import pandas as pd
    f = os.path.join(root, "nuplan_test", scene, "agent_data_dt0.10.feather")
    df = pd.read_feather(f)
    ego = df[df["agent_id"] == "ego"].sort_values("scene_ts")
    return ego[["x", "y"]].values.astype(np.float64)


@lru_cache(maxsize=64)
def _frame_track_global(scene: str, root: str = DATASET_ROOT) -> np.ndarray:
    """Video frame ego (x,y) lifted into the same global frame. Shape (F,2)."""
    v = _load_video_dict(scene, root)
    g2w = np.asarray(v["global2world_translation"], dtype=np.float64)[:2]
    loc = np.array([np.asarray(fi["ego2global_translation"], dtype=np.float64)[:2]
                    for fi in v["frame_infos"]])
    return loc + g2w


def _quat_wxyz(q) -> np.ndarray:
    """pyquaternion Quaternion or array-like -> (4,) wxyz float array."""
    for attr in ("elements",):
        if hasattr(q, attr):
            return np.asarray(getattr(q, attr), dtype=np.float64)
    return np.asarray(q, dtype=np.float64)


def resolve(token: str, cams=DEFAULT_CAMS, root: str = DATASET_ROOT,
            want_gt: bool = False) -> ObservationSpec:
    """Resolve a label token to an ObservationSpec.

    Raises SceneNotRenderable if the scene has no staged splat — which is the
    common case: only ~12% of teacher-labelled scenes are renderable.
    """
    scene, _, ts_s = token.rpartition("_")
    ts = int(ts_s)

    ck = splat_path(scene, root)                 # raises if not renderable
    ego = _ego_track_global(scene, root)
    frames = _frame_track_global(scene, root)
    if ts >= len(ego):
        raise IndexError(f"{token}: ts {ts} beyond trajdata length {len(ego)}")

    d = np.linalg.norm(frames - ego[ts][None, :], axis=1)
    fi_idx = int(d.argmin())
    v = _load_video_dict(scene, root)
    fi = v["frame_infos"][fi_idx]

    specs = {}
    for c in cams:
        cd = fi["cams"].get(c)
        if cd is None:
            continue
        specs[c] = CameraSpec(
            name=c,
            intrinsic=np.asarray(cd["cam_intrinsic"], dtype=np.float64),
            distortion=np.asarray(cd["distortion"], dtype=np.float64),
            sensor2ego_translation=np.asarray(cd["sensor2ego_translation"],
                                              dtype=np.float64),
            sensor2ego_rotation=_quat_wxyz(cd["sensor2ego_rotation"]),
            data_path=str(cd.get("data_path", "")),
            valid=bool(cd.get("valid", True)),
        )

    return ObservationSpec(
        token=token, scene=scene, label_ts=ts, frame_idx=fi_idx,
        align_residual_m=float(d[fi_idx]), splat_ckpt=ck,
        ego_global=ego[ts], cams=specs,
        gt_boxes=np.asarray(fi["gt_boxes"]) if want_gt else None,
        gt_names=np.asarray(fi["gt_names"]) if want_gt else None,
    )


# ---------------------------------------------------------------------------
# frame sources — the swappable half
# ---------------------------------------------------------------------------

class FrameSource(Protocol):
    """Produces (C,H,W) float32 in [0,1] for one camera of one ObservationSpec."""

    def frame(self, spec: ObservationSpec, cam: str) -> np.ndarray: ...


@dataclass
class SyntheticFrameSource:
    """Deterministic procedural frames. CPU tests only — never for training.

    Content is a function of (token, cam) so tests can assert determinism and
    that different tokens yield different observations.
    """
    hw: tuple[int, int] = (180, 320)

    def frame(self, spec: ObservationSpec, cam: str) -> np.ndarray:
        h, w = self.hw
        seed = abs(hash((spec.token, cam))) % (2**31)
        rng = np.random.default_rng(seed)
        base = rng.random((3, h, w), dtype=np.float32) * 0.2
        # a deterministic horizon band so the image is not pure noise
        base[:, : h // 2, :] += 0.5
        return np.clip(base, 0.0, 1.0)


@dataclass
class RenderedFrameSource:
    """Reads frames pre-rendered from the splats by a staged GPU pass.

    Layout: <cache>/<scene>/<frame_idx:04d>_<cam>.jpg — matches eval exactly
    (same renderer), so there is no domain gap. Rendering itself is GPU work and
    is NOT done here.
    """
    cache_dir: str
    hw: tuple[int, int] = (180, 320)
    use_draft: bool = True          # JPEG DCT-scaled decode; ~2x cheaper

    def path(self, spec: ObservationSpec, cam: str) -> str:
        return os.path.join(self.cache_dir, spec.scene,
                            f"{spec.frame_idx:04d}_{cam}.jpg")

    def frame(self, spec: ObservationSpec, cam: str) -> np.ndarray:
        from PIL import Image
        p = self.path(spec, cam)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"no rendered frame at {p} -- run the staged render pass first")
        im = Image.open(p)
        if self.use_draft:
            im.draft("RGB", (self.hw[1], self.hw[0]))
        im = im.convert("RGB").resize((self.hw[1], self.hw[0]))
        return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)


@dataclass
class RealFrameSource:
    """Raw nuPlan sensor JPEGs. NOT AVAILABLE — fails loudly by design.

    Kept so the fork stays visible in code rather than becoming a silent
    assumption. If sensor data is ever staged, point `sensor_root` at it.
    """
    sensor_root: str = ""

    def frame(self, spec: ObservationSpec, cam: str) -> np.ndarray:
        dp = spec.cams[cam].data_path
        if not self.sensor_root:
            raise SceneNotRenderable(
                "REAL frames are not staged in this dataset. "
                f"frame_info references {dp!r}, which does not exist under the "
                "dataset root. Staging raw nuPlan sensor data is a multi-TB "
                "decision AND reintroduces the render/real domain gap that the "
                "RENDERED source avoids. Use RenderedFrameSource.")
        p = os.path.join(self.sensor_root, dp)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        from PIL import Image
        im = Image.open(p).convert("RGB")
        return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# dataset-level helper
# ---------------------------------------------------------------------------

def renderable_tokens(label_dir: str, root: str = DATASET_ROOT
                      ) -> tuple[list[str], list[str]]:
    """Split label tokens into (renderable, not_renderable)."""
    import glob
    ok, no = [], []
    have = set(os.listdir(os.path.join(root, "navtest", "assets")))
    for f in sorted(glob.glob(os.path.join(label_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        tok = str(z["token"])
        (ok if tok.rpartition("_")[0] in have else no).append(tok)
    return ok, no


if __name__ == "__main__":
    import sys
    lab = sys.argv[1] if len(sys.argv) > 1 else "rl/teacher_labels"
    ok, no = renderable_tokens(lab)
    print(f"labels: {len(ok)+len(no)}   renderable: {len(ok)}   not: {len(no)}")
    if ok:
        s = resolve(ok[0], cams=ALL_CAMS, want_gt=True)
        print(f"\n{s.token}\n  scene={s.scene}\n  ts={s.label_ts} -> frame={s.frame_idx}"
              f"  residual={s.align_residual_m:.3f} m")
        print(f"  cams={list(s.cams)}")
        print(f"  gt_boxes={None if s.gt_boxes is None else s.gt_boxes.shape}")
