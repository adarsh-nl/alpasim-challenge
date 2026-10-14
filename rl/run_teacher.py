"""
Diffusion-Planner as a privileged teacher over the AlpaSim trajdata cache.

Reads scenes via trajdata_to_dp.py, runs the released checkpoint, and writes
ego trajectories (+ neighbour predictions) to disk as distillation targets.

The teacher sees the vector map, agent tracks and traffic lights -- none of
which reach the egodriver at inference. That asymmetry is the entire point:
the student learns camera -> trajectory, supervised by a model that had
ground-truth geometry.

Usage
-----
    # smoke test, 2 scenes, CPU or GPU
    python run_teacher.py --root $ALPASIM_NUPLAN_ROOT \\
        --repo ./dp_repo --ckpt ./dp_ckpt --limit 2 --out /tmp/teacher

    # sharded, for an sbatch array
    python run_teacher.py --root $ALPASIM_NUPLAN_ROOT \\
        --repo ./dp_repo --ckpt ./dp_ckpt \\
        --shard $SLURM_ARRAY_TASK_ID --n-shards 64 --out ./teacher_labels

Notes
-----
* Inputs are passed RAW. The model's ObservationNormalizer scales them and
  then re-zeros masked rows; pre-normalizing would make zero padding nonzero,
  unmasking every pad row into a phantom agent/lane.
* Output prediction is [B, 1 + predicted_neighbor_num, future_len, 4] where
  the trailing 4 is (x, y, cos, sin) in the ego frame. Index 0 is ego.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------

def _stub_mmengine() -> None:
    """Satisfy diffusion_planner.utils.train_utils' `from mmengine import fileio`.

    mmengine is only reached via openjson(), which our path never calls --
    Config loads args.json with plain json.load. Stubbing avoids installing the
    OpenMMLab stack into the venv that also runs the simulator.
    """
    import types
    if "mmengine" in sys.modules:
        return

    def _load(path, *a, **kw):
        with open(path, "r") as f:
            return json.load(f)

    def _get_text(path, *a, **kw):
        with open(path, "r") as f:
            return f.read()

    fileio = types.ModuleType("mmengine.fileio")
    fileio.load = _load
    fileio.get_text = _get_text
    fileio.get = _get_text
    fileio.dump = lambda obj, path, *a, **kw: Path(path).write_text(json.dumps(obj))
    fileio.exists = lambda path, *a, **kw: Path(path).exists()

    mm = types.ModuleType("mmengine")
    mm.fileio = mm.__dict__["fileio"] = fileio
    sys.modules["mmengine"] = mm
    sys.modules["mmengine.fileio"] = fileio


def load_model(repo: Path, ckpt_dir: Path, device: str, use_ema: bool = True):
    """Build Diffusion_Planner and load the released checkpoint."""
    sys.path.insert(0, str(repo))
    _stub_mmengine()

    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from diffusion_planner.utils.config import Config

    config = Config(str(ckpt_dir / "args.json"), guidance_fn=None)
    config.device = device

    model = Diffusion_Planner(config)

    blob = torch.load(ckpt_dir / "model.pth", map_location="cpu", weights_only=False)
    key = "ema_state_dict" if (use_ema and "ema_state_dict" in blob) else "model"
    sd = blob[key]

    # checkpoint was saved under DDP -> strip the wrapper prefix
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    model.to(device).eval()

    # decoder needs the normalizers for the inverse transform on x0
    dec = model.decoder.decoder
    if getattr(dec, "_state_normalizer", None) is None:
        dec._state_normalizer = config.state_normalizer
    if getattr(dec, "_observation_normalizer", None) is None:
        dec._observation_normalizer = config.observation_normalizer

    return model, config


# ----------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------

_MODEL_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
)


def to_batch(samples: list[dict], device: str) -> dict[str, torch.Tensor]:
    """Stack adapter dicts into a device batch. Raw values -- no normalization."""
    out = {}
    for k in _MODEL_KEYS:
        if k not in samples[0]:
            continue
        arr = np.stack([np.asarray(s[k]) for s in samples])
        dtype = torch.bool if arr.dtype == np.bool_ else torch.float32
        out[k] = torch.as_tensor(arr, dtype=dtype, device=device)
    return out


@torch.no_grad()
def predict(model, config, batch: dict[str, torch.Tensor]) -> np.ndarray:
    """-> [B, 1 + predicted_neighbor_num, future_len, 4], ego frame."""
    inputs = config.observation_normalizer(batch)
    _, dec_out = model(inputs)
    pred = dec_out["prediction"]
    return pred.detach().float().cpu().numpy()


# ----------------------------------------------------------------------------
# Sanity checks on teacher output
# ----------------------------------------------------------------------------

def audit(ego_traj: np.ndarray, dt: float = 0.1) -> dict:
    """Kinematic plausibility of one ego trajectory [future_len, 4]."""
    xy = ego_traj[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = step / dt
    accel = np.diff(speed) / dt
    return {
        "len_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "path_m": float(step.sum()),
        "v_mean": float(speed.mean()),
        "v_max": float(speed.max()),
        "a_absmax": float(np.abs(accel).max()) if len(accel) else 0.0,
        "backwards": bool(xy[-1, 0] < xy[0, 0] - 1.0),
        "finite": bool(np.isfinite(ego_traj).all()),
    }


def audit_flags(a: dict) -> list[str]:
    flags = []
    if not a["finite"]:
        flags.append("NONFINITE")
    if a["v_max"] > 40.0:
        flags.append("V_MAX>40")
    if a["a_absmax"] > 12.0:
        flags.append("ACCEL>12")
    if a["backwards"]:
        flags.append("BACKWARDS")
    return flags


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="ALPASIM_NUPLAN_ROOT")
    ap.add_argument("--repo", required=True, type=Path, help="Diffusion-Planner clone")
    ap.add_argument("--ckpt", required=True, type=Path, help="dir with args.json + model.pth")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--adapter", type=Path, default=Path(__file__).parent,
                    help="dir containing trajdata_to_dp.py")
    ap.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    ap.add_argument("--stride", type=int, default=5, help="5 => 2Hz sim cadence")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--audit-every", type=int, default=50)
    args = ap.parse_args()

    sys.path.insert(0, str(args.adapter))
    import trajdata_to_dp as A
    from trajdata import MapAPI

    print(f"device={args.device}  ema={not args.no_ema}")
    model, config = load_model(args.repo, args.ckpt, args.device,
                               use_ema=not args.no_ema)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model loaded: {n_param/1e6:.1f}M params, "
          f"future_len={config.future_len}, "
          f"predicted_neighbor_num={config.predicted_neighbor_num}")

    cache = args.root / "nuplan_test"
    scenes = sorted(p for p in cache.iterdir() if p.is_dir() and p.name != "maps")
    if args.limit:
        scenes = scenes[: args.limit]
    scenes = scenes[args.shard :: args.n_shards]
    print(f"shard {args.shard}/{args.n_shards}: {len(scenes)} scenes")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    api = MapAPI(args.root)
    map_cache: dict[str, object] = {}

    pending: list[dict] = []
    n_done = 0
    n_flagged = 0
    t0 = time.time()

    def flush():
        nonlocal pending, n_done, n_flagged
        if not pending:
            return
        batch = to_batch(pending, args.device)
        pred = predict(model, config, batch)

        for i, s in enumerate(pending):
            ego = pred[i, 0]           # [future_len, 4]
            neigh = pred[i, 1:]        # [predicted_neighbor_num, future_len, 4]
            a = audit(ego)
            flags = audit_flags(a)
            if flags:
                n_flagged += 1
                print(f"  [FLAG {','.join(flags)}] {s['token']} "
                      f"v_max={a['v_max']:.1f} a={a['a_absmax']:.1f} "
                      f"len={a['len_m']:.1f}m")
            if args.out:
                np.savez_compressed(
                    args.out / f"{s['map_name']}_{s['token']}.npz",
                    ego_future=ego.astype(np.float32),
                    neighbor_future=neigh.astype(np.float32),
                    token=s["token"], map_name=s["map_name"],
                )
            n_done += 1
            if args.audit_every and n_done % args.audit_every == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"  {n_done} done ({rate:.1f}/s, {n_flagged} flagged) "
                      f"last: v_mean={a['v_mean']:.1f} path={a['path_m']:.1f}m")
        pending = []

    for sd in scenes:
        try:
            scene = A.load_scene(sd)
        except Exception as e:
            print(f"  [skip] {sd.name}: {type(e).__name__}: {e}")
            continue

        if scene.location not in map_cache:
            map_cache[scene.location] = api.get_map(f"nuplan_test:{scene.location}")
        vm = map_cache[scene.location]

        for ts in A.valid_timesteps(scene)[:: args.stride]:
            try:
                pending.append(A.build_sample(scene, vm, ts))
            except Exception as e:
                print(f"  [skip] {scene.name}@{ts}: {type(e).__name__}: {e}")
                continue
            if len(pending) >= args.batch_size:
                flush()
    flush()

    dt = time.time() - t0
    print(f"\ndone: {n_done} samples in {dt:.1f}s ({n_done/max(dt,1e-6):.1f}/s), "
          f"{n_flagged} flagged ({100*n_flagged/max(n_done,1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
