"""Reconstruct a released slice from its k-space, maps, and acquisition mask."""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .dataset import SlamT2
from .recon import cg_sense


def _save_npz(path, **arrays):
    """Publish a complete file without replacing an existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--lam", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.slot < 0:
        parser.error("slot must be nonnegative")
    if args.iters < 1:
        parser.error("iters must be positive")
    if not math.isfinite(args.tol) or args.tol <= 0:
        parser.error("tol must be positive and finite")
    if not math.isfinite(args.lam) or args.lam < 0:
        parser.error("lam must be nonnegative and finite")
    torch.set_num_threads(args.threads)
    ds = SlamT2(args.root)
    if args.slot >= len(ds):
        parser.error(f"slot must be less than {len(ds)}")
    item = ds[args.slot]

    def tensor(name):
        return torch.from_numpy(np.array(item[name], copy=True)).to(args.device)

    image, residual, steps = cg_sense(
        tensor("ksp")[None],
        tensor("mps")[None],
        tensor("mask"),
        lam=args.lam,
        iters=args.iters,
        tol=args.tol,
    )
    reconstruction = image[0].cpu().numpy()
    denominator = max(float(np.linalg.norm(item["gt"])), np.finfo(np.float32).tiny)
    difference = float(np.linalg.norm(reconstruction - item["gt"])) / denominator
    converged = bool((residual < args.tol).all())
    _save_npz(
        args.output,
        reconstruction=reconstruction,
        slot=np.int64(args.slot),
        normal_residual=residual.cpu().numpy(),
        iterations=steps.cpu().numpy(),
        converged=np.bool_(converged),
        lam=np.float64(args.lam),
        tol=np.float64(args.tol),
        max_iterations=np.int64(args.iters),
        device=np.str_(args.device),
        threads=np.int64(args.threads),
        numpy_version=np.str_(np.__version__),
        torch_version=np.str_(torch.__version__),
    )
    print(
        json.dumps(
            {
                "normal_residual": float(residual[0]),
                "iterations": int(steps[0]),
                "relative_difference_from_reference": difference,
                "converged": converged,
            }
        )
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
