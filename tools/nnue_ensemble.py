"""Average several trained networks into one file the engine can load.

With a clipped-ReLU hidden layer the average of k networks' outputs is exactly
one network whose hidden layer is the k hidden layers side by side and whose
output weights are each divided by k. So an ensemble needs no engine change;
it is just a wider net, and inference costs k times as much.

    python -m tools.nnue_ensemble a.npz b.npz c.npz --out nbchess/nnue.safetensors

The output is a .safetensors file (the format the platform names as allowed
for weights), written by nbchess.nnue.save_safetensors.
"""
import argparse
from pathlib import Path

import numpy as np

from nbchess.nnue import save_safetensors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nets", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=None, help="override output damping")
    args = ap.parse_args()
    zs = [np.load(p) for p in args.nets]
    k = len(zs)
    h = zs[0]["b1"].shape[0]
    for z in zs:
        assert z["b1"].shape[0] == h, "all nets must share a hidden size"
        assert int(z["residual"]) == int(zs[0]["residual"]), "mixing residual and plain nets"
    w1 = np.concatenate([z["w1"] for z in zs], axis=1).astype(np.float32)       # 769 x kH
    b1 = np.concatenate([z["b1"] for z in zs]).astype(np.float32)               # kH
    us = np.concatenate([z["w2"][:h] for z in zs]) / k
    them = np.concatenate([z["w2"][h:] for z in zs]) / k
    w2 = np.concatenate([us, them]).astype(np.float32)                          # 2kH
    b2 = np.float32(sum(float(z["b2"]) for z in zs) / k)
    scale = np.float32(args.scale) if args.scale is not None else (
        zs[0]["scale"] if "scale" in zs[0] else np.float32(1.0))
    save_safetensors(args.out, {
        "w1": w1, "b1": b1, "w2": w2,
        "b2": np.asarray([b2], dtype=np.float32),
        "residual": np.asarray([float(zs[0]["residual"])], dtype=np.float32),
        "scale": np.asarray([float(scale)], dtype=np.float32)})
    print(f"wrote {args.out}: {k} nets, hidden {k * h}, scale {float(scale)}")


if __name__ == "__main__":
    main()
