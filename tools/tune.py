"""Texel tuning: fit the evaluation weights to game outcomes.

The method is Peter Osterlund's. Take positions labelled with the result of the
game they came from, squash the evaluation through a sigmoid so it reads as a
win probability, and minimise the squared error against the actual results. The
weights that best predict outcomes are, by construction, the ones that best
describe a position.

Published measurements put tuning above any single evaluation term - Tcheran
gained +53 Elo from tuning an evaluation that was already complete. Our
piece-square tables are already tuned (PeSTO); what is untuned is every weight
in terms.py, which is what this fits.

    python -m tools.tune --data tuning.txt --sweeps 6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core.types import float64, int8, uint64

from nbchess.fen import set_fen
from nbchess.search import _W_RO as _W_RO_1D
from nbchess.search import evaluate_w
from nbchess.terms import (
    I_BI_EG,
    I_BI_MG,
    I_BP_EG,
    I_BP_MG,
    I_DBL_EG,
    I_DBL_MG,
    I_ISO_EG,
    I_ISO_MG,
    I_KN_EG,
    I_KN_MG,
    I_KS,
    I_PA_EG,
    I_PA_MG,
    I_QU_EG,
    I_QU_MG,
    I_RK_EG,
    I_RK_MG,
    N_BI,
    N_KN,
    N_PA,
    N_QU,
    N_RK,
    N_WEIGHTS,
    WEIGHTS,
)


@njit(float64(uint64[:, :], int8[:, :], float64[:], _W_RO_1D, float64),
      cache=False, nogil=True)
def loss(states, mbs, results, w, k):
    """Mean squared error between the sigmoid of the evaluation and the result.

    The evaluation is taken from White's point of view, because that is the
    point of view the labels are in.
    """
    total = 0.0
    n = states.shape[0]
    for i in range(n):
        sc = evaluate_w(states[i], mbs[i], w)
        if states[i][14] != uint64(0):
            sc = -sc
        p = 1.0 / (1.0 + 10.0 ** (-k * sc / 400.0))
        d = results[i] - p
        total += d * d
    return total / n


#: Weight on the engine's evaluation versus the game result. Stockfish trains
#: its own networks at 0.7 and the value is not delicate; the point is that the
#: evaluation carries a distinct label per position where the result does not.
LAMBDA = 0.7


def load(path: Path, k: float) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int8],
                                        npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Read `fen|result[|engine_cp]` rows.

    Returns the positions, a blended target, and a game id per row. The game id
    matters: positions from one game share a result, so a holdout split that
    cuts through a game leaks and flatters itself. Games are detected by the
    move counter resetting, since each game is written contiguously.
    """
    fens: list[str] = []
    targets: list[float] = []
    games: list[int] = []
    game = 0
    last_ply = 10**9
    for line in path.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        fen, result = parts[0], float(parts[1])
        ply = int(fen.split()[-1])
        if ply < last_ply:
            game += 1
        last_ply = ply
        if len(parts) >= 3:
            cp = float(parts[2])
            wdl = 1.0 / (1.0 + 10.0 ** (-k * cp / 400.0))
            targets.append(LAMBDA * wdl + (1.0 - LAMBDA) * result)
        else:
            targets.append(result)
        fens.append(fen)
        games.append(game)
    states = np.zeros((len(fens), 19), dtype=np.uint64)
    mbs = np.full((len(fens), 64), 12, dtype=np.int8)
    for i, fen in enumerate(fens):
        set_fen(states[i], mbs[i], fen)
    return (states, mbs, np.array(targets, dtype=np.float64),
            np.array(games, dtype=np.int64))


#: Curves are tuned by scale and offset rather than entry by entry.
#:
#: Fitting all 155 entries independently overfits badly: a knight with eight
#: safe squares is rare, so its parameter is fitted on a handful of positions
#: and comes out non-monotonic, which is not a mobility curve. Scaling a
#: hand-designed concave shape keeps the monotonicity - real chess knowledge
#: the data cannot supply - and lets the data set the magnitude, which is the
#: part that was actually guessed.
CURVES = [
    ("knight mob mg", I_KN_MG, N_KN), ("knight mob eg", I_KN_EG, N_KN),
    ("bishop mob mg", I_BI_MG, N_BI), ("bishop mob eg", I_BI_EG, N_BI),
    ("rook mob mg", I_RK_MG, N_RK), ("rook mob eg", I_RK_EG, N_RK),
    ("queen mob mg", I_QU_MG, N_QU), ("queen mob eg", I_QU_EG, N_QU),
    ("passed mg", I_PA_MG, N_PA), ("passed eg", I_PA_EG, N_PA),
]
SCALARS = [
    ("isolated mg", I_ISO_MG), ("isolated eg", I_ISO_EG),
    ("doubled mg", I_DBL_MG), ("doubled eg", I_DBL_EG),
    ("bishop pair mg", I_BP_MG), ("bishop pair eg", I_BP_EG),
    ("king safety scale", I_KS),
]



def build(theta: list[int]) -> npt.NDArray[np.int32]:
    """Map the small tuned vector back onto the full weight array.

    Returned readonly, because the evaluation is compiled for exactly one array
    signature and compiling a second copy of it for a writable array would cost
    real time out of the 60 second initialisation budget.
    """
    w = WEIGHTS.copy()
    for i, (_name, base, count) in enumerate(CURVES):
        scale, offset = theta[2 * i], theta[2 * i + 1]
        for j in range(count):
            w[base + j] = np.int32(int(WEIGHTS[base + j]) * scale // 100 + offset)
    for i, (_name, idx) in enumerate(SCALARS):
        w[idx] = np.int32(theta[2 * len(CURVES) + i])
    w.flags.writeable = False
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("tuning.txt"))
    ap.add_argument("--sweeps", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("tuned.txt"))
    args = ap.parse_args()

    # K is needed to convert engine centipawns into the target, and again to
    # squash our own evaluation. Fit it once against a plain-result target, then
    # reuse it, so the two sides of the comparison share a scale.
    states, mbs, results, games = load(args.data, 1.0)
    n_games = int(games.max())
    print(f"{states.shape[0]:,} positions from {n_games:,} games "
          f"({states.shape[0] / max(n_games, 1):.0f} per game)")

    # Hold a fifth of the data back. Tuning that only improves the training
    # error is overfitting, and the holdout is how you see it.
    holdout_from = int(n_games * 0.8)
    mask = games <= holdout_from
    tr = (states[mask], mbs[mask], results[mask])
    va = (states[~mask], mbs[~mask], results[~mask])
    print(f"train {tr[0].shape[0]:,} positions / {holdout_from:,} games   "
          f"holdout {va[0].shape[0]:,} / {n_games - holdout_from:,} games")

    theta = [100, 0] * len(CURVES) + [int(WEIGHTS[i]) for _n, i in SCALARS]
    w = build(theta)

    # K scales the evaluation into probability space; fit it first, holding the
    # weights fixed, or every later step is measured against the wrong curve.
    best_k, best = 1.0, 1e9
    for k in [0.10 + 0.05 * i for i in range(40)]:
        e = loss(tr[0], tr[1], tr[2], w, k)
        if e < best:
            best, best_k = e, k
    start_train = best
    start_val = loss(va[0], va[1], va[2], w, best_k)
    print(f"K = {best_k:.2f}   train {start_train:.6f}   holdout {start_val:.6f}")

    labels: list[str] = []
    for c in CURVES:
        labels += [f"{c[0]} scale", f"{c[0]} offset"]
    labels += [s2[0] for s2 in SCALARS]

    started = time.perf_counter()
    for sweep in range(args.sweeps):
        step = max(1, 16 >> sweep)
        improved = 0
        for idx in range(len(theta)):
            for delta in (step, -step):
                trial = list(theta)
                trial[idx] += delta
                if idx % 2 == 0 and idx < 2 * len(CURVES) and trial[idx] < 0:
                    continue
                e = loss(tr[0], tr[1], tr[2], build(trial), best_k)
                if e < best - 1e-12:
                    best, theta, improved = e, trial, improved + 1
                    break
        val = loss(va[0], va[1], va[2], build(theta), best_k)
        print(f"  sweep {sweep + 1}  step {step:>3d}  train {best:.6f}  "
              f"holdout {val:.6f}  {improved} moved  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        if improved == 0:
            break

    w = build(theta)
    final_val = loss(va[0], va[1], va[2], w, best_k)
    print(f"\nholdout error {start_val:.6f} -> {final_val:.6f} "
          f"({100 * (start_val - final_val) / start_val:+.2f}%)")
    print("\n--- tuned parameters ---")
    baseline = [100, 0] * len(CURVES) + [int(WEIGHTS[i]) for _n, i in SCALARS]
    for lbl, before, after in zip(labels, baseline, theta, strict=True):
        if before != after:
            print(f"  {lbl:24s} {before:>5d} -> {after:>5d}")
    args.out.write_text(",".join(str(int(x)) for x in w) + "\n")
    print(f"\nwrote {N_WEIGHTS} weights to {args.out}")


if __name__ == "__main__":
    main()
