"""Train a small NNUE-style evaluation on labelled positions.

Rows are `fen|result|cp`, as written by tools.pipeline. The network is the
simplest one that works: 768 piece-square inputs per perspective (own pieces
first, then the opponent's, board flipped for Black), one shared hidden layer
with clipped ReLU, and a linear output over the side-to-move and other-side
accumulators. Output is in centipawns; the loss squashes it through the same
sigmoid the Texel tuner uses so that the two evaluations share a scale.

    python -m tools.nnue_train --data a.txt b.txt --hidden 128 --epochs 10 \
        --out nbchess/nnue.npz
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

PAD = 768          # padding index: a fixed zero row
MAXP = 32          # pieces on the board, at most
K = 0.9            # Texel K; cp -> probability
SCALE = K * math.log(10.0) / 400.0
LAMBDA = 1.0       # weight on the engine label versus the game result
PIECE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}


#: Output buckets by piece count: the last layer has one row per bucket, so
#: an ending is scored by weights fitted only on endings. Free at inference.
BUCKET_EDGES = (12, 22)        # pieces <= 12 -> 0, <= 22 -> 1, else 2
N_BUCKETS = len(BUCKET_EDGES) + 1


def bucket_of(pieces: int) -> int:
    for k, edge in enumerate(BUCKET_EDGES):
        if pieces <= edge:
            return k
    return N_BUCKETS - 1


def featurise(fen: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Indices for the white and black perspectives, and side to move."""
    rows, stm = fen.split()[0], fen.split()[1]
    w = np.full(MAXP, PAD, dtype=np.int32)
    b = np.full(MAXP, PAD, dtype=np.int32)
    n = 0
    rank = 7
    for row in rows.split("/"):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
                continue
            pc = PIECE[ch]
            colour, ptype = pc // 6, pc % 6
            sq = rank * 8 + file
            w[n] = colour * 384 + ptype * 64 + sq
            b[n] = (1 - colour) * 384 + ptype * 64 + (sq ^ 56)
            n += 1
            file += 1
        rank -= 1
    return w, b, 0 if stm == "w" else 1


def load(paths: list[Path], limit: int | None, residual: bool, mirror: bool = False,
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Positions, targets and - for a residual net - the hand evaluation the
    network is trained to correct, all from the side to move's view."""
    if residual:
        from nbchess.fen import new_stack, set_fen
        from nbchess.search import evaluate_w
        from nbchess.terms import WEIGHTS
        st, mbs, _ = new_stack()
    # Two passes: count, then fill preallocated arrays. Building a Python
    # list of five million small arrays first needs more memory than the
    # arrays themselves, and that is what ran out at 5.3M rows.
    total = 0
    for p in paths:
        with p.open() as fh:
            for line in fh:
                if line.count("|") >= 2:
                    total += 1
    if limit:
        total = min(total, limit)
    W2 = np.full((total, MAXP), PAD, dtype=np.int32)
    B2 = np.full((total, MAXP), PAD, dtype=np.int32)
    S2 = np.zeros(total, dtype=np.int64)
    T2 = np.zeros(total, dtype=np.float32)
    E2 = np.zeros(total, dtype=np.float32)
    K2 = np.zeros(total, dtype=np.int64)
    i = 0
    for p in paths:
        with p.open() as fh:
            for line in fh:
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                if i >= total:
                    break
                fen, result, cp = parts[0], float(parts[1]), float(parts[2])
                w, b, stm = featurise(fen)
                W2[i] = w
                B2[i] = b
                S2[i] = stm
                K2[i] = bucket_of(int((w < PAD).sum()))
                p_cp = 1.0 / (1.0 + math.exp(-cp * SCALE))
                t = LAMBDA * p_cp + (1.0 - LAMBDA) * result
                T2[i] = t if stm == 0 else 1.0 - t
                if residual:
                    set_fen(st[0], mbs[0], fen)
                    E2[i] = float(evaluate_w(st[0], mbs[0], WEIGHTS))
                i += 1
        if i >= total:
            break
    W2, B2, S2, T2, E2, K2 = W2[:i], B2[:i], S2[:i], T2[:i], E2[:i], K2[:i]
    n_hold = len(T2) // 20
    if mirror:
        # Mirror every position left-right: same label, same side to move,
        # every square sq -> sq ^ 7 (castling rights aside, chess is symmetric).
        # Doubles the data for free; hold-out rows are not mirrored.
        n = len(T2)
        cut = n - n // 20
        def flip(a: np.ndarray) -> np.ndarray:
            f = a[:cut].copy()
            real = f < PAD
            f[real] = (f[real] & ~7) | (7 - (f[real] & 7))
            return f
        W2 = np.concatenate([flip(W2), W2]); B2 = np.concatenate([flip(B2), B2])
        S2 = np.concatenate([S2[:cut], S2]); T2 = np.concatenate([T2[:cut], T2])
        E2 = np.concatenate([E2[:cut], E2]); K2 = np.concatenate([K2[:cut], K2])
    return (W2, B2, S2, T2, E2, K2, n_hold)


class Net(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(769, hidden, mode="sum", padding_idx=PAD)
        self.ft_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(2 * hidden, N_BUCKETS)
        torch.nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    def forward(self, w: torch.Tensor, b: torch.Tensor, stm: torch.Tensor,
                k: torch.Tensor) -> torch.Tensor:
        aw = torch.clamp(self.ft(w) + self.ft_bias, 0.0, 1.0)
        ab = torch.clamp(self.ft(b) + self.ft_bias, 0.0, 1.0)
        us = torch.where(stm[:, None] == 0, aw, ab)
        them = torch.where(stm[:, None] == 0, ab, aw)
        # the linear layer works in probability units; centipawns are 1/SCALE
        # times larger, so the trainable weights stay O(1)
        heads = self.out(torch.cat([us, them], dim=1))          # one column per bucket
        return torch.gather(heads, 1, k[:, None]).squeeze(1) / SCALE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("nbchess/nnue.npz"))
    ap.add_argument("--residual", action="store_true",
                    help="train the net to correct the hand evaluation rather than replace it")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mirror", action="store_true",
                    help="add the left-right mirror of every training position")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    t0 = time.perf_counter()
    W, B, S, T, E, K, n_hold = load(args.data, args.limit, args.residual, args.mirror)
    n = len(T)
    print(f"{n:,} positions loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    # Hold out the LAST 5% as a contiguous block. The pipeline writes many
    # positions per game, so a random split would put siblings of every
    # holdout position in the training set and the holdout would flatter
    # the net (and pick an overfitted epoch). Contiguous rows are whole games.
    hold = np.arange(n - n_hold, n)
    train = np.arange(0, n - n_hold)
    W, B, S, T, E, K = (torch.from_numpy(x) for x in (W, B, S, T, E, K))

    net = Net(args.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    def loss_on(ix: np.ndarray) -> float:
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(ix), 65536):
                j = torch.from_numpy(ix[i:i + 65536])
                p = torch.sigmoid((net(W[j].long(), B[j].long(), S[j], K[j]) + E[j]) * SCALE)
                tot += float(((p - T[j]) ** 2).sum())
            return tot / len(ix)

    print(f"initial   train {loss_on(train[:200000]):.6f}  holdout {loss_on(hold):.6f}", flush=True)
    best_val, best_state = 1e9, None
    for ep in range(args.epochs):
        t1 = time.perf_counter()
        perm = np.random.default_rng(ep).permutation(train)
        for i in range(0, len(perm), args.batch):
            j = torch.from_numpy(perm[i:i + args.batch])
            p = torch.sigmoid((net(W[j].long(), B[j].long(), S[j], K[j]) + E[j]) * SCALE)
            loss = ((p - T[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                net.ft.weight[PAD].zero_()
        sched.step()
        val = loss_on(hold)
        if val < best_val:                       # keep the epoch the holdout liked best
            best_val = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep + 1:2d}  train {loss_on(train[:200000]):.6f}  "
              f"holdout {val:.6f}{'  *' if val == best_val else ''}  ({time.perf_counter() - t1:.0f}s)", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"restored best epoch: holdout {best_val:.6f}")

    w1 = net.ft.weight.detach().numpy().astype(np.float32)           # 769 x H
    b1 = net.ft_bias.detach().numpy().astype(np.float32)
    w2 = net.out.weight.detach().numpy().astype(np.float32) / SCALE      # buckets x 2H, in cp
    b2 = (net.out.bias.detach().numpy() / SCALE).astype(np.float32)      # buckets
    np.savez(args.out, w1=w1, b1=b1, w2=w2, b2=b2, residual=np.int32(1 if args.residual else 0))
    print(f"wrote {args.out}  ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
