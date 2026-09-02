"""Generate labelled positions for texel tuning.

Texel tuning fits evaluation weights so that the evaluation of a position
predicts the result of the game it came from. That needs positions paired with
outcomes, which self-play produces for free.

Only quiet positions are kept - not in check, and with a stable evaluation -
because a position in the middle of an exchange is scored by the search, not by
the evaluation, so fitting to it teaches the wrong thing.

    python -m tools.gendata --games 400 --ms 20 --out data.txt
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import chess

from nbchess.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--ms", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=Path("tuning.txt"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    engine = Engine()
    rng = random.Random(args.seed)
    started = time.perf_counter()
    written = 0
    # Written as we go: a long run that gets interrupted should still leave
    # usable data behind.
    out = args.out.open("w")

    for game in range(args.games):
        board = chess.Board()
        # A random opening keeps the sample from collapsing onto one line.
        for _ in range(rng.choice((4, 6, 8, 10))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue

        seen: list[str] = []
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 260:
            ranked = engine.think(board.fen(), budget_ms=args.ms, hard_ms=args.ms * 3,
                                  game_ply=len(board.move_stack))
            if not ranked:
                break
            move = chess.Move.from_uci(ranked[0][0])
            if move not in board.legal_moves:
                break
            # Keep the position only if it is quiet: the search's own move is
            # not a capture and the side to move is not in check.
            if not board.is_check() and not board.is_capture(move) and len(board.move_stack) > 8:
                seen.append(board.fen())
            board.push(move)

        result = board.result(claim_draw=True)
        score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
        for fen in seen:
            out.write(f"{fen}|{score}\n")
        written += len(seen)
        out.flush()

        if (game + 1) % 10 == 0:
            print(f"  {game+1:>4d} games  {written:>7,} positions  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    out.close()
    print(f"\nwrote {written:,} positions to {args.out}")


if __name__ == "__main__":
    main()
