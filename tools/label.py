"""Relabel tuning positions with a reference engine's evaluation.

Game results are a terrible label for tuning at our scale: every position in a
game carries the same number, so 200 games give 200 independent labels no
matter how many positions we cut them into. That is what made the first tuning
attempt overfit.

An engine's evaluation of each position is a distinct label per position, which
turns 200 independent observations into ~18,000. The competition rules permit
this explicitly - "training data: unrestricted, including positions annotated
by an existing engine. The ban covers only what ships inside the submission."
Nothing from here ends up in the zip.

    python -m tools.label --in tuning.txt --out labelled.txt --nodes 25000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import chess.engine

STOCKFISH = Path("/usr/games/stockfish")
#: Evaluations beyond this are clipped: a position that is winning by 10 pawns
#: and one winning by 30 teach the same lesson, and the difference would
#: dominate the fit.
CLIP_CP = 1500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, default=Path("tuning.txt"))
    ap.add_argument("--out", type=Path, default=Path("labelled.txt"))
    ap.add_argument("--nodes", type=int, default=25000)
    args = ap.parse_args()

    rows = [ln for ln in args.src.read_text().splitlines() if "|" in ln]
    print(f"{len(rows):,} positions to label at {args.nodes:,} nodes each")

    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    engine.configure({"Threads": 1, "Hash": 128})
    limit = chess.engine.Limit(nodes=args.nodes)
    out = args.out.open("w")
    started = time.perf_counter()
    done = skipped = 0

    try:
        for i, line in enumerate(rows):
            fen, result = line.rsplit("|", 1)
            board = chess.Board(fen)
            try:
                info = engine.analyse(board, limit)
            except chess.engine.EngineError:
                skipped += 1
                continue
            score = info["score"].white()
            cp = score.score(mate_score=CLIP_CP * 2)
            if cp is None:
                skipped += 1
                continue
            cp = max(-CLIP_CP, min(CLIP_CP, cp))
            # fen | game result | engine score, both from White's point of view
            out.write(f"{fen}|{result}|{cp}\n")
            done += 1
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {i+1:>7,}/{len(rows):,}  {rate:.0f}/s  "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
                out.flush()
    finally:
        engine.quit()
        out.close()
    print(f"\nlabelled {done:,}, skipped {skipped}, -> {args.out}")


if __name__ == "__main__":
    main()
