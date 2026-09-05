"""Continuous training-data pipeline: play, sample, label, append.

Runs in the background and appends `fen|result|engine_cp` rows to a file,
flushing after every game, so it can be stopped and resumed at any time and
never loses more than one game's work.

Games are played by the reference engine against itself at a shallow node
limit, from randomised openings, so the sample is large and diverse and comes
from stronger play than our own engine's - which is the point: we want to learn
positions we do not yet understand, not ones we already play well. Positions
are sampled sparsely within each game (the labels within a game are
correlated; more positions from one game are worth little) and labelled with a
deeper search.

The competition rules permit this explicitly: training data is unrestricted,
including positions annotated by an existing engine. Nothing here ships.

    python -m tools.pipeline --out big.txt --games 100000
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import chess
import chess.engine

#: Reference engine for playing and labelling; override to run the pipeline
#: on another machine (it is never part of the submission).
STOCKFISH = os.environ.get("STOCKFISH", "/usr/games/stockfish")
CLIP_CP = 1500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=100_000)
    ap.add_argument("--play-nodes", type=int, default=6000)
    ap.add_argument("--label-nodes", type=int, default=15000)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--late-weight", type=float, default=0.0,
                    help="3.0 makes the last position four times as likely as the first")
    ap.add_argument("--start-fens", type=Path, default=None,
                    help="file of FENs to start games from (the rated openings) instead of "
                         "the initial position; a few random moves are still played first")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    starts = ([line.strip() for line in args.start_fens.read_text().splitlines() if line.strip()]
              if args.start_fens else [])
    player = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    player.configure({"Threads": 1, "Hash": 64})
    labeller = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    labeller.configure({"Threads": 1, "Hash": 64})
    play_lim = chess.engine.Limit(nodes=args.play_nodes)
    label_lim = chess.engine.Limit(nodes=args.label_nodes)

    out = args.out.open("a")
    started = time.perf_counter()
    games = rows = 0
    try:
        for _ in range(args.games):
            board = chess.Board(rng.choice(starts)) if starts else chess.Board()
            for _ in range(rng.randint(1, 4) if starts else rng.randint(4, 12)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if board.is_game_over():
                continue

            candidates: list[str] = []
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 240:
                res = player.play(board, play_lim)
                mv = res.move
                if mv is None:
                    break
                if (not board.is_check() and not board.is_capture(mv)
                        and (len(board.move_stack) > 10 or starts)):
                    candidates.append(board.fen())
                board.push(mv)
            result = {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)

            # sparse sample: correlated labels within a game are worth little.
            # Weighted toward the late game: endgames are where the hand
            # evaluation is weakest, and a uniform sample under-represents them.
            if len(candidates) > args.per_game:
                weights = [1.0 + args.late_weight * i / len(candidates)
                           for i in range(len(candidates))]
                picked: set[int] = set()
                while len(picked) < args.per_game:
                    picked.add(rng.choices(range(len(candidates)), weights)[0])
                candidates = [candidates[i] for i in sorted(picked)]
            for fen in candidates:
                info = labeller.analyse(chess.Board(fen), label_lim)
                cp = info["score"].white().score(mate_score=CLIP_CP * 2)
                if cp is None:
                    continue
                cp = max(-CLIP_CP, min(CLIP_CP, cp))
                out.write(f"{fen}|{result}|{cp}\n")
                rows += 1
            out.flush()
            games += 1
            if games % 25 == 0:
                el = time.perf_counter() - started
                print(f"{games:>6d} games  {rows:>7d} rows  "
                      f"{games/el*3600:.0f} games/h  ({el:.0f}s)", flush=True)
    finally:
        out.close()
        player.quit()
        labeller.quit()


if __name__ == "__main__":
    main()
