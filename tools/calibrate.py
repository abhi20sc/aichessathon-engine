"""Measure our engine against a strength-limited reference engine.

Every result so far has been against our own baselines, which tells us we are
improving but not where we stand. Stockfish's UCI_LimitStrength gives an
external, approximately-calibrated scale.

This is a TESTING tool. Nothing it touches ships: the competition rules
prohibit third-party engines inside the submission but place no restriction on
what we test or train against.

Two caveats on the number it produces. UCI_Elo is Stockfish's own calibration
and is approximate, and a strength-limited engine plays differently from a
genuine engine of that rating - it makes occasional large errors rather than
consistently weaker moves. Treat the output as a bracket, not a rating.

    python -m tools.calibrate --elo 2000 --games 30 --ms 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import chess.engine

from nbchess.engine import Engine
from tools.ab import elo as elo_from_score
from tools.ab import openings

STOCKFISH = Path("/usr/games/stockfish")


def play(ours: Engine, sf: chess.engine.SimpleEngine, fen: str,
         ms: float, ours_white: bool) -> float:
    """One game. Returns our score: 1.0, 0.5 or 0.0."""
    board = chess.Board(fen)
    limit = chess.engine.Limit(time=ms / 1000.0)
    while True:
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            if r == "1/2-1/2":
                return 0.5
            won_white = r == "1-0"
            return 1.0 if won_white == ours_white else 0.0
        if len(board.move_stack) >= 300:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            bal = sum(v * (len(board.pieces(p, chess.WHITE))
                           - len(board.pieces(p, chess.BLACK)))
                      for p, v in vals.items())
            if bal == 0:
                return 0.5
            return 1.0 if (bal > 0) == ours_white else 0.0

        our_turn = (board.turn == chess.WHITE) == ours_white
        if our_turn:
            ranked = ours.think(board.fen(), budget_ms=ms, hard_ms=ms * 3,
                                game_ply=len(board.move_stack))
            if not ranked:
                return 0.0
            move = chess.Move.from_uci(ranked[0][0])
            if move not in board.legal_moves:
                return 0.0
        else:
            move = sf.play(board, limit).move
            if move is None:
                return 1.0
        board.push(move)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elo", type=int, default=2000, help="reference engine strength")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()

    ours = Engine()
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1, "Hash": 64})

    fens = openings((args.games + 1) // 2, seed=args.seed)
    wins = draws = losses = 0
    started = time.perf_counter()
    try:
        for i in range(args.games):
            s = play(ours, sf, fens[i // 2], args.ms, ours_white=(i % 2 == 0))
            if s == 1.0:
                wins += 1
            elif s == 0.5:
                draws += 1
            else:
                losses += 1
            if (i + 1) % 5 == 0:
                sc = (wins + draws / 2) / (i + 1)
                d, m = elo_from_score(sc, i + 1)
                print(f"  {i+1:>3d} games  +{wins} ={draws} -{losses}  {sc:6.1%}  "
                      f"implied {args.elo + d:.0f} +/-{m:.0f}  "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        sf.quit()

    n = args.games
    sc = (wins + draws / 2) / n
    d, m = elo_from_score(sc, n)
    print(f"\nvs Stockfish 16 limited to Elo {args.elo}, {args.ms:.0f}ms/move, {n} games")
    print(f"+{wins} ={draws} -{losses}   score {sc:.1%}")
    print(f"implied strength: {args.elo + d:.0f} +/- {m:.0f}")
    print("(interval is 95% on the score; the reference scale itself is approximate)")



if __name__ == "__main__":
    main()
