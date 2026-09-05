"""A/B match driver for measuring an engine change.

Spawning a process per game costs ~20 seconds of JIT each time, which makes a
statistically meaningful match impossibly slow. This imports two copies of the
engine package into one process instead, so the compile is paid once and
hundreds of games cost only the games.

Both sides get the same opening positions and play each one with both colours,
so the comparison is not contaminated by opening luck.

    python -m tools.ab <dir_a> <dir_b> --games 200 --ms 60
"""
from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
import threading
import time
from pathlib import Path

import chess


def load(path: Path, alias: str):
    """Import an engine package from `path` under a unique module name.

    The submodule has to be imported explicitly: importing a package does not
    import its children.
    """
    sys.path.insert(0, str(path.parent))
    importlib.import_module(alias)
    return importlib.import_module(f"{alias}.engine")


def openings(count: int, seed: int = 17) -> list[str]:
    """Near-level positions a few moves in. Rated games start from curated
    openings rather than the initial position, so testing from the start would
    measure the wrong thing."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        b = chess.Board()
        for _ in range(rng.choice((4, 6, 8))):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.is_game_over():
            continue
        # keep it roughly level: no side more than a pawn up
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9}
        bal = sum(v * (len(b.pieces(p, chess.WHITE)) - len(b.pieces(p, chess.BLACK)))
                  for p, v in vals.items())
        if abs(bal) <= 1:
            out.append(b.fen())
    return out


def play_clock(white, black, wa, ba, fen: str, base: float, inc: float,
               ply_cap: int = 300) -> float:
    """One game under a real clock, each side running its own allocator.

    A fixed per-move budget cannot test a time-management change: the whole
    point of such a change is how the budget is chosen. Here each side is
    handed a clock and spends from it, and running out loses, exactly as the
    referee scores it.
    """
    board = chess.Board(fen)
    engines = {chess.WHITE: white, chess.BLACK: black}
    alloc = {chess.WHITE: wa, chess.BLACK: ba}
    clock = {chess.WHITE: base, chess.BLACK: base}
    while True:
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            return {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)
        if len(board.move_stack) >= ply_cap:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            bal = sum(v * (len(board.pieces(p, chess.WHITE))
                           - len(board.pieces(p, chess.BLACK)))
                      for p, v in vals.items())
            return 1.0 if bal > 0 else 0.0 if bal < 0 else 0.5

        side = board.turn
        try:                                     # newer allocators take the ply
            soft, hard = alloc[side](clock[side], inc, 200.0, len(board.move_stack))
        except TypeError:
            soft, hard = alloc[side](clock[side], inc, 200.0)
        t = time.perf_counter()
        ranked = engines[side].think(board.fen(), budget_ms=soft, hard_ms=hard,
                                     game_ply=len(board.move_stack))
        spent = (time.perf_counter() - t) * 1000.0
        clock[side] -= spent
        if clock[side] < 0:                      # flag: an immediate loss
            return 0.0 if side == chess.WHITE else 1.0
        clock[side] += inc
        if not ranked:
            return 0.0 if side == chess.WHITE else 1.0
        move = chess.Move.from_uci(ranked[0][0])
        if move not in board.legal_moves:
            return 0.0 if side == chess.WHITE else 1.0
        board.push(move)


class Ponder:
    """Minimal ponder driver for a match: think on the other side's time.
    Needs a second core to mean anything - on one core it only steals from
    the opponent, which flatters the pondering side."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.thread: threading.Thread | None = None
        self.event = threading.Event()

    def start(self, fen: str, game_ply: int) -> None:
        self.event.clear()
        self.thread = threading.Thread(
            target=self.engine.think, args=(fen, 600_000.0, 600_000.0),
            kwargs={"game_ply": game_ply, "stop": self.event}, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.event.set()
        while self.thread.is_alive():
            self.engine.abort()
            self.thread.join(0.005)
        self.thread = None


def play(white, black, fen: str, ms: float, ply_cap: int = 300,
         ms_white: float | None = None, ms_black: float | None = None,
         ponder=None) -> float:
    """One game. Returns White's score: 1.0, 0.5 or 0.0.

    The two sides may be given different budgets, which is how you price what
    extra thinking time is worth without simulating a whole clock. `ponder`
    names an engine that thinks on its opponent's time.
    """
    budget = {chess.WHITE: ms_white or ms, chess.BLACK: ms_black or ms}
    board = chess.Board(fen)
    engines = {chess.WHITE: white, chess.BLACK: black}
    pond = Ponder(ponder) if ponder is not None else None
    try:
        while True:
            if board.is_game_over(claim_draw=True):
                r = board.result(claim_draw=True)
                return {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)
            if len(board.move_stack) >= ply_cap:
                vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9}
                bal = sum(v * (len(board.pieces(p, chess.WHITE))
                               - len(board.pieces(p, chess.BLACK)))
                          for p, v in vals.items())
                return 1.0 if bal > 0 else 0.0 if bal < 0 else 0.5
            eng = engines[board.turn]
            bms = budget[board.turn]
            if pond is not None and eng is ponder:
                pond.stop()
            ranked = eng.think(board.fen(), budget_ms=bms, hard_ms=bms * 3,
                               game_ply=len(board.move_stack))
            if not ranked:
                return 0.0 if board.turn == chess.WHITE else 1.0
            try:
                move = chess.Move.from_uci(ranked[0][0])
            except ValueError:
                return 0.0 if board.turn == chess.WHITE else 1.0
            if move not in board.legal_moves:
                return 0.0 if board.turn == chess.WHITE else 1.0
            board.push(move)
            if pond is not None and eng is ponder:
                pond.start(board.fen(), len(board.move_stack))
    finally:
        if pond is not None:
            pond.stop()


def elo(score: float, n: int) -> tuple[float, float]:
    """Elo difference and a 95% interval, from a score rate over n games."""
    if score <= 0.0:
        return (-800.0, 0.0)
    if score >= 1.0:
        return (800.0, 0.0)
    diff = -400.0 * math.log10(1.0 / score - 1.0)
    se = math.sqrt(score * (1.0 - score) / n)
    lo = max(0.0001, score - 1.96 * se)
    hi = min(0.9999, score + 1.96 * se)
    margin = (-400.0 * math.log10(1.0 / hi - 1.0) - (-400.0 * math.log10(1.0 / lo - 1.0))) / 2
    return diff, margin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path, help="directory holding package `nbchess_a`")
    ap.add_argument("b", type=Path, help="directory holding package `nbchess_b`")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--ms", type=float, default=60.0)
    ap.add_argument("--ms-a", type=float, default=None,
                    help="give A a different budget, to price extra thinking time")
    ap.add_argument("--clock", action="store_true",
                    help="play with real clocks and each side's own allocator")
    ap.add_argument("--base-ms", type=float, default=120_000.0)
    ap.add_argument("--inc-ms", type=float, default=500.0)
    ap.add_argument("--ponder-a", action="store_true",
                    help="let A think on B's time (run on two cores)")
    ap.add_argument("--seed", type=int, default=17,
                    help="opening seed; change it for an independent sample")
    args = ap.parse_args()

    t = time.perf_counter()
    mod_a = load(args.a / "nbchess_a", "nbchess_a")
    mod_b = load(args.b / "nbchess_b", "nbchess_b")
    ea = mod_a.Engine()
    eb = mod_b.Engine()
    aa, ab_ = mod_a.allocate, mod_b.allocate
    print(f"both engines compiled in {time.perf_counter() - t:.0f}s", flush=True)

    fens = openings((args.games + 1) // 2, seed=args.seed)
    wins = draws = losses = 0
    started = time.perf_counter()
    for i in range(args.games):
        fen = fens[i // 2]
        a_is_white = i % 2 == 0
        w, bl = (ea, eb) if a_is_white else (eb, ea)
        wa, ba = (aa, ab_) if a_is_white else (ab_, aa)
        if args.clock:
            s = play_clock(w, bl, wa, ba, fen, args.base_ms, args.inc_ms)
        else:
            a_ms = args.ms_a if args.ms_a is not None else args.ms
            mw, mb = (a_ms, args.ms) if a_is_white else (args.ms, a_ms)
            s = play(w, bl, fen, args.ms, ms_white=mw, ms_black=mb,
                     ponder=ea if args.ponder_a else None)
        a_score = s if a_is_white else 1.0 - s
        if a_score == 1.0:
            wins += 1
        elif a_score == 0.5:
            draws += 1
        else:
            losses += 1
        if (i + 1) % 10 == 0:
            sc = (wins + draws / 2) / (i + 1)
            d, m = elo(sc, i + 1)
            print(f"  {i+1:>4d} games  +{wins} ={draws} -{losses}  "
                  f"{sc:6.1%}  {d:+7.0f} +/-{m:.0f} Elo  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    n = args.games
    sc = (wins + draws / 2) / n
    d, m = elo(sc, n)
    print(f"\nA vs B over {n} games at {args.ms:.0f}ms/move")
    print(f"+{wins} ={draws} -{losses}   score {sc:.1%}")
    print(f"Elo difference: {d:+.0f} +/- {m:.0f}  (95%)")


if __name__ == "__main__":
    main()
