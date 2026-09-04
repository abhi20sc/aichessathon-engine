"""Search driver: time allocation, iterative deepening, aspiration windows.

The clock is handled in two places. A hard deadline goes into the compiled
search and is polled every 2048 nodes, so a single long iteration cannot
overrun. A softer bound is checked between iterations and decides whether to
start another one at all - scaled by how stable the best move has been, since a
search that keeps changing its mind is worth more time.
"""
from __future__ import annotations

import time

import numpy as np

from .clock import available as clock_available
from .clock import new_timebuf, now_ns
from .fen import new_stack, set_fen
from .search import (
    C_CONTEMPT,
    C_DEADLINE,
    C_GAMEPLY,
    C_NODECAP,
    C_NODES,
    C_REPBASE,
    C_STOPPED,
    age_history,
    search_root,
)
from .tune import ASPIRATION_DELTA, CONTEMPT, INF, MATE_IN_MAX, MAX_PLY, TT_SIZE

_PROMO_CHARS = "nbrq"
MAX_ROOT_MOVES = 256
MAX_HISTORY = 512

#: Milliseconds assumed lost to transport per move. Replaced by a measurement
#: once the agent has seen the referee's clock disagree with its own.
DEFAULT_OVERHEAD_MS = 200.0
MOVES_TO_GO = 50

#: Fraction of the projected time pool to spend on one move.
#:
#: Raised from 0.024 after two rated games showed us finishing with 54-82s of a
#: 120s clock unspent. Measured: giving this engine 1.6x the thinking time is
#: worth +85 +/- 71 Elo over 100 games, and the 300-ply worst case still
#: survives with ~2s to spare. Real games run 65-86 plies, where the extra
#: spending is never clawed back by a draining clock.
SOFT_FRACTION = 0.038


def move_to_uci(mv: int) -> str:
    frm = mv & 63
    to = (mv >> 6) & 63
    promo = (mv >> 12) & 7
    return (
        chr(97 + frm % 8) + str(frm // 8 + 1)
        + chr(97 + to % 8) + str(to // 8 + 1)
        + ("" if promo == 0 else _PROMO_CHARS[promo - 1])
    )


def allocate(time_left_ms: float, increment_ms: float, overhead_ms: float) -> tuple[float, float]:
    """Return (soft, hard) budgets in milliseconds.

    Soft is the target; hard is the ceiling the search is never allowed past.
    Both are clamped so that we cannot spend more clock than we hold.
    """
    usable = max(1.0, time_left_ms - overhead_ms)
    pool = time_left_ms + increment_ms * (MOVES_TO_GO - 1) - overhead_ms * (2 + MOVES_TO_GO)
    soft = SOFT_FRACTION * max(pool, increment_ms)
    soft = max(soft, min(0.6 * increment_ms, 0.5 * usable))
    hard = min(5.0 * soft, 0.75 * usable)
    soft = min(soft, hard)
    return soft, hard


class Engine:
    """One engine per game. Compiles on construction, inside the init budget."""

    def __init__(self) -> None:
        self.stack, self.mbs, self.buf = new_stack(MAX_PLY)
        self.sbuf = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.scratch = np.zeros((2, 19), dtype=np.uint64)
        self.scratch_mb = np.full((2, 64), 12, dtype=np.int8)

        self.tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_move = np.zeros(TT_SIZE, dtype=np.uint32)
        self.tt_score = np.zeros(TT_SIZE, dtype=np.int16)
        self.tt_depth = np.zeros(TT_SIZE, dtype=np.int8)
        self.tt_bound = np.zeros(TT_SIZE, dtype=np.uint8)

        self.killers = np.zeros((MAX_PLY, 2), dtype=np.uint32)
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.rep = np.zeros(MAX_HISTORY + MAX_PLY, dtype=np.uint64)
        self.evals = np.zeros(MAX_PLY, dtype=np.int32)

        self.ctl = np.zeros(12, dtype=np.int64)
        self.tbuf = new_timebuf()
        self.out_moves = np.zeros(MAX_ROOT_MOVES, dtype=np.uint32)
        self.out_scores = np.zeros(MAX_ROOT_MOVES, dtype=np.int32)

        self.overhead_ms = DEFAULT_OVERHEAD_MS
        self.nps = 1_500_000.0
        self.depth_reached = 0
        self.nodes_last = 0
        self.last_search_ms = 0.0
        self.has_clock = clock_available()
        self._warmup()

    def _warmup(self) -> None:
        """Compile every kernel and take a first speed reading."""
        self.think("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                   budget_ms=250.0, hard_ms=400.0, history=())
        self.tt_key[:] = 0
        self.tt_move[:] = 0
        self.tt_depth[:] = 0
        self.tt_bound[:] = 0
        self.history[:] = 0
        self.killers[:] = 0

    def _load_history(self, history: tuple[str, ...]) -> int:
        """Hash the positions already seen this game so the search can detect a
        repetition against the real game, not only within its own tree."""
        n = 0
        for fen in history[-MAX_HISTORY:]:
            set_fen(self.scratch[0], self.scratch_mb[0], fen)
            self.rep[n] = self.scratch[0][18]
            n += 1
        return n

    def think(self, fen: str, budget_ms: float, hard_ms: float,
              history: tuple[str, ...] = (), game_ply: int = 0,
              max_depth: int = MAX_PLY - 8,
              ) -> list[tuple[str, int]]:
        """Search `fen` and return (uci, score) pairs, best first."""
        started = time.perf_counter()
        set_fen(self.stack[0], self.mbs[0], fen)
        age_history(self.history)
        self.killers[:] = 0

        base = self._load_history(history)
        self.ctl[C_NODES] = 0
        self.ctl[C_STOPPED] = 0
        self.ctl[C_REPBASE] = base
        self.ctl[C_CONTEMPT] = CONTEMPT
        self.ctl[C_GAMEPLY] = game_ply
        self.ctl[C_DEADLINE] = now_ns(self.tbuf) + int(hard_ms * 1_000_000)
        self.ctl[C_NODECAP] = max(4096, int(self.nps * hard_ms / 1000.0 * 2.0))

        ranked: list[tuple[str, int]] = []
        prev_best = ""
        stable = 0
        score = 0
        iter_ms = 0.0
        last_elapsed = 0.0

        for depth in range(1, max_depth + 1):
            alpha, beta = np.int32(-INF), np.int32(INF)
            delta = ASPIRATION_DELTA
            if depth >= 5:
                alpha = np.int32(max(-INF, score - delta))
                beta = np.int32(min(INF, score + delta))

            while True:
                count = search_root(
                    self.stack, self.mbs, self.buf, self.sbuf,
                    self.tt_key, self.tt_move, self.tt_score, self.tt_depth, self.tt_bound,
                    self.killers, self.history, self.rep, self.evals,
                    depth, alpha, beta, self.ctl, self.tbuf,
                    self.out_moves, self.out_scores)
                if count <= 0 or self.ctl[C_STOPPED] == 1:
                    break
                top = max(int(self.out_scores[i]) for i in range(count))
                if top <= int(alpha) and int(alpha) > -INF:
                    delta *= 2
                    alpha = np.int32(max(-INF, top - delta))
                    continue
                if top >= int(beta) and int(beta) < INF:
                    delta *= 2
                    beta = np.int32(min(INF, top + delta))
                    continue
                break

            if count > 0 and self.ctl[C_STOPPED] != 1:
                pairs = sorted(
                    ((move_to_uci(int(self.out_moves[i])), int(self.out_scores[i]))
                     for i in range(count)),
                    key=lambda p: p[1], reverse=True)
                ranked = pairs
                score = pairs[0][1]
                self.depth_reached = depth
                stable = stable + 1 if pairs[0][0] == prev_best else 0
                prev_best = pairs[0][0]
            elif not ranked and count != 0:
                # never come back empty-handed, even from an aborted iteration
                k = abs(count)
                ranked = sorted(
                    ((move_to_uci(int(self.out_moves[i])), int(self.out_scores[i]))
                     for i in range(k)),
                    key=lambda p: p[1], reverse=True)
                self.depth_reached = depth

            if self.ctl[C_STOPPED] == 1:
                break
            if abs(score) >= MATE_IN_MAX:      # proven mate, nothing left to find
                break
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            iter_ms = elapsed_ms - last_elapsed
            last_elapsed = elapsed_ms
            # a search that keeps changing its mind deserves more time
            soft = budget_ms * (1.20 - 0.04 * min(stable, 10))
            if elapsed_ms > soft:
                break
            # starting an iteration we cannot finish just burns clock: the next
            # one costs roughly twice the last, so predict before committing
            if depth >= 5 and elapsed_ms + iter_ms * 1.9 > soft:
                break

        spent = time.perf_counter() - started
        self.last_search_ms = spent * 1000.0
        self.nodes_last = int(self.ctl[C_NODES])
        if spent > 0.02 and self.nodes_last > 20_000:
            self.nps = 0.7 * self.nps + 0.3 * (self.nodes_last / spent)
        return ranked
