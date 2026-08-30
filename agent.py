"""AI Chessathon submission.

Move selection is a numba-compiled bitboard search (see `nbchess/`). This file
is the contract layer: budget the clock, hand the search the real game history
so it can see a repetition coming, and guarantee that a legal move reaches the
referee no matter what happens underneath.
"""

from __future__ import annotations

import os

# One core. Threads past the first cost time rather than winning it.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# The filesystem is read-only apart from /tmp, so the JIT cache must live there.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import time  # noqa: E402

import chess  # noqa: E402

#: The referee starts its stopwatch before the request reaches us and stops it
#: after our reply is parsed, so transport is charged to our clock. Measured
#: per move and smoothed, starting from a conservative guess.
INITIAL_OVERHEAD_MS = 200.0
INCREMENT_MS = 500.0
#: Centipawn score above which a draw is a bad outcome worth steering around.
WINNING_MARGIN_CP = 80

_engine = None
_engine_error: str | None = None

try:
    from nbchess.engine import Engine, allocate

    _engine = Engine()          # compiles here, inside the 60 s init budget
except Exception as exc:
    _engine_error = f"{type(exc).__name__}: {exc}"


class GameTracker:
    """Reconstructs the real game from the positions we are shown.

    We are only handed a FEN, but the referee claims threefold and fifty-move
    draws automatically against the full move stack. Replaying the game locally
    - inferring the opponent's reply by matching it against the FEN we were just
    given - is what lets the search score a repetition correctly instead of
    walking into one.
    """

    def __init__(self) -> None:
        self.board: chess.Board | None = None

    @staticmethod
    def _key(board: chess.Board) -> tuple[str, bool, int, int | None]:
        return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)

    def sync(self, fen: str) -> chess.Board:
        target = self._key(chess.Board(fen))
        if self.board is not None:
            if self._key(self.board) == target:
                return self.board
            for move in self.board.legal_moves:      # the opponent's reply
                self.board.push(move)
                if self._key(self.board) == target:
                    return self.board
                self.board.pop()
        self.board = chess.Board(fen)                # first move, or lost the thread
        return self.board

    def commit(self, uci: str) -> None:
        if self.board is not None:
            try:
                self.board.push_uci(uci)
            except ValueError:
                self.board = None

    @staticmethod
    def history_fens(board: chess.Board) -> tuple[str, ...]:
        """Positions seen since the last capture or pawn move, oldest first.
        Anything older cannot repeat, so it does not need hashing."""
        depth = min(board.halfmove_clock, len(board.move_stack))
        if depth == 0:
            return ()
        undone = [board.pop() for _ in range(depth)]
        fens = []
        for _ in range(depth):
            fens.append(board.fen())
            board.push(undone.pop())
        return tuple(fens)


_tracker = GameTracker()
_overhead_ms = INITIAL_OVERHEAD_MS


def _choose(board: chess.Board, ranked: list[tuple[str, int]]) -> str | None:
    """Best ranked move, skipping ones that hand the referee a draw claim.

    Only applies when we are winning: from a worse position a repetition is a
    good result, not a mistake.
    """
    if not ranked:
        return None
    if ranked[0][1] < WINNING_MARGIN_CP:
        return ranked[0][0]
    for uci, score in ranked:
        if score < WINNING_MARGIN_CP:
            break
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            board.push(move)
            claimable = board.is_repetition(3) or board.is_fifty_moves()
            board.pop()
        except ValueError:
            continue
        if not claimable:
            return uci
    return ranked[0][0]


def _fallback(board: chess.Board) -> str:
    """Material-greedy one-ply search. Used only if the compiled engine is gone."""
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    best, best_score = None, -10**9
    for move in board.legal_moves:
        score = 0
        captured = board.piece_at(move.to_square)
        if captured is not None:
            score += values[captured.piece_type]
        if move.promotion:
            score += values[move.promotion]
        board.push(move)
        if board.is_checkmate():
            score += 10**6
        board.pop()
        if score > best_score:
            best, best_score = move, score
    return (best or next(iter(board.legal_moves))).uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation. This function must never raise."""
    entered = time.perf_counter()
    global _overhead_ms

    board = chess.Board(fen)
    legal = {move.uci() for move in board.legal_moves}
    if len(legal) == 1:                       # nothing to think about
        only = next(iter(legal))
        _tracker.commit(only)
        return only

    chosen: str | None = None
    if _engine is not None:
        try:
            tracked = _tracker.sync(fen)
            history = GameTracker.history_fens(tracked)
            soft, hard = allocate(float(time_left_ms), INCREMENT_MS, _overhead_ms)
            ranked = _engine.think(fen, budget_ms=soft, hard_ms=hard, history=history)
            candidate = _choose(tracked, ranked)
            if candidate in legal:
                chosen = candidate
        except Exception:
            chosen = None

    if chosen is None:
        try:
            candidate = _fallback(board)
            if candidate in legal:
                chosen = candidate
        except Exception:
            chosen = None

    if chosen is None:
        chosen = next(iter(legal))
    _tracker.commit(chosen)

    # Learn how much wall time we lose outside the search itself.
    spent_ms = (time.perf_counter() - entered) * 1000.0
    if _engine is not None:
        slack = spent_ms - getattr(_engine, "last_search_ms", spent_ms)
        if 0.0 <= slack < 2000.0:
            _overhead_ms = 0.8 * _overhead_ms + 0.2 * (slack + 100.0)
    return chosen
