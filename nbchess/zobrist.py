"""Zobrist keys.

Generated from a fixed seed so that a position always hashes to the same value
across processes - the two sides of a game run in separate containers, and a
transposition table entry written in one search must be readable by the next.
"""
import numpy as np
import numpy.typing as npt

_rng = np.random.default_rng(0xC4E55A17)


def _keys(*shape: int) -> npt.NDArray[np.uint64]:
    return _rng.integers(0, 1 << 64, size=shape, dtype=np.uint64)


#: PIECE_KEY[piece, square] - piece codes are WP..WK then BP..BK.
PIECE_KEY: npt.NDArray[np.uint64] = _keys(12, 64)
#: CASTLE_KEY[rights] - indexed by the whole 4-bit rights mask, so a change is
#: one XOR out and one XOR in rather than four conditional XORs.
CASTLE_KEY: npt.NDArray[np.uint64] = _keys(16)
#: EP_KEY[file] - only mixed in when an en-passant square actually exists.
EP_KEY: npt.NDArray[np.uint64] = _keys(8)
#: Mixed in whenever it is Black to move.
SIDE_KEY: np.uint64 = _keys(1)[0]
