"""Masks and weights for the positional evaluation terms.

Ordered by the Elo each is worth, measured by engines that published SPRT
results for them: mobility and passed pawns first, king safety next, bishop
pair last and barely significant. Everything here is a starting point to be
tuned, not a tuned result.
"""
import numpy as np
import numpy.typing as npt

FILE_BB = np.array([0x0101010101010101 << f for f in range(8)], dtype=np.uint64)
RANK_BB = np.array([0xFF << (8 * r) for r in range(8)], dtype=np.uint64)

#: Files either side of a given file - a pawn with none of these and none of
#: its own file occupied by an enemy pawn ahead of it is passed.
ADJACENT_FILES = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    m = np.uint64(0)
    if _f > 0:
        m |= FILE_BB[_f - 1]
    if _f < 7:
        m |= FILE_BB[_f + 1]
    ADJACENT_FILES[_f] = m

#: PASSED_MASK[colour][square]: every square from which an enemy pawn could stop
#: or capture this pawn on its way to promotion.
PASSED_MASK = np.zeros((2, 64), dtype=np.uint64)
#: Squares in front of a pawn on its own file, used for doubled-pawn detection.
FRONT_MASK = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    f, r = _sq % 8, _sq // 8
    span = np.uint64(0)
    for rr in range(r + 1, 8):
        span |= RANK_BB[rr]
    PASSED_MASK[0, _sq] = span & (FILE_BB[f] | ADJACENT_FILES[f])
    FRONT_MASK[0, _sq] = span & FILE_BB[f]
    span = np.uint64(0)
    for rr in range(0, r):
        span |= RANK_BB[rr]
    PASSED_MASK[1, _sq] = span & (FILE_BB[f] | ADJACENT_FILES[f])
    FRONT_MASK[1, _sq] = span & FILE_BB[f]

#: KING_ZONE[colour][square]: the king's own square, its neighbours, and the
#: rank in front of it, which is where an attack actually lands.
KING_ZONE = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    f, r = _sq % 8, _sq // 8
    for colour, direction in ((0, 1), (1, -1)):
        z = np.uint64(0)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1, direction * 2):
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    z |= np.uint64(1) << np.uint64(nr * 8 + nf)
        KING_ZONE[colour, _sq] = z


def _ramp(n: int, lo: int, hi: int) -> list[int]:
    """A concave bonus curve: the first few squares of freedom are worth far
    more than the last few, which is how mobility actually behaves."""
    return [round(lo + (hi - lo) * (i / (n - 1)) ** 0.6) for i in range(n)]


# Mobility, indexed by the number of safe destination squares.
KNIGHT_MOB_MG = np.array(_ramp(9, -32, 22), dtype=np.int32)
KNIGHT_MOB_EG = np.array(_ramp(9, -30, 20), dtype=np.int32)
BISHOP_MOB_MG = np.array(_ramp(14, -28, 30), dtype=np.int32)
BISHOP_MOB_EG = np.array(_ramp(14, -30, 32), dtype=np.int32)
ROOK_MOB_MG = np.array(_ramp(15, -22, 20), dtype=np.int32)
ROOK_MOB_EG = np.array(_ramp(15, -34, 40), dtype=np.int32)
QUEEN_MOB_MG = np.array(_ramp(28, -18, 16), dtype=np.int32)
QUEEN_MOB_EG = np.array(_ramp(28, -26, 30), dtype=np.int32)

#: Passed pawn bonus by how far the pawn has advanced (its own relative rank).
PASSED_MG = np.array([0, 4, 8, 18, 34, 60, 96, 0], dtype=np.int32)
PASSED_EG = np.array([0, 10, 20, 40, 72, 120, 180, 0], dtype=np.int32)

ISOLATED_MG, ISOLATED_EG = np.int32(-12), np.int32(-16)
DOUBLED_MG, DOUBLED_EG = np.int32(-8), np.int32(-20)
BISHOP_PAIR_MG, BISHOP_PAIR_EG = np.int32(24), np.int32(42)

#: Attack units contributed per attacked king-zone square, by piece type.
KS_WEIGHT = np.array([0, 2, 2, 3, 5, 0], dtype=np.int32)
#: Scaling by how many distinct pieces join the attack. One attacker is not an
#: attack; the classical advice is to ignore fewer than two.
KS_ATTACKERS = np.array([0, 0, 50, 75, 88, 94, 97, 99, 99], dtype=np.int32)
#: Nonlinear penalty ramp indexed by attack units, capped.
KS_TABLE: npt.NDArray[np.int32] = np.array(
    [min(500, (i * i) // 4) for i in range(100)], dtype=np.int32)


# ---------------------------------------------------------------------------
# Packed weight vector.
#
# The evaluation reads its tunable weights from one flat array rather than from
# a dozen module globals. Numba freezes globals at compile time, so a tuner that
# wanted to try a different value would have to recompile for every candidate;
# passing the array as an argument makes tuning a matter of swapping numbers.
# The shipped evaluation passes WEIGHTS and behaves exactly as before.
# ---------------------------------------------------------------------------

I_KN_MG, N_KN = 0, 9
I_KN_EG = I_KN_MG + N_KN
I_BI_MG, N_BI = I_KN_EG + N_KN, 14
I_BI_EG = I_BI_MG + N_BI
I_RK_MG, N_RK = I_BI_EG + N_BI, 15
I_RK_EG = I_RK_MG + N_RK
I_QU_MG, N_QU = I_RK_EG + N_RK, 28
I_QU_EG = I_QU_MG + N_QU
I_PA_MG, N_PA = I_QU_EG + N_QU, 8
I_PA_EG = I_PA_MG + N_PA
I_ISO_MG = I_PA_EG + N_PA
I_ISO_EG = I_ISO_MG + 1
I_DBL_MG = I_ISO_EG + 1
I_DBL_EG = I_DBL_MG + 1
I_BP_MG = I_DBL_EG + 1
I_BP_EG = I_BP_MG + 1
#: Percentage scale applied to the king-safety penalty; 100 leaves it unchanged.
I_KS = I_BP_EG + 1
N_WEIGHTS = I_KS + 1


def pack() -> npt.NDArray[np.int32]:
    w = np.zeros(N_WEIGHTS, dtype=np.int32)
    w[I_KN_MG:I_KN_MG + N_KN] = KNIGHT_MOB_MG
    w[I_KN_EG:I_KN_EG + N_KN] = KNIGHT_MOB_EG
    w[I_BI_MG:I_BI_MG + N_BI] = BISHOP_MOB_MG
    w[I_BI_EG:I_BI_EG + N_BI] = BISHOP_MOB_EG
    w[I_RK_MG:I_RK_MG + N_RK] = ROOK_MOB_MG
    w[I_RK_EG:I_RK_EG + N_RK] = ROOK_MOB_EG
    w[I_QU_MG:I_QU_MG + N_QU] = QUEEN_MOB_MG
    w[I_QU_EG:I_QU_EG + N_QU] = QUEEN_MOB_EG
    w[I_PA_MG:I_PA_MG + N_PA] = PASSED_MG
    w[I_PA_EG:I_PA_EG + N_PA] = PASSED_EG
    w[I_ISO_MG] = ISOLATED_MG
    w[I_ISO_EG] = ISOLATED_EG
    w[I_DBL_MG] = DOUBLED_MG
    w[I_DBL_EG] = DOUBLED_EG
    w[I_BP_MG] = BISHOP_PAIR_MG
    w[I_BP_EG] = BISHOP_PAIR_EG
    w[I_KS] = 100
    return w


WEIGHTS: npt.NDArray[np.int32] = pack()
