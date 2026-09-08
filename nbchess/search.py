"""Negamax search.

Structure, in the order the pieces matter:

  transposition table   caches results and, more importantly, supplies the
                        TT move for ordering - which is where most of its
                        strength actually lives
  move ordering         TT move, then captures by MVV-LVA, then killers,
                        then quiet moves by history score
  PVS                   full window on the first move, null window after
  null-move pruning     give the opponent a free move; if we are still winning
                        the position is good enough to prune
  LMR                   search late, quiet moves shallower and re-search only
                        if they beat alpha
  RFP / LMP             give up early when a shallow node is far above beta,
                        or when we have already tried enough quiet moves
  repetition            twofold inside the search is scored as a draw, which is
                        what stops a winning engine from shuffling

The clock is a real deadline polled every 2048 nodes, not a node estimate.
"""
import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core.types import (
    Array,
    boolean,
    float32,
    int8,
    int32,
    int64,
    uint8,
    uint32,
    uint64,
)
from numba.experimental import jitclass  # type: ignore[attr-defined]

from .clock import now_ns
from .core import (
    attacked,
    bishop_att,
    gen_captures,
    gen_moves,
    lsb,
    make,
    make_null,
    popcount,
    queen_att,
    rook_att,
)
from .evaltables import MATERIAL
from .nnue import RESIDUAL as NN_RESIDUAL
from .nnue import USE_NNUE, acc_copy, acc_update, bucket_of, nn_eval, nn_output
from .pesto import PHASE_MAX
from .tables import KING_ATT, KNIGHT_ATT, PAWN_ATT
from .terms import (
    ADJACENT_FILES,
    FILE_BB,
    FRONT_MASK,
    I_BI_EG,
    I_BI_MG,
    I_BP_EG,
    I_BP_MG,
    I_DBL_EG,
    I_DBL_MG,
    I_ISO_EG,
    I_ISO_MG,
    I_KN_EG,
    I_KN_MG,
    I_KS,
    I_MAT_EG,
    I_MAT_MG,
    I_PA_EG,
    I_PA_MG,
    I_PK_ENEMY,
    I_PK_OWN,
    I_QU_EG,
    I_QU_MG,
    I_RK_EG,
    I_RK_MG,
    I_ROPEN_EG,
    I_ROPEN_MG,
    I_RSEMI_EG,
    I_RSEMI_MG,
    I_SHELTER,
    I_STORM,
    I_THREAT_EG,
    I_THREAT_MG,
    KING_ZONE,
    KS_ATTACKERS,
    KS_TABLE,
    KS_WEIGHT,
    PASSED_MASK,
    WEIGHTS,
)
from .tune import (
    ADJUDICATION_BLEND,
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    FIFTY_MOVE_SCALE_FROM,
    FUTILITY_DEPTH,
    FUTILITY_IMPROVING,
    FUTILITY_MARGIN,
    HISTORY_MAX,
    IIR_MIN_DEPTH,
    INF,
    LMP,
    LMP_MAX_DEPTH,
    LMR,
    MATE,
    MATE_IN_MAX,
    NMP_MIN_DEPTH,
    PLY_CAP,
    REF_VALUES,
    RFP_MARGIN,
    RFP_MAX_DEPTH,
    SE_MARGIN,
    SE_MIN_DEPTH,
    SEE_PRUNE_DEPTH,
    SEE_PRUNE_MARGIN,
    TT_MASK,
)

#: A global numpy array reaches a jitted function as a readonly array. Only that
#: one signature is compiled: the evaluation is the largest kernel here, and a
#: second signature would compile the whole of it again for no benefit. Callers
#: with a writable array (the tuner) mark it readonly first.
_W_RO = Array(int32, 1, "C", readonly=True)  # type: ignore[no-untyped-call]

# ctl slots, so the search can report back without returning tuples
C_NODES, C_DEADLINE, C_STOPPED, C_REPBASE, C_NODECAP, C_CONTEMPT = 0, 1, 2, 3, 4, 5
C_GAMEPLY = 6


FILE_A = uint64(0x0101010101010101)
FILE_H = uint64(0x8080808080808080)


@njit(int32(uint64[:], int32), cache=False, nogil=True)
def endgame_adjust(s: npt.NDArray[np.uint64], score: np.int32) -> np.int32:
    """Two corrections applied to any evaluation, in White's view: a mop-up
    bonus against a bare king, and a scale-down of material that cannot win."""
    # ---- mop-up: drive a bare king to the edge (or the right corner) ----
    # Piece-square tables alone leave KBB and KBN unfinished before the fifty-
    # move rule intervenes. Once one side has only a king and the other has
    # mating material, reward cornering the king and bringing ours close.
    for side in range(2):
        loser = uint64(1) - uint64(side)
        if s[uint64(12) + loser] != s[loser * uint64(6) + uint64(5)]:
            continue  # the other side still has more than a king
        off = uint64(side) * uint64(6)
        nb = int32(popcount(s[off + uint64(2)]))
        nn = int32(popcount(s[off + uint64(1)]))
        heavy = s[off + uint64(3)] | s[off + uint64(4)]
        if heavy == uint64(0) and nb + nn < int32(2):
            break  # no mating material at all
        if heavy == uint64(0) and nb == int32(0):
            break  # knights alone cannot force mate
        wk = lsb(s[off + uint64(5)])
        lk = lsb(s[loser * uint64(6) + uint64(5)])
        lf = int32(lk & uint64(7))
        lr = int32(lk >> uint64(3))
        md = abs(int32(wk & uint64(7)) - lf) + abs(int32(wk >> uint64(3)) - lr)
        if heavy == uint64(0) and nb == int32(1) and nn >= int32(1):
            # bishop and knight: mate only in a corner of the bishop's colour
            bsq = lsb(s[off + uint64(2)])
            dark = ((int32(bsq & uint64(7)) + int32(bsq >> uint64(3))) & int32(1)) == int32(0)
            if dark:
                cd = min(lf + lr, (int32(7) - lf) + (int32(7) - lr))
            else:
                cd = min((int32(7) - lf) + lr, lf + (int32(7) - lr))
            bonus = int32(5) * (int32(14) - cd)
        else:
            cf = lf if lf < int32(4) else int32(7) - lf
            cr = lr if lr < int32(4) else int32(7) - lr
            bonus = int32(10) * (int32(6) - cf - cr)
        bonus += int32(4) * (int32(14) - md)
        score += bonus if side == 0 else -bonus
        break

    # ---- drawish material: a lone minor, or two knights, cannot win ----
    # Without this the engine happily trades down into KB v K "a bishop up".
    strong = uint64(0) if score > int32(0) else uint64(1)
    soff = strong * uint64(6)
    if s[soff] == uint64(0):  # no pawns
        heavy = s[soff + uint64(3)] | s[soff + uint64(4)]
        if heavy == uint64(0):
            nb = int32(popcount(s[soff + uint64(2)]))
            nn = int32(popcount(s[soff + uint64(1)]))
            if nb + nn <= int32(1):
                score = score // int32(16)
            elif nb == int32(0) and nn == int32(2):
                weak = uint64(1) - strong
                if s[uint64(12) + weak] == s[weak * uint64(6) + uint64(5)]:
                    score = score // int32(16)
    return score


@njit(int32(uint64[:], int8[:], _W_RO), cache=False, nogil=True)
def evaluate_raw(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8],
                 w: npt.NDArray[np.int32]) -> np.int32:
    """Tapered evaluation in WHITE's view, before the endgame corrections:
    material and piece-square tables, plus mobility,
    pawn structure, king safety and the bishop pair.

    Two scores are accumulated - one weighted for the middlegame, one for the
    endgame - and interpolated by how much material remains. That interpolation
    is the point: a king belongs behind its pawns at move ten and in the centre
    at move sixty, and one table cannot say both.

    Returned from the side to move's point of view.
    """
    mg = int32(0)
    eg = int32(0)
    phase = int32(0)
    occ = s[12] | s[13]
    wp = s[0]
    bp = s[6]

    # Squares each side's pawns attack. Mobility that walks into a pawn is not
    # mobility, so these are excluded from the safe-square counts below.
    wp_att = ((wp & ~FILE_A) << uint64(7)) | ((wp & ~FILE_H) << uint64(9))
    bp_att = ((bp & ~FILE_A) >> uint64(9)) | ((bp & ~FILE_H) >> uint64(7))

    # ---- material and piece-square: kept up to date by make() ----
    mg = int32(int64(s[19]))
    eg = int32(int64(s[20]))
    phase = int32(int64(s[21]))
    for pt in range(5):
        diff = int32(popcount(s[pt])) - int32(popcount(s[uint64(6) + uint64(pt)]))
        mg += diff * w[I_MAT_MG + pt]
        eg += diff * w[I_MAT_EG + pt]

    # ---- mobility, counted on squares not attacked by an enemy pawn ----
    # The same attack sets feed the king-safety count against the enemy
    # king, so both are gathered in one pass.
    ks_units_w = int32(0)      # attack units against WHITE's king (by black)
    ks_att_w = int32(0)
    ks_units_b = int32(0)
    ks_att_b = int32(0)
    for side in range(2):
        off = uint64(side) * uint64(6)
        own = s[12 + uint64(side)]
        bad = bp_att if side == 0 else wp_att
        sign = int32(1) if side == 0 else int32(-1)
        ek = lsb(s[(uint64(1) - uint64(side)) * uint64(6) + uint64(5)])
        ezone = KING_ZONE[1 - side, ek]
        units = int32(0)
        attackers = int32(0)

        hits = int32(0)
        b = s[off + uint64(1)]
        while b:
            sq = lsb(b)
            b &= b - uint64(1)
            a = KNIGHT_ATT[sq]
            c = popcount(a & ~own & ~bad)
            mg += sign * w[I_KN_MG + c]
            eg += sign * w[I_KN_EG + c]
            n = popcount(a & ezone)
            if n > uint64(0):
                hits += int32(n)
                attackers += int32(1)
        units += KS_WEIGHT[1] * hits
        hits = int32(0)
        b = s[off + uint64(2)]
        while b:
            sq = lsb(b)
            b &= b - uint64(1)
            a = bishop_att(sq, occ)
            c = popcount(a & ~own & ~bad)
            mg += sign * w[I_BI_MG + c]
            eg += sign * w[I_BI_EG + c]
            n = popcount(a & ezone)
            if n > uint64(0):
                hits += int32(n)
                attackers += int32(1)
        units += KS_WEIGHT[2] * hits
        hits = int32(0)
        b = s[off + uint64(3)]
        while b:
            sq = lsb(b)
            b &= b - uint64(1)
            a = rook_att(sq, occ)
            c = popcount(a & ~own & ~bad)
            mg += sign * w[I_RK_MG + c]
            eg += sign * w[I_RK_EG + c]
            n = popcount(a & ezone)
            if n > uint64(0):
                hits += int32(n)
                attackers += int32(1)
        units += KS_WEIGHT[3] * hits
        hits = int32(0)
        b = s[off + uint64(4)]
        while b:
            sq = lsb(b)
            b &= b - uint64(1)
            a = queen_att(sq, occ)
            c = popcount(a & ~own & ~bad)
            mg += sign * w[I_QU_MG + c]
            eg += sign * w[I_QU_EG + c]
            n = popcount(a & ezone)
            if n > uint64(0):
                hits += int32(n)
                attackers += int32(1)
        units += KS_WEIGHT[4] * hits
        if side == 0:
            ks_units_b = units          # white's pieces attack black's king
            ks_att_b = attackers
        else:
            ks_units_w = units
            ks_att_w = attackers

        # ---- bishop pair ----
        if popcount(s[off + uint64(2)]) >= uint64(2):
            mg += sign * w[I_BP_MG]
            eg += sign * w[I_BP_EG]

    # ---- pawn structure ----
    for side in range(2):
        pawns = s[uint64(side) * uint64(6)]
        theirs = bp if side == 0 else wp
        sign = int32(1) if side == 0 else int32(-1)
        b = pawns
        while b:
            sq = lsb(b)
            b &= b - uint64(1)
            f = sq & uint64(7)
            rel = sq >> uint64(3) if side == 0 else uint64(7) - (sq >> uint64(3))
            if (PASSED_MASK[side, sq] & theirs) == uint64(0):
                mg += sign * w[I_PA_MG + rel]
                eg += sign * w[I_PA_EG + rel]
                # in the endgame the kings decide: distance to the stop square
                if rel >= uint64(3):
                    stop = sq + uint64(8) if side == 0 else sq - uint64(8)
                    sf = int32(stop & uint64(7))
                    sr = int32(stop >> uint64(3))
                    ok = lsb(s[uint64(side) * uint64(6) + uint64(5)])
                    ek = lsb(s[(uint64(1) - uint64(side)) * uint64(6) + uint64(5)])
                    d_own = max(abs(int32(ok & uint64(7)) - sf), abs(int32(ok >> uint64(3)) - sr))
                    d_enemy = max(abs(int32(ek & uint64(7)) - sf), abs(int32(ek >> uint64(3)) - sr))
                    pk = w[I_PK_ENEMY] * d_enemy - w[I_PK_OWN] * d_own
                    eg += sign * int32(rel) * pk // int32(4)
            if (ADJACENT_FILES[f] & pawns) == uint64(0):
                mg += sign * w[I_ISO_MG]
                eg += sign * w[I_ISO_EG]
            if (FRONT_MASK[side, sq] & pawns) != uint64(0):
                mg += sign * w[I_DBL_MG]
                eg += sign * w[I_DBL_EG]

    # ---- threats and rook files ----
    for side in range(2):
        off = uint64(side) * uint64(6)
        sign = int32(1) if side == 0 else int32(-1)
        bad = bp_att if side == 0 else wp_att
        pieces = s[off + uint64(1)] | s[off + uint64(2)] | s[off + uint64(3)] | s[off + uint64(4)]
        n = int32(popcount(pieces & bad))
        mg += sign * n * w[I_THREAT_MG]
        eg += sign * n * w[I_THREAT_EG]
        own_p = s[off]
        their_p = s[(uint64(1) - uint64(side)) * uint64(6)]
        b = s[off + uint64(3)]
        while b:
            sq = lsb(b)
            b &= b - uint64(1)
            fbb = FILE_BB[sq & uint64(7)]
            if (fbb & own_p) == uint64(0):
                if (fbb & their_p) == uint64(0):
                    mg += sign * w[I_ROPEN_MG]
                    eg += sign * w[I_ROPEN_EG]
                else:
                    mg += sign * w[I_RSEMI_MG]
                    eg += sign * w[I_RSEMI_EG]

    # ---- king safety, middlegame only ----
    for side in range(2):
        ksq = lsb(s[uint64(side) * uint64(6) + uint64(5)])
        them = uint64(1) - uint64(side)
        off = them * uint64(6)

        # pawn shelter and storm on the king's file and its neighbours
        own_p = s[uint64(side) * uint64(6)]
        their_p = s[off]
        kf = int32(ksq & uint64(7))
        kr = int32(ksq >> uint64(3))
        step = int32(1) if side == 0 else int32(-1)
        shelter = int32(0)
        for df in range(-1, 2):
            ff = kf + int32(df)
            if ff < int32(0) or ff > int32(7):
                continue
            d_own = int32(7)
            d_their = int32(7)
            for d in range(1, 8):
                rr = kr + step * int32(d)
                if rr < int32(0) or rr > int32(7):
                    break
                bit = uint64(1) << uint64(rr * int32(8) + ff)
                if d_own == int32(7) and (own_p & bit) != uint64(0):
                    d_own = int32(d)
                if d_their == int32(7) and (their_p & bit) != uint64(0):
                    d_their = int32(d)
                if d_own != int32(7) and d_their != int32(7):
                    break
            shelter += w[I_SHELTER + d_own] + w[I_STORM + d_their]
        mg += shelter if side == 0 else -shelter
        if side == 0:
            units = ks_units_w
            attackers = ks_att_w
        else:
            units = ks_units_b
            attackers = ks_att_b
        if attackers >= int32(2):
            idx = (units * KS_ATTACKERS[min(attackers, int32(8))]) // int32(100)
            if idx > int32(99):
                idx = int32(99)
            penalty = (KS_TABLE[idx] * w[I_KS]) // int32(100)
            mg -= penalty if side == 0 else -penalty

    if phase > int32(PHASE_MAX):
        phase = int32(PHASE_MAX)
    score = (mg * phase + eg * (int32(PHASE_MAX) - phase)) // int32(PHASE_MAX)
    return score


@njit(int32(uint64[:], int8[:], _W_RO), cache=False, nogil=True)
def evaluate_w(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8],
               w: npt.NDArray[np.int32]) -> np.int32:
    """The hand evaluation from the side to move's view, endgame corrections
    applied."""
    score = endgame_adjust(s, evaluate_raw(s, mb, w))
    return score if s[14] == uint64(0) else -score


@njit(int32(uint64[:], int8[:], float32[:, ::1]), cache=False, nogil=True)
def evaluate_acc(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8],
                 acc: npt.NDArray[np.float32]) -> np.int32:
    """The shipped evaluation, with the network's accumulators for this node
    already maintained by the search. USE_NNUE is a compile-time constant,
    so the branch not taken costs nothing."""
    if USE_NNUE:
        pieces = int64(popcount(s[12] | s[13]))
        sc = nn_output(s[14], acc, bucket_of(pieces))   # side to move's view
        white = sc if s[14] == uint64(0) else -sc
        if NN_RESIDUAL:
            white += evaluate_raw(s, mb, WEIGHTS)
        # mop-up and drawn-material scaling apply to the whole evaluation,
        # so a bare bishop stays a draw whatever the network adds
        white = endgame_adjust(s, white)
        return white if s[14] == uint64(0) else -white
    return evaluate_w(s, mb, WEIGHTS)


@njit(int32(uint64[:], int8[:]), cache=False, nogil=True)
def evaluate(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8]) -> np.int32:
    """Evaluate a bare position: rebuilds the accumulators. For tools; the
    search uses evaluate_acc."""
    if USE_NNUE:
        sc = nn_eval(s, mb)
        white = sc if s[14] == uint64(0) else -sc
        if NN_RESIDUAL:
            white += evaluate_raw(s, mb, WEIGHTS)
        white = endgame_adjust(s, white)
        return white if s[14] == uint64(0) else -white
    return evaluate_w(s, mb, WEIGHTS)


@njit(int32(uint64[:]), cache=False, nogil=True)
def material_ref(s: npt.NDArray[np.uint64]) -> np.int32:
    """Material from the side to move's view, on the REFEREE's scale.

    This is what a game reaching the 300-ply cap is scored on, so near the cap
    it matters more than anything our own evaluation thinks.
    """
    us = s[14]
    sc = int32(0)
    for pt in range(5):
        mine = popcount(s[us * uint64(6) + uint64(pt)])
        theirs = popcount(s[(uint64(1) - us) * uint64(6) + uint64(pt)])
        sc += REF_VALUES[pt] * (int32(mine) - int32(theirs))
    return sc


@njit(int32(uint64[:], int8[:], float32[:, ::1], int64), cache=False, nogil=True)
def eval_adjusted(
        s: npt.NDArray[np.uint64],
        mb: npt.NDArray[np.int8],
        acc: npt.NDArray[np.float32],
        total_ply: int,
) -> np.int32:
    """Static evaluation, corrected for the two things the referee does that a
    normal chess evaluation does not model."""
    sc = evaluate_acc(s, mb, acc)

    # As the fifty-move clock climbs the position really is closer to a draw,
    # so shrink the advantage. This also makes the engine prefer a capture or
    # pawn move, which resets the clock, over shuffling.
    hm = int64(s[17])
    if hm > FIFTY_MOVE_SCALE_FROM:
        left = 100 - hm
        if left < 0:
            left = 0
        sc = int32(int64(sc) * left // (100 - FIFTY_MOVE_SCALE_FROM))

    # Approaching the ply cap the game is drawn whatever the position, so
    # an advantage is worth less the closer the cap comes (the rules changed
    # on 6 Sep from a material adjudication at 300 plies to a draw at 600).
    if total_ply > PLY_CAP - ADJUDICATION_BLEND:
        w = total_ply - (PLY_CAP - ADJUDICATION_BLEND)
        if w > ADJUDICATION_BLEND:
            w = ADJUDICATION_BLEND
        sc = int32(int64(sc) * (ADJUDICATION_BLEND - w) // ADJUDICATION_BLEND)
    return sc


@njit(boolean(uint64[:], int64, int64), cache=False, nogil=True)
def is_repetition(rep: npt.NDArray[np.uint64], idx: int, halfmove: int) -> bool:
    """Has the position at rep[idx] occurred before within the current
    halfmove window? One prior occurrence is scored as a draw - the standard
    twofold-in-search convention."""
    i = idx - 2
    stop = idx - halfmove
    if stop < 0:
        stop = 0
    while i >= stop:
        if rep[i] == rep[idx]:
            return True
        i -= 2
    return False


@njit(int32(int32, int64), forceinline=True, cache=False, nogil=True)
def to_tt(v: np.int32, ply: int) -> np.int32:
    """Mate scores are stored relative to the node, not the root."""
    if v >= MATE_IN_MAX:
        return v + int32(ply)
    if v <= -MATE_IN_MAX:
        return v - int32(ply)
    return v


@njit(int32(int32, int64), forceinline=True, cache=False, nogil=True)
def from_tt(v: np.int32, ply: int) -> np.int32:
    if v >= MATE_IN_MAX:
        return v - int32(ply)
    if v <= -MATE_IN_MAX:
        return v + int32(ply)
    return v



SEE_VALUE = np.array([100, 300, 300, 500, 900, 0], dtype=np.int32)


@njit(boolean(uint64[:], int8[:], uint32, int32), cache=False, nogil=True)
def see_ge(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8], mv: np.uint32,
           threshold: np.int32) -> bool:
    """Static exchange evaluation: does playing `mv` and letting both sides
    capture on the target square, least valuable attacker first, come out at
    least `threshold` ahead? Pins are ignored, as in most engines.

    In the usual swap-list formulation `swap` is the running balance
    from the point of view of the side whose turn it is to recapture, and the
    loop breaks as soon as either side would lose by continuing.
    """
    frm = uint64(mv & uint32(63))
    to = uint64((mv >> uint32(6)) & uint32(63))
    flag = (mv >> uint32(15)) & uint32(7)
    if flag == uint32(2):                       # castling never loses material
        return threshold <= int32(0)
    victim = int64(mb[to])
    if flag == uint32(1):
        swap = SEE_VALUE[0] - threshold         # en passant takes a pawn
    elif victim == 12:
        swap = -threshold
    else:
        swap = SEE_VALUE[victim % 6] - threshold
    if swap < int32(0):
        return False
    attacker = int64(mb[frm]) % 6
    swap = SEE_VALUE[attacker] - swap
    if swap <= int32(0):
        return True

    occ = (s[12] | s[13]) ^ (uint64(1) << frm)
    occ |= uint64(1) << to
    if flag == uint32(1):
        occ ^= uint64(1) << (to ^ uint64(8))
    stm = s[14]
    bq = s[2] | s[4] | s[8] | s[10]
    rq = s[3] | s[4] | s[9] | s[10]
    attackers = ((PAWN_ATT[1, to] & s[0]) | (PAWN_ATT[0, to] & s[6])
                 | (KNIGHT_ATT[to] & (s[1] | s[7]))
                 | (KING_ATT[to] & (s[5] | s[11]))
                 | (bishop_att(to, occ) & bq)
                 | (rook_att(to, occ) & rq))
    res = int32(1)
    while True:
        stm = uint64(1) - stm
        attackers &= occ
        mine = attackers & s[uint64(12) + stm]
        if mine == uint64(0):
            break
        res ^= int32(1)
        off = stm * uint64(6)
        bb = mine & s[off]
        if bb != uint64(0):
            swap = SEE_VALUE[0] - swap
            if swap < res:
                break
            occ ^= bb & (uint64(0) - bb)
            attackers |= bishop_att(to, occ) & bq
            continue
        bb = mine & s[off + uint64(1)]
        if bb != uint64(0):
            swap = SEE_VALUE[1] - swap
            if swap < res:
                break
            occ ^= bb & (uint64(0) - bb)
            continue
        bb = mine & s[off + uint64(2)]
        if bb != uint64(0):
            swap = SEE_VALUE[2] - swap
            if swap < res:
                break
            occ ^= bb & (uint64(0) - bb)
            attackers |= bishop_att(to, occ) & bq
            continue
        bb = mine & s[off + uint64(3)]
        if bb != uint64(0):
            swap = SEE_VALUE[3] - swap
            if swap < res:
                break
            occ ^= bb & (uint64(0) - bb)
            attackers |= rook_att(to, occ) & rq
            continue
        bb = mine & s[off + uint64(4)]
        if bb != uint64(0):
            swap = SEE_VALUE[4] - swap
            if swap < res:
                break
            occ ^= bb & (uint64(0) - bb)
            attackers |= (bishop_att(to, occ) & bq) | (rook_att(to, occ) & rq)
            continue
        # only the king is left: it may capture only if nothing defends
        if (attackers & ~s[uint64(12) + stm]) != uint64(0):
            return (res ^ int32(1)) != int32(0)
        return res != int32(0)
    return res != int32(0)

@njit(boolean(uint64[:]), forceinline=True, cache=False, nogil=True)
def has_non_pawn(s: npt.NDArray[np.uint64]) -> bool:
    """Null move is unsound in zugzwang, which is a pawn-endgame phenomenon."""
    us = s[14]
    off = us * uint64(6)
    occupied = (s[off + uint64(1)] | s[off + uint64(2)]
                | s[off + uint64(3)] | s[off + uint64(4)])
    return bool(occupied != uint64(0))


@njit(int64(uint64[:], int8[:], uint32[:], int32[:], int64, uint32, uint32[:, :],
            int32[:, :, :], int64, uint32), cache=False, nogil=True)
def score_moves(
        s: npt.NDArray[np.uint64],
        mb: npt.NDArray[np.int8],
        moves: npt.NDArray[np.uint32],
        scores: npt.NDArray[np.int32],
        n: int,
        tt_mv: np.uint32,
        killers: npt.NDArray[np.uint32],
        history: npt.NDArray[np.int32],
        ply: int,
        counter_mv: np.uint32,
) -> int:
    """Assign an ordering score to each generated move. `counter_mv` is the
    quiet move that last refuted the opponent's previous move."""
    us = int64(s[14])
    for i in range(n):
        mv = moves[i]
        if mv == tt_mv:
            scores[i] = int32(2_000_000)
            continue
        frm = (mv >> uint32(0)) & uint32(63)
        to = (mv >> uint32(6)) & uint32(63)
        promo = (mv >> uint32(12)) & uint32(7)
        victim = mb[to]
        if victim == 12 and ((mv >> uint32(15)) & uint32(7)) == uint32(1):
            victim = np.int8(0)                 # en passant captures a pawn
        sc = int32(0)
        if victim != 12:
            sc = int32(10) * MATERIAL[victim % 6] - MATERIAL[mb[frm] % 6]
            # a capture that loses the exchange goes after every quiet move
            sc += int32(1_000_000) if see_ge(s, mb, mv, int32(0)) else int32(-1_000_000)
        elif promo != uint32(0):
            sc = int32(1_500_000) + MATERIAL[promo]
        elif mv == killers[ply, 0]:
            sc = int32(900_000)
        elif mv == killers[ply, 1]:
            sc = int32(800_000)
        elif mv == counter_mv and mv != uint32(0):
            sc = int32(700_000)
        else:
            sc = history[us, frm, to]
        if promo != uint32(0) and victim != 12:
            sc += int32(500_000)
        scores[i] = sc
    return n


@njit(int64(uint32[:], int32[:], int64, int64), cache=False, nogil=True)
def sort_desc(moves: npt.NDArray[np.uint32], scores: npt.NDArray[np.int32],
              lo: int, hi: int) -> int:
    """Quicksort moves[lo:hi] by score, best first. Used once a node has
    gone past its first few moves and will probably try them all: from
    there one sort is cheaper than a selection pass per move."""
    while hi - lo > 12:
        pivot = scores[(lo + hi) // 2]
        i = lo
        j = hi - 1
        while i <= j:
            while scores[i] > pivot:
                i += 1
            while scores[j] < pivot:
                j -= 1
            if i <= j:
                scores[i], scores[j] = scores[j], scores[i]
                moves[i], moves[j] = moves[j], moves[i]
                i += 1
                j -= 1
        if j - lo < hi - i:
            sort_desc(moves, scores, lo, j + 1)
            lo = i
        else:
            sort_desc(moves, scores, i, hi)
            hi = j + 1
    for i in range(lo + 1, hi):                  # insertion sort for the rest
        sc = scores[i]
        mv = moves[i]
        j = i - 1
        while j >= lo and scores[j] < sc:
            scores[j + 1] = scores[j]
            moves[j + 1] = moves[j]
            j -= 1
        scores[j + 1] = sc
        moves[j + 1] = mv
    return 0


@njit(int64(uint32[:], int32[:], int64, int64), forceinline=True, cache=False, nogil=True)
def pick_best(
        moves: npt.NDArray[np.uint32],
        scores: npt.NDArray[np.int32],
        start: int,
        n: int,
) -> int:
    """Selection sort one move at a time - cheaper than sorting the whole list
    when a cutoff usually happens in the first few moves."""
    b = start
    for j in range(start + 1, n):
        if scores[j] > scores[b]:
            b = j
    if b != start:
        scores[start], scores[b] = scores[b], scores[start]
        moves[start], moves[b] = moves[b], moves[start]
    return b


@njit(int32(int32[:, :, :], int64, uint32, int32), forceinline=True, cache=False, nogil=True)
def hist_update(
        history: npt.NDArray[np.int32],
        us: int,
        mv: np.uint32,
        bonus: np.int32,
) -> np.int32:
    """History with gravity: the increment shrinks as the entry grows, so old
    information decays instead of saturating."""
    frm = (mv >> uint32(0)) & uint32(63)
    to = (mv >> uint32(6)) & uint32(63)
    h = history[us, frm, to]
    if bonus > int32(HISTORY_MAX):
        bonus = int32(HISTORY_MAX)
    if bonus < int32(-HISTORY_MAX):
        bonus = int32(-HISTORY_MAX)
    ab = bonus if bonus >= 0 else -bonus
    history[us, frm, to] = h + bonus - (h * ab) // int32(HISTORY_MAX)
    return np.int32(history[us, frm, to])


#: Everything the search touches, in one object. A numba call copies every
#: array argument by value (a dozen words each), and the search makes two or
#: three calls per node; one reference to this object costs a single word.
_CTX_SPEC = [
    ("stack", uint64[:, ::1]), ("mbs", int8[:, ::1]), ("buf", uint32[:, ::1]),
    ("sbuf", int32[:, ::1]), ("tt", uint64[:, ::1]), ("killers", uint32[:, ::1]),
    ("history", int32[:, :, ::1]), ("counter", uint32[:, :, ::1]), ("rep", uint64[::1]),
    ("evals", int32[::1]), ("ctl", int64[::1]), ("tbuf", int64[::1]),
    ("acc", float32[:, :, ::1]), ("out_moves", uint32[::1]), ("out_scores", int32[::1]),
]


@jitclass(_CTX_SPEC)  # type: ignore[no-untyped-call]
class SearchState:
    def __init__(self, stack, mbs, buf, sbuf, tt, killers, history, counter, rep, evals,  # type: ignore[no-untyped-def]
                 ctl, tbuf, acc, out_moves, out_scores):
        self.stack = stack
        self.mbs = mbs
        self.buf = buf
        self.sbuf = sbuf
        self.tt = tt
        self.killers = killers
        self.history = history
        self.counter = counter
        self.rep = rep
        self.evals = evals
        self.ctl = ctl
        self.tbuf = tbuf
        self.acc = acc
        self.out_moves = out_moves
        self.out_scores = out_scores


_CTX = SearchState.class_type.instance_type  # type: ignore[attr-defined]


@njit(int32(_CTX, int64, int32, int32), cache=False, nogil=True)
def quiesce(c: SearchState, ply: int, alpha: np.int32, beta: np.int32) -> np.int32:
    """Search captures only, so the evaluation is never measured mid-exchange.
    In check there is no standing pat: every evasion is searched, so a
    check at the horizon is answered, not assumed away."""
    c.ctl[C_NODES] += 1
    # The node-count test is the cheap gate: it keeps the clock syscall to one
    # poll per 2048 nodes, which measures at well under 0.1% of search time.
    if (c.ctl[C_NODES] & 2047) == 0 and (
            now_ns(c.tbuf) >= c.ctl[C_DEADLINE] or c.ctl[C_NODES] >= c.ctl[C_NODECAP]):
        c.ctl[C_STOPPED] = 1
    if c.ctl[C_STOPPED] == 1 or ply >= 120:
        return alpha

    s = c.stack[ply]
    us = s[14]
    in_check = attacked(s, lsb(s[us * uint64(6) + uint64(5)]), uint64(1) - us)
    if in_check:
        stand = int32(-MATE + ply)           # no standing pat: find an evasion
    else:
        stand = eval_adjusted(s, c.mbs[ply], c.acc[ply], c.ctl[C_GAMEPLY] + ply)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand

    if in_check:
        n = gen_moves(s, c.mbs[ply], c.buf[ply])
    else:
        n = gen_captures(s, c.mbs[ply], c.buf[ply])
    # captures and promotions only, ordered by MVV-LVA; every move in check
    for i in range(n):
        mv = c.buf[ply][i]
        to = (mv >> uint32(6)) & uint32(63)
        frm = mv & uint32(63)
        promo = (mv >> uint32(12)) & uint32(7)
        victim = c.mbs[ply][to]
        if victim == 12 and ((mv >> uint32(15)) & uint32(7)) == uint32(1):
            victim = np.int8(0)
        if victim != 12:
            c.sbuf[ply][i] = (int32(1_000_000) + int32(10) * MATERIAL[victim % 6]
                            - MATERIAL[c.mbs[ply][frm] % 6])
        elif promo == uint32(4):
            c.sbuf[ply][i] = int32(900_000)
        elif in_check:
            c.sbuf[ply][i] = int32(1)
        else:
            c.sbuf[ply][i] = int32(-1)

    best = stand
    legal = 0
    for i in range(n):
        pick_best(c.buf[ply], c.sbuf[ply], i, n)
        if c.sbuf[ply][i] < int32(0):
            break
        mv = c.buf[ply][i]
        if not in_check:
            # delta pruning: even winning this material would not reach alpha
            to = (mv >> uint32(6)) & uint32(63)
            victim = c.mbs[ply][to]
            gain = MATERIAL[victim % 6] if victim != 12 else int32(0)
            if stand + gain + int32(200) < alpha:
                continue
            # a losing capture cannot rescue a quiet position
            if not see_ge(s, c.mbs[ply], mv, int32(0)):
                continue
        if not make(c.stack[ply], c.mbs[ply], c.stack[ply + 1], c.mbs[ply + 1], mv):
            continue
        legal += 1
        if USE_NNUE:
            acc_update(c.acc[ply], c.acc[ply + 1], c.mbs[ply], c.mbs[ply + 1], s[14], mv)
        sc = -quiesce(c, ply + 1, -beta, -alpha)
        if c.ctl[C_STOPPED] == 1:
            return alpha
        if sc > best:
            best = sc
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            break
    if in_check and legal == 0:
        return int32(-MATE + ply)
    return best


@njit(int64(int64, int64, boolean, boolean, boolean, boolean, int32), cache=False, nogil=True)
def lmr_reduction(depth: int, legal: int, is_pv: bool, improving: bool,
                  is_killer: bool, bad_capture: bool, hist: np.int32) -> int:
    """How many plies to take off a late move. Kept out of negamax so the
    search kernel stays small enough to compile inside the init budget.
    A quiet move with a strong history is reduced less; one that has
    repeatedly failed, more."""
    dd = depth if depth < 63 else 63
    mm = legal if legal < 63 else 63
    r = int64(LMR[dd, mm])
    if is_pv:
        r -= 1
    if improving:
        r -= 1
    if is_killer:
        r -= 1
    if bad_capture:
        r -= 1
    r -= int64(hist) // int64(HISTORY_MAX // 2)
    if r < 0:
        r = 0
    if r > depth - 2:
        r = depth - 2
    return r


@njit(int64(uint32[:, :], int32[:, :, :], uint32[:, :, :], uint32[:], int64, int64, uint32,
            int64, int64), cache=False, nogil=True)
def on_cutoff(killers: npt.NDArray[np.uint32], history: npt.NDArray[np.int32],
              counter: npt.NDArray[np.uint32], quiet_list: npt.NDArray[np.uint32],
              ply: int, us: int, mv: np.uint32, depth: int, quiets: int) -> int:
    """A quiet move refuted the position: remember it as a killer for this
    ply and as the counter to the opponent's previous move, and mark down the
    quiets tried before it."""
    if killers[ply, 0] != mv:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = mv
    if ply > 0:
        prev = killers[ply - 1, 2]
        if prev != uint32(0):
            counter[us, prev & uint32(63), (prev >> uint32(6)) & uint32(63)] = mv
    bonus = int32(depth * depth * 16)
    hist_update(history, us, mv, bonus)
    for q in range(quiets - 1):
        hist_update(history, us, quiet_list[q], -bonus)
    return 0


# The table is one array of (key, data) pairs so that a probe touches a
# single cache line. data packs move (18 bits), score (16, two's complement),
# depth (8) and bound (2).
@njit(uint32(uint64), forceinline=True, cache=False, nogil=True)
def tt_mv_of(data: np.uint64) -> np.uint32:
    return uint32(data & uint64(0x3FFFF))


@njit(int32(uint64), forceinline=True, cache=False, nogil=True)
def tt_score_of(data: np.uint64) -> np.int32:
    v = int32((data >> uint64(18)) & uint64(0xFFFF))
    return v - int32(65536) if v >= int32(32768) else v


@njit(int64(uint64), forceinline=True, cache=False, nogil=True)
def tt_depth_of(data: np.uint64) -> int:
    return int64((data >> uint64(34)) & uint64(0xFF))


@njit(uint8(uint64), forceinline=True, cache=False, nogil=True)
def tt_bound_of(data: np.uint64) -> np.uint8:
    return uint8((data >> uint64(42)) & uint64(3))


@njit(uint64(uint32, int32, int64, uint8), forceinline=True, cache=False, nogil=True)
def tt_pack(mv: np.uint32, score: np.int32, depth: int, bound: np.uint8) -> np.uint64:
    return (uint64(mv & uint32(0x3FFFF)) | (uint64(int64(score) & 0xFFFF) << uint64(18))
            | (uint64(depth & 0xFF) << uint64(34)) | (uint64(bound) << uint64(42)))


@njit(int64(uint64[:, ::1], uint64, uint64, int32, int64, int64, uint8, uint32),
      cache=False, nogil=True)
def tt_store(tt: npt.NDArray[np.uint64], idx: np.uint64, key: np.uint64,
             best: np.int32, ply: int, depth: int, bound: np.uint8,
             best_move: np.uint32) -> int:
    """Depth-preferred replacement; an exact-key match always wins its slot."""
    same_key = tt[idx, 0] == key
    old = tt[idx, 1]
    replace = (not same_key or tt_depth_of(old) <= depth
               or bound == uint8(BOUND_EXACT))
    if replace:
        # a fail-low has no move worth keeping over one already stored here,
        # but a slot taken from another position must not keep its stale move
        mv = best_move if (bound != uint8(BOUND_UPPER) or not same_key) else tt_mv_of(old)
        tt[idx, 0] = key
        tt[idx, 1] = tt_pack(mv, to_tt(best, ply), depth, bound)
    return 0


@njit(int32(_CTX, int64, int64, int32, int32, boolean), cache=False, nogil=True)
def negamax(c: SearchState, ply: int, depth: int, alpha: np.int32, beta: np.int32,
            is_pv: bool) -> np.int32:
    c.stack = c.stack
    c.rep = c.rep
    c.ctl[C_NODES] += 1
    # The node-count test is the cheap gate: it keeps the clock syscall to one
    # poll per 2048 nodes, which measures at well under 0.1% of search time.
    if (c.ctl[C_NODES] & 2047) == 0 and (
            now_ns(c.tbuf) >= c.ctl[C_DEADLINE] or c.ctl[C_NODES] >= c.ctl[C_NODECAP]):
        c.ctl[C_STOPPED] = 1
    if c.ctl[C_STOPPED] == 1:
        return int32(0)

    s = c.stack[ply]
    us = s[14]
    them = uint64(1) - us
    key = s[18]
    ridx = c.ctl[C_REPBASE] + ply
    c.rep[ridx] = key

    # Contempt must be expressed from the ROOT player's point of view. In
    # negamax the sign flips on every ply, so a flat -contempt makes draws look
    # good to us at odd plies - exactly backwards.
    draw_score = (np.int32(-c.ctl[C_CONTEMPT]) if (ply & 1) == 0
                  else np.int32(c.ctl[C_CONTEMPT]))
    if ply > 0:
        # The referee stops here and the game is drawn, whatever is on the
        # board.
        if c.ctl[C_GAMEPLY] + ply >= PLY_CAP:
            return draw_score
        if s[17] >= uint64(100):
            return draw_score
        if is_repetition(c.rep, ridx, int64(s[17])):
            return draw_score
        # mate distance pruning: a shorter mate is already available
        if alpha < int32(-MATE + ply):
            alpha = int32(-MATE + ply)
        if beta > int32(MATE - ply - 1):
            beta = int32(MATE - ply - 1)
        if alpha >= beta:
            return alpha

    ksq = lsb(s[us * uint64(6) + uint64(5)])
    in_check = attacked(s, ksq, them)
    if in_check:
        depth += 1                      # check extension

    if depth <= 0:
        return quiesce(c, ply, alpha, beta)

    # ---- transposition table ----
    # A singular-extension probe searches this same position with one move
    # excluded; the table must then neither answer for it nor learn from it.
    excluded = c.killers[ply, 3]
    idx = key & uint64(TT_MASK)
    tt_hit = c.tt[idx, 0] == key and excluded == uint32(0)
    tt_data = c.tt[idx, 1] if tt_hit else uint64(0)
    tt_mv = tt_mv_of(tt_data) if tt_hit else uint32(0)
    if tt_hit and not is_pv and tt_depth_of(tt_data) >= depth:
        v = from_tt(tt_score_of(tt_data), ply)
        b = tt_bound_of(tt_data)
        if b == uint8(BOUND_EXACT):
            return v
        if b == uint8(BOUND_LOWER) and v >= beta:
            return v
        if b == uint8(BOUND_UPPER) and v <= alpha:
            return v

    # internal iterative reduction: with no hash move the ordering is poor,
    # so search a ply shallower and let the next iteration fix it up
    if depth >= IIR_MIN_DEPTH and tt_mv == uint32(0) and not in_check:
        depth -= 1

    static = eval_adjusted(s, c.mbs[ply], c.acc[ply], c.ctl[C_GAMEPLY] + ply)
    c.evals[ply] = static
    improving = (not in_check) and ply >= 2 and static > c.evals[ply - 2]

    if not is_pv and not in_check and excluded == uint32(0):
        # reverse futility: so far above beta that the opponent cannot claw back
        if depth <= RFP_MAX_DEPTH and static - int32(RFP_MARGIN) * int32(depth) >= beta:
            return static
        # null move: hand the opponent a free move and see if we still win
        if (depth >= NMP_MIN_DEPTH and static >= beta and has_non_pawn(s)
                and beta > int32(-MATE_IN_MAX)):
            r = 3 + depth // 5
            d = depth - 1 - r
            if d < 0:
                d = 0
            make_null(c.stack[ply], c.mbs[ply], c.stack[ply + 1], c.mbs[ply + 1])
            c.killers[ply, 2] = uint32(0)          # no move to c.counter
            if USE_NNUE:
                acc_copy(c.acc[ply], c.acc[ply + 1])
            sc = -negamax(c, ply + 1, d, -beta, -beta + int32(1), False)
            if c.ctl[C_STOPPED] == 1:
                return int32(0)
            if sc >= beta:
                return beta if sc >= int32(MATE_IN_MAX) else sc

    # ---- singular extension ----
    # If the hash move is far better than every alternative at a reduced
    # depth, it is the only move worth anything here: search it one ply
    # deeper, so the line the whole search rests on is seen further.
    extension = 0
    if (depth >= SE_MIN_DEPTH and excluded == uint32(0) and tt_hit and tt_mv != uint32(0)
            and tt_bound_of(tt_data) != uint8(BOUND_UPPER) and tt_depth_of(tt_data) >= depth - 3):
        tt_v = from_tt(tt_score_of(tt_data), ply)
        if abs(tt_v) < int32(MATE_IN_MAX):
            sbeta = tt_v - int32(SE_MARGIN) * int32(depth)
            c.killers[ply, 3] = tt_mv
            sv = negamax(c, ply, (depth - 1) // 2, sbeta - int32(1), sbeta, False)
            c.killers[ply, 3] = uint32(0)
            if c.ctl[C_STOPPED] == 1:
                return int32(0)
            if sv < sbeta:
                extension = 1

    counter_mv = uint32(0)
    if ply > 0:
        prev = c.killers[ply - 1, 2]
        if prev != uint32(0):
            counter_mv = c.counter[us, prev & uint32(63), (prev >> uint32(6)) & uint32(63)]
    n = gen_moves(s, c.mbs[ply], c.buf[ply])
    score_moves(s, c.mbs[ply], c.buf[ply], c.sbuf[ply], n, tt_mv, c.killers, c.history, ply,
                counter_mv)

    old_alpha = alpha
    best = int32(-INF)
    best_move = uint32(0)
    legal = 0
    quiets = 0
    quiet_list = np.empty(64, dtype=np.uint32)

    for i in range(n):
        if i < 4:
            pick_best(c.buf[ply], c.sbuf[ply], i, n)
        elif i == 4:
            sort_desc(c.buf[ply], c.sbuf[ply], i, n)
        mv = c.buf[ply][i]
        if mv == excluded:
            continue
        to = (mv >> uint32(6)) & uint32(63)
        is_quiet = (c.mbs[ply][to] == 12 and ((mv >> uint32(12)) & uint32(7)) == uint32(0)
                    and ((mv >> uint32(15)) & uint32(7)) != uint32(1))
        bad_capture = c.sbuf[ply][i] < int32(-500_000)

        if not is_pv and not in_check and best > int32(-MATE_IN_MAX):
            # late move pruning: enough quiet moves tried, and none of them helped
            if (is_quiet and depth <= LMP_MAX_DEPTH
                    and quiets >= LMP[1 if improving else 0, depth]):
                continue
            # futility: a quiet move cannot lift a hopeless static score to alpha
            if (is_quiet and depth <= FUTILITY_DEPTH
                    and static + int32(FUTILITY_MARGIN) * int32(depth)
                    + (int32(FUTILITY_IMPROVING) if improving else int32(0)) <= alpha):
                continue
            # a capture that loses more than the depth could win back
            if (bad_capture and depth <= SEE_PRUNE_DEPTH
                    and not see_ge(s, c.mbs[ply], mv, int32(-SEE_PRUNE_MARGIN * depth))):
                continue

        if not make(s, c.mbs[ply], c.stack[ply + 1], c.mbs[ply + 1], mv):
            continue
        new_depth = depth - 1 + (extension if mv == tt_mv else 0)
        c.killers[ply, 2] = mv                     # what the child is replying to
        if USE_NNUE:
            acc_update(c.acc[ply], c.acc[ply + 1], c.mbs[ply], c.mbs[ply + 1], us, mv)
        legal += 1
        if is_quiet and quiets < 64:
            quiet_list[quiets] = mv
            quiets += 1

        # ---- late move reductions ----
        r = 0
        if depth >= 3 and legal > 2 and (is_quiet or bad_capture):
            hist = c.history[us, mv & uint32(63), to] if is_quiet else int32(0)
            r = lmr_reduction(depth, legal, is_pv, improving,
                              (mv == c.killers[ply, 0] or mv == c.killers[ply, 1]
                               or mv == counter_mv),
                              bad_capture, hist)

        # ---- principal variation search ----
        if legal == 1:
            sc = -negamax(c, ply + 1, new_depth, -beta, -alpha, is_pv)
        else:
            sc = -negamax(c, ply + 1, new_depth - r, -alpha - int32(1), -alpha, False)
            if sc > alpha and r > 0:          # reduced search beat alpha: verify
                sc = -negamax(c, ply + 1, new_depth, -alpha - int32(1), -alpha, False)
            if is_pv and sc > alpha and sc < beta:
                sc = -negamax(c, ply + 1, new_depth, -beta, -alpha, True)
        if c.ctl[C_STOPPED] == 1:
            return int32(0)

        if sc > best:
            best = sc
            best_move = mv
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            if is_quiet:
                on_cutoff(c.killers, c.history, c.counter, quiet_list, ply, int64(us), mv, depth,
                          quiets)
            break

    if legal == 0:
        if excluded != uint32(0):
            return alpha                         # only the excluded move was legal
        return int32(-MATE + ply) if in_check else int32(0)
    if excluded != uint32(0):
        return best                              # a probe result, not this position's

    bound = (uint8(BOUND_LOWER) if best >= beta
             else uint8(BOUND_EXACT) if best > old_alpha
             else uint8(BOUND_UPPER))
    tt_store(c.tt, idx, key, best, ply, depth, bound, best_move)
    return best


@njit(int64(_CTX, int64, int32, int32), cache=False, nogil=True)
def search_root(c: SearchState, depth: int, alpha: np.int32, beta: np.int32) -> int:
    """One iteration at `depth`. Scores every root move so the driver can reject
    one for a reason the search cannot see, and returns how many it scored."""
    c.ctl = c.ctl
    s = c.stack[0]
    key = s[18]
    c.rep[c.ctl[C_REPBASE]] = key
    idx = key & uint64(TT_MASK)
    tt_mv = tt_mv_of(c.tt[idx, 1]) if c.tt[idx, 0] == key else uint32(0)

    n = gen_moves(s, c.mbs[0], c.buf[0])
    score_moves(s, c.mbs[0], c.buf[0], c.sbuf[0], n, tt_mv, c.killers, c.history, 0, uint32(0))

    best = int32(-INF)
    best_move = uint32(0)
    count = 0
    for i in range(n):
        pick_best(c.buf[0], c.sbuf[0], i, n)
        mv = c.buf[0][i]
        if not make(s, c.mbs[0], c.stack[1], c.mbs[1], mv):
            continue
        c.killers[0, 2] = mv
        if USE_NNUE:
            acc_update(c.acc[0], c.acc[1], c.mbs[0], c.mbs[1], s[14], mv)
        if count == 0:
            sc = -negamax(c, 1, depth - 1, -beta, -alpha, True)
        else:
            sc = -negamax(c, 1, depth - 1, -alpha - int32(1), -alpha, False)
            if sc > alpha:
                sc = -negamax(c, 1, depth - 1, -beta, -alpha, True)
        if c.ctl[C_STOPPED] == 1:
            return -count if count > 0 else 0
        c.out_moves[count] = mv
        c.out_scores[count] = sc
        count += 1
        if sc > best:
            best = sc
            best_move = mv
        if sc > alpha:
            alpha = sc

    if count > 0:
        c.tt[idx, 0] = key
        c.tt[idx, 1] = tt_pack(best_move, to_tt(best, 0), depth, uint8(BOUND_EXACT))
    return count


@njit(int64(int32[:, :, :]), cache=False, nogil=True)
def age_history(history: npt.NDArray[np.int32]) -> int:
    """Halve history between moves: the previous position's refutations are a
    hint, not a fact, once the position has changed."""
    for a in range(2):
        for b in range(64):
            for c in range(64):
                history[a, b, c] //= int32(2)
    return 0
