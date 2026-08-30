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
from numba import njit
from numba.types import boolean, int8, int16, int32, int64, uint8, uint32, uint64

from .clock import now_ns
from .core import attacked, gen_moves, lsb, make, make_null
from .evaltables import FLIP_SQ, MATERIAL, PST
from .tune import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    HISTORY_MAX,
    INF,
    LMP,
    LMP_MAX_DEPTH,
    LMR,
    MATE,
    MATE_IN_MAX,
    NMP_MIN_DEPTH,
    RFP_MARGIN,
    RFP_MAX_DEPTH,
    TT_MASK,
)

_J = dict(cache=False, nogil=True)

# ctl slots, so the search can report back without returning tuples
C_NODES, C_DEADLINE, C_STOPPED, C_REPBASE, C_NODECAP, C_CONTEMPT = 0, 1, 2, 3, 4, 5


@njit(int32(uint64[:], int8[:]), **_J)
def evaluate(s, mb):
    """Material and piece-square, from the side to move's point of view."""
    sc = int32(0)
    for sq in range(64):
        pc = mb[sq]
        if pc == 12:
            continue
        if pc < 6:
            sc += MATERIAL[pc] + PST[pc][sq]
        else:
            sc -= MATERIAL[pc - 6] + PST[pc - 6][FLIP_SQ[sq]]
    return sc if s[14] == uint64(0) else -sc


@njit(boolean(uint64[:], int64, int64), **_J)
def is_repetition(rep, idx, halfmove):
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


@njit(int32(int32, int64), inline="always", **_J)
def to_tt(v, ply):
    """Mate scores are stored relative to the node, not the root."""
    if v >= MATE_IN_MAX:
        return v + int32(ply)
    if v <= -MATE_IN_MAX:
        return v - int32(ply)
    return v


@njit(int32(int32, int64), inline="always", **_J)
def from_tt(v, ply):
    if v >= MATE_IN_MAX:
        return v - int32(ply)
    if v <= -MATE_IN_MAX:
        return v + int32(ply)
    return v


@njit(boolean(uint64[:]), inline="always", **_J)
def has_non_pawn(s):
    """Null move is unsound in zugzwang, which is a pawn-endgame phenomenon."""
    us = s[14]
    off = us * uint64(6)
    return (s[off + uint64(1)] | s[off + uint64(2)]
            | s[off + uint64(3)] | s[off + uint64(4)]) != uint64(0)


@njit(int64(uint64[:], int8[:], uint32[:], int32[:], int64, uint32, uint32[:, :],
            int32[:, :, :], int64), **_J)
def score_moves(s, mb, moves, scores, n, tt_mv, killers, history, ply):
    """Assign an ordering score to each generated move."""
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
        sc = int32(0)
        if victim != 12:
            sc = int32(1_000_000) + int32(10) * MATERIAL[victim % 6] - MATERIAL[mb[frm] % 6]
        elif promo != uint32(0):
            sc = int32(1_500_000) + MATERIAL[promo]
        elif mv == killers[ply, 0]:
            sc = int32(900_000)
        elif mv == killers[ply, 1]:
            sc = int32(800_000)
        else:
            sc = history[us, frm, to]
        if promo != uint32(0) and victim != 12:
            sc += int32(500_000)
        scores[i] = sc
    return n


@njit(int64(uint32[:], int32[:], int64, int64), inline="always", **_J)
def pick_best(moves, scores, start, n):
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


@njit(int32(int32[:, :, :], int64, uint32, int32), inline="always", **_J)
def hist_update(history, us, mv, bonus):
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
    return history[us, frm, to]


@njit(int32(uint64[:, :], int8[:, :], uint32[:, :], int32[:, :], int64, int32, int32,
            int64[:], int64[::1]), **_J)
def quiesce(stack, mbs, buf, sbuf, ply, alpha, beta, ctl, tbuf):
    """Search captures only, so the evaluation is never measured mid-exchange."""
    ctl[C_NODES] += 1
    if (ctl[C_NODES] & 2047) == 0:
        if now_ns(tbuf) >= ctl[C_DEADLINE] or ctl[C_NODES] >= ctl[C_NODECAP]:
            ctl[C_STOPPED] = 1
    if ctl[C_STOPPED] == 1 or ply >= 120:
        return alpha

    s = stack[ply]
    stand = evaluate(s, mbs[ply])
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand

    n = gen_moves(s, mbs[ply], buf[ply])
    # captures and promotions only, ordered by MVV-LVA
    for i in range(n):
        mv = buf[ply][i]
        to = (mv >> uint32(6)) & uint32(63)
        frm = mv & uint32(63)
        promo = (mv >> uint32(12)) & uint32(7)
        victim = mbs[ply][to]
        if victim != 12:
            sbuf[ply][i] = (int32(1_000_000) + int32(10) * MATERIAL[victim % 6]
                            - MATERIAL[mbs[ply][frm] % 6])
        elif promo == uint32(4):
            sbuf[ply][i] = int32(900_000)
        else:
            sbuf[ply][i] = int32(-1)

    best = stand
    for i in range(n):
        pick_best(buf[ply], sbuf[ply], i, n)
        if sbuf[ply][i] < int32(0):
            break
        # delta pruning: even winning this material would not reach alpha
        mv = buf[ply][i]
        to = (mv >> uint32(6)) & uint32(63)
        victim = mbs[ply][to]
        gain = MATERIAL[victim % 6] if victim != 12 else int32(0)
        if stand + gain + int32(200) < alpha:
            continue
        if not make(stack[ply], mbs[ply], stack[ply + 1], mbs[ply + 1], mv):
            continue
        sc = -quiesce(stack, mbs, buf, sbuf, ply + 1, -beta, -alpha, ctl, tbuf)
        if ctl[C_STOPPED] == 1:
            return alpha
        if sc > best:
            best = sc
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            break
    return best


@njit(int32(uint64[:, :], int8[:, :], uint32[:, :], int32[:, :],
            uint64[:], uint32[:], int16[:], int8[:], uint8[:],
            uint32[:, :], int32[:, :, :], uint64[:], int32[:],
            int64, int64, int32, int32, boolean, int64[:], int64[::1]), **_J)
def negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth, tt_bound,
            killers, history, rep, evals, ply, depth, alpha, beta, is_pv, ctl, tbuf):
    ctl[C_NODES] += 1
    if (ctl[C_NODES] & 2047) == 0:
        if now_ns(tbuf) >= ctl[C_DEADLINE] or ctl[C_NODES] >= ctl[C_NODECAP]:
            ctl[C_STOPPED] = 1
    if ctl[C_STOPPED] == 1:
        return int32(0)

    s = stack[ply]
    us = s[14]
    them = uint64(1) - us
    key = s[18]
    ridx = ctl[C_REPBASE] + ply
    rep[ridx] = key

    # Contempt must be expressed from the ROOT player's point of view. In
    # negamax the sign flips on every ply, so a flat -contempt makes draws look
    # good to us at odd plies - exactly backwards.
    draw_score = int32(-ctl[C_CONTEMPT]) if (ply & 1) == 0 else int32(ctl[C_CONTEMPT])
    if ply > 0:
        if s[17] >= uint64(100):
            return draw_score
        if is_repetition(rep, ridx, int64(s[17])):
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
        return quiesce(stack, mbs, buf, sbuf, ply, alpha, beta, ctl, tbuf)

    # ---- transposition table ----
    idx = key & uint64(TT_MASK)
    tt_hit = tt_key[idx] == key
    tt_mv = tt_move[idx] if tt_hit else uint32(0)
    if tt_hit and not is_pv and int64(tt_depth[idx]) >= depth:
        v = from_tt(int32(tt_score[idx]), ply)
        b = tt_bound[idx]
        if b == uint8(BOUND_EXACT):
            return v
        if b == uint8(BOUND_LOWER) and v >= beta:
            return v
        if b == uint8(BOUND_UPPER) and v <= alpha:
            return v

    static = evaluate(s, mbs[ply])
    evals[ply] = static
    improving = (not in_check) and ply >= 2 and static > evals[ply - 2]

    if not is_pv and not in_check:
        # reverse futility: so far above beta that the opponent cannot claw back
        if depth <= RFP_MAX_DEPTH and static - int32(RFP_MARGIN) * int32(depth) >= beta:
            return static
        # null move: hand the opponent a free move and see if we still win
        if (depth >= NMP_MIN_DEPTH and static >= beta and has_non_pawn(s)
                and beta > int32(-MATE_IN_MAX)):
            r = int64(3) + depth // 5
            d = depth - 1 - r
            if d < 0:
                d = 0
            make_null(stack[ply], mbs[ply], stack[ply + 1], mbs[ply + 1])
            sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                          tt_bound, killers, history, rep, evals, ply + 1, d,
                          -beta, -beta + int32(1), False, ctl, tbuf)
            if ctl[C_STOPPED] == 1:
                return int32(0)
            if sc >= beta:
                return beta if sc >= int32(MATE_IN_MAX) else sc

    n = gen_moves(s, mbs[ply], buf[ply])
    score_moves(s, mbs[ply], buf[ply], sbuf[ply], n, tt_mv, killers, history, ply)

    old_alpha = alpha
    best = int32(-INF)
    best_move = uint32(0)
    legal = 0
    quiets = 0
    quiet_list = np.empty(64, dtype=np.uint32)

    for i in range(n):
        pick_best(buf[ply], sbuf[ply], i, n)
        mv = buf[ply][i]
        to = (mv >> uint32(6)) & uint32(63)
        is_quiet = mbs[ply][to] == 12 and ((mv >> uint32(12)) & uint32(7)) == uint32(0)

        # late move pruning: enough quiet moves tried, and none of them helped
        if (not is_pv and not in_check and is_quiet and best > int32(-MATE_IN_MAX)
                and depth <= LMP_MAX_DEPTH
                and quiets >= LMP[1 if improving else 0, depth]):
            continue

        if not make(s, mbs[ply], stack[ply + 1], mbs[ply + 1], mv):
            continue
        legal += 1
        if is_quiet and quiets < 64:
            quiet_list[quiets] = mv
            quiets += 1

        # ---- late move reductions ----
        r = int64(0)
        if depth >= 3 and legal > 2 and is_quiet:
            dd = depth if depth < 63 else 63
            mm = legal if legal < 63 else 63
            r = int64(LMR[dd, mm])
            if is_pv:
                r -= 1
            if improving:
                r -= 1
            if mv == killers[ply, 0] or mv == killers[ply, 1]:
                r -= 1
            if r < 0:
                r = 0
            if r > depth - 2:
                r = depth - 2

        # ---- principal variation search ----
        if legal == 1:
            sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                          tt_bound, killers, history, rep, evals, ply + 1, depth - 1,
                          -beta, -alpha, is_pv, ctl, tbuf)
        else:
            sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                          tt_bound, killers, history, rep, evals, ply + 1, depth - 1 - r,
                          -alpha - int32(1), -alpha, False, ctl, tbuf)
            if sc > alpha and r > 0:          # reduced search beat alpha: verify
                sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                              tt_bound, killers, history, rep, evals, ply + 1, depth - 1,
                              -alpha - int32(1), -alpha, False, ctl, tbuf)
            if is_pv and sc > alpha and sc < beta:
                sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                              tt_bound, killers, history, rep, evals, ply + 1, depth - 1,
                              -beta, -alpha, True, ctl, tbuf)
        if ctl[C_STOPPED] == 1:
            return int32(0)

        if sc > best:
            best = sc
            best_move = mv
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            if is_quiet:
                if killers[ply, 0] != mv:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = mv
                bonus = int32(depth * depth * 16)
                hist_update(history, int64(us), mv, bonus)
                for q in range(quiets - 1):     # penalise the quiets that failed
                    hist_update(history, int64(us), quiet_list[q], -bonus)
            break

    if legal == 0:
        return int32(-MATE + ply) if in_check else int32(0)

    bound = (uint8(BOUND_LOWER) if best >= beta
             else uint8(BOUND_EXACT) if best > old_alpha
             else uint8(BOUND_UPPER))
    # depth-preferred replacement, but an exact-key match always wins its slot
    if tt_key[idx] != key or int64(tt_depth[idx]) <= depth or bound == uint8(BOUND_EXACT):
        tt_key[idx] = key
        tt_score[idx] = int16(to_tt(best, ply))
        tt_depth[idx] = int8(depth)
        tt_bound[idx] = bound
        if bound != uint8(BOUND_UPPER) or tt_key[idx] != key:
            tt_move[idx] = best_move
    return best


@njit(int64(uint64[:, :], int8[:, :], uint32[:, :], int32[:, :],
            uint64[:], uint32[:], int16[:], int8[:], uint8[:],
            uint32[:, :], int32[:, :, :], uint64[:], int32[:],
            int64, int32, int32, int64[:], int64[::1], uint32[:], int32[:]), **_J)
def search_root(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth, tt_bound,
                killers, history, rep, evals, depth, alpha, beta, ctl, tbuf,
                out_moves, out_scores):
    """One iteration at `depth`. Scores every root move so the driver can reject
    one for a reason the search cannot see, and returns how many it scored."""
    s = stack[0]
    key = s[18]
    rep[ctl[C_REPBASE]] = key
    idx = key & uint64(TT_MASK)
    tt_mv = tt_move[idx] if tt_key[idx] == key else uint32(0)

    n = gen_moves(s, mbs[0], buf[0])
    score_moves(s, mbs[0], buf[0], sbuf[0], n, tt_mv, killers, history, 0)

    best = int32(-INF)
    best_move = uint32(0)
    count = 0
    for i in range(n):
        pick_best(buf[0], sbuf[0], i, n)
        mv = buf[0][i]
        if not make(s, mbs[0], stack[1], mbs[1], mv):
            continue
        if count == 0:
            sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                          tt_bound, killers, history, rep, evals, 1, depth - 1,
                          -beta, -alpha, True, ctl, tbuf)
        else:
            sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                          tt_bound, killers, history, rep, evals, 1, depth - 1,
                          -alpha - int32(1), -alpha, False, ctl, tbuf)
            if sc > alpha:
                sc = -negamax(stack, mbs, buf, sbuf, tt_key, tt_move, tt_score, tt_depth,
                              tt_bound, killers, history, rep, evals, 1, depth - 1,
                              -beta, -alpha, True, ctl, tbuf)
        if ctl[C_STOPPED] == 1:
            return -count if count > 0 else 0
        out_moves[count] = mv
        out_scores[count] = sc
        count += 1
        if sc > best:
            best = sc
            best_move = mv
        if sc > alpha:
            alpha = sc

    if count > 0:
        tt_key[idx] = key
        tt_move[idx] = best_move
        tt_score[idx] = int16(to_tt(best, 0))
        tt_depth[idx] = int8(depth)
        tt_bound[idx] = uint8(BOUND_EXACT)
    return count


@njit(int64(int32[:, :, :]), **_J)
def age_history(history):
    """Halve history between moves: the previous position's refutations are a
    hint, not a fact, once the position has changed."""
    for a in range(2):
        for b in range(64):
            for c in range(64):
                history[a, b, c] //= int32(2)
    return 0
