"""Search constants and precomputed tables.

Values follow Ethereal and Berserk where those engines publish them; anything
marked "tune" is a starting point to be measured by self-play, not a result.
"""
import math

import numpy as np
import numpy.typing as npt

MAX_PLY = 128
MATE = 30000
#: Scores at or beyond this are mate scores and need ply adjustment in the TT.
MATE_IN_MAX = MATE - MAX_PLY
INF = 1 << 20

TT_BITS = 22                      # 4.2M entries ~= 75 MB, well inside 2 GB
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
BOUND_NONE, BOUND_LOWER, BOUND_UPPER, BOUND_EXACT = 0, 1, 2, 3

#: Draws are scored slightly against us so that a winning engine does not accept
#: a repetition. Small on purpose - the 1990s cautionary cases are engines that
#: refused a saving repetition and lost outright. Decayed when not winning.
CONTEMPT = 30

RFP_MARGIN = 75                   # per ply of depth, reverse futility  (tune)
RFP_MAX_DEPTH = 8
NMP_MIN_DEPTH = 3
LMP_MAX_DEPTH = 8
FUTILITY_DEPTH = 6                # prune quiet moves below this depth...
FUTILITY_MARGIN = 90              # ...when static + margin*depth cannot reach alpha
FUTILITY_IMPROVING = 60           # extra margin when the position is improving
SEE_PRUNE_DEPTH = 6               # prune losing captures below this depth
SEE_PRUNE_MARGIN = 100            # ...when they lose more than this per ply
ASPIRATION_DELTA = 25

#: LMR[depth][move_number]. Berserk's formula: 0.7844 + ln(d)*ln(m)/2.4696.
LMR: npt.NDArray[np.int8] = np.zeros((64, 64), dtype=np.int8)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR[_d, _m] = int(0.7844 + math.log(_d) * math.log(_m) / 2.4696)

#: LMP[improving][depth] - how many quiet moves to try before pruning the rest.
LMP: npt.NDArray[np.int16] = np.zeros((2, LMP_MAX_DEPTH + 1), dtype=np.int16)
for _d in range(LMP_MAX_DEPTH + 1):
    LMP[0, _d] = int((3 + _d * _d) / 2)
    LMP[1, _d] = int(3 + _d * _d)

HISTORY_MAX = 16384
IIR_MIN_DEPTH = 4     # internal iterative reduction: no hash move at this depth or more

#: The referee stops the game at this many plies and awards it on raw material,
#: using its own scale - not on our evaluation.
PLY_CAP = 300
#: Over the last this many plies before the cap, the evaluation is blended
#: toward that raw material count, because that is what the game will actually
#: be scored on. A crushing attack is worth nothing to an adjudicator.
ADJUDICATION_BLEND = 80
#: Score returned for a position that reaches the cap materially ahead. Well
#: above any positional evaluation, well below a real mate.
ADJUDICATION_WIN = 20000
#: Referee's piece values, in centipawns, on its scale: P1 N3 B3 R5 Q9.
#: An array rather than a tuple so the search can index it with a loop variable.
REF_VALUES: npt.NDArray[np.int32] = np.array([100, 300, 300, 500, 900, 0], dtype=np.int32)
#: Above this halfmove clock the advantage is scaled down, because the position
#: is genuinely closer to a draw and converting sooner is worth more.
FIFTY_MOVE_SCALE_FROM = 20
