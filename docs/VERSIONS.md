# Chessathon builds — what each one contains and what was measured

Every figure is from `tools/ab.py`: the two builds play the same openings with
both colours, 300 games at 60 ms per move unless stated, Elo with a 95%
interval. A change ships only when the interval does not sit below zero.

## v1 — uploaded 3 Sep (git tag `v1-uploaded`, commit f488b1e)

Numba bitboard engine: magic bitboards, PVS with transposition table,
null-move, late-move reductions, reverse futility and late-move pruning,
quiescence with delta pruning, tapered PeSTO evaluation plus mobility, pawn
structure, king-safety attack units and the bishop pair. Repetition and
fifty-move aware, referee's 300-ply adjudication modelled.

    vs Stockfish 16 @ Elo 2400, 100 ms/move    implied 2414 +/- 147

Rated ladder with v1: W W D L W L W (rounds 1-7).

## v2 — 4 Sep (tag `v2`, commit 9a2c2da)

| change | measured |
|---|---|
| Spend more of the clock (0.024 -> 0.038 of the pool per move) | +85 +/- 71 for 1.6x time |
| Static exchange evaluation: losing captures pruned in quiescence, ordered last, pruned at low depth, reduced | **+75 +/- 40** |
| Futility pruning of quiet moves at depth <= 6; stale hash-move fix | +19 +/- 40 |
| Mop-up evaluation for KBB / KBN v K; lone minor / two knights scaled to a draw | not measurable in games; KBB/KBN now mated instead of drawn |

Rated ladder with v2: L (round 8, Black).

## v3 — 4 Sep (tag `v3`, commit f713e47)

| change | measured |
|---|---|
| Pondering: search the position the opponent is looking at on their clock, filling the hash table; stopped and joined before every real search | **+109 +/- 51** (200 games, two cores) |
| Aspiration window opens fully after three failures or on a mate score (round 8 burnt 4.5 s of a 5.9 s clock on one move re-searching) | clock safety |
| Hard time ceiling: half of the remaining clock less one second, 50 ms floor | clock safety; replay of round 8's ending never below 1.4 s |

## Final standing — 11 Sep

Two stages, scored separately by the platform.

### Qualifier ladder (Rated 1–109, 4–11 Sep) — #47 of 465

Rating **2319**, peak **2372** (after round 76), record **35W 37D 31L** over 103
decided games (109 pairings; 6 voided). Top 10% of the field. 21 of the 24 builds
were uploaded and validated on the platform.

### Final qualification Swiss (Final 110–122, 11 Sep afternoon) — #28 of 334

13 rounds over locked builds, the stage that seeds the London final (50 seats,
one per UK university student in seed order). Score **9.0/13** — **7W 4D 2L**,
Buchholz 106.0, stage rating **2516**, our best rating of the event. Played
entirely on v24. We then withdrew from the London final — we could not attend the
event on 12 Sep — so round 122 was the engine's last competitive game.

Combined over both stages: **42W 41D 33L** in 116 decided games.

### Rank through the ladder, reconstructed

Every team's `/team/<id>` page carries a rating graph whose SVG path is the exact
per-round rating series, and the axis grid gives the calibration. Scraping all 465
of them and sorting at each round recovers the ladder table as it stood after any
round. The reconstruction reproduces the published final table exactly (#47, 2319),
and every team's last point matches its published rating to within 3 points.
Full series in `docs/ladder_rank.csv`; chart in `docs/ladder_rank.png`.

| after round | day | rank | teams rated | rating |
|---|---|---|---|---|
| 15 | 4 Sep | **15** | 237 | 1820 |
| 30 | 5 Sep | **19** | 290 | 1911 |
| 45 | 6 Sep | **10** | 329 | 2238 |
| 60 | 7 Sep | **28** | 369 | 2132 |
| 75 | 8 Sep | **17** | 402 | 2361 |
| 90 | 9 Sep | **39** | 427 | 2252 |
| 105 | 10 Sep | **54** | 448 | 2222 |
| **109** | **11 Sep** | **47** | **465** | **2319** |

Best rank of the event: **7th, after round 43** (6 Sep, field of 325). Worst after
the field had settled: **66th, after round 63** (8 Sep) — the bottom of the slide
caused by the three unmeasured clock changes shipped on 7 Sep, which v19 reverted;
rank was back to 17th by the end of 8 Sep. From there the drift down to ~54th over
9–10 Sep was not a regression in our play — the rating held between 2200 and 2372 —
but the field arriving and improving faster than we did.

v24 was the active build from round 103 on. Over the last seven ladder rounds it
went **4W 1D 2L** (2225 → 2319), then **7W 4D 2L** across the 13-round final Swiss.
Three of the ladder wins were as Black in the closed structures that had cost
R77/R82/R83 two days earlier. Twenty games is still well inside noise for a change
of this size: recorded as what happened, not as a measured gain.

## v24 — 10 Sep afternoon (tag `v24`)

| change | measured |
|---|---|
| Deep opening book: 547 new positions — every position where we were to move in the first 16 plies of our 78 rated games and which the book did not cover — each thought for 40 s by the v23 engine (`tools/build_book2.py`, `tools/merge_book2.py`); 858 entries in all. Only 21 of the 68 start positions seen on the ladder were in `data/start_fens.txt`, so 53 of 78 games had begun outside the old book | safe by construction (own engine, every move checked legal); book hit rate on the first 10 plies of our games 390/395 |

## Night of 9-10 Sep: last experiments

| candidate | measured |
|---|---|
| v23 net at 200 ms vs v22 | +19, +2 (+/- 56) — the +10 holds at depth |
| Network with 1.55M rows relabelled at 40k nodes (55% of the middlegame rows), cosine; hold194 0.007244 (best) | vs v23: -14, +14 (+/- 40) — level |
| Contempt 90 vs the weaker build | +61 (contempt 60: +78, 30: +38) — 60 kept |
| "Small edge, no conversion" positions from R87-R89 replayed at 2.5 s and 10 s | only 1 of 5 errors fixed by more time; the rest are evaluation-limited |

At the time these read level, v23 looked like the final build; v24 (the deep book) followed on 10 Sep.

## v23 — 9 Sep afternoon (tag `v23`)

| change | measured |
|---|---|
| Contempt 60 (was 30): a draw counts as -60 cp, so the engine plays on in level positions instead of repeating | vs the weaker v19 build from identical openings: contempt 60 **+78 +/- 41** (61.3%), contempt 30 +38 +/- 40 (55.3%) |
| Network: 6.95M rows with 803k middlegame rows relabelled by Stockfish at 40k nodes (was 15k), cosine LR schedule; hold194 0.007266 (v22 net 0.007308) | vs v22: +19, +13, -1, +12 (+/- 40 each; 1200 games pooled +10, -10..+30). Components alone: cosine +13/+6, relabel (345k rows) +7/-8 |

Ladder context: v22 scored = = L = = = = L L on 9 Sep against teams around our level (26th-27th); the losses (R77, R82, R83) were positional slides from the opening, all as Black in closed structures. Contempt is a deliberate variance choice for the last ladder day.

## v22 — 9 Sep morning (tag `v22`)

| change | measured |
|---|---|
| Network retrained on 6.95M positions (mac_all11: the 5.3M set plus the 7 Sep Mac deltas), same 256-wide recipe; hold194 0.007308 (was 0.007441) | on the v21 engine: +24, +16, +7, +20, +23, +31 (+/- 40 each, 1800 games pooled **+20, +4..+36**); +21, +28 (+/- 56) at 200 ms; miss test 44/76 vs 50/76 |

The same net measured -22 +/- 40 on the v19 engine on 7 Sep (300 games) and was rejected then; that sample sits inside the pooled interval.

Relabel experiment (9 Sep morning): 345k middlegame rows (6.5% of mac_all10) relabelled by Stockfish at 40k nodes instead of 15k (median shift 12 cp, mean 22, 7% of rows moved >= 50 cp); same net recipe: hold194 0.007442, +7 +/- 39 vs v21, -8 +/- 39 vs v22. Inconclusive at that coverage; the relabelling continues towards ~1M rows.

## v21 — 8 Sep evening (tag `v21`)

Speed only; the search is unchanged. Profiled by calling each kernel twice
per node and measuring the extra time: hand evaluation 283 ns, accumulator
update 266 ns, make 208 ns, network output 154 ns, move scoring 119 ns,
move generation 81 ns of a 1410 ns node. Changes, all verified to give the
same node counts and scores at fixed depth (except the last, which only
changes the order of equal-value captures):

| change | speed |
|---|---|
| material / piece-square / phase sums kept incrementally in make() (state slots 19-21) | +3% |
| king-zone attack counts gathered in the mobility loop instead of recomputing every attack set | +3% |
| accumulator update fused into one pass per perspective (was copy + 2-4 passes) | +5% |
| captures-only generator in quiescence (gen_captures; underpromoting captures kept for identical move sets) | +13% |

Total about +25% nodes per second. Tried and dropped: LLVM ctpop/cttz
intrinsics (LLVM already emitted POPCNT; no change), pawn-king term cache
(+1%, not worth a state field), forceinline on the magic lookups (nothing),
a smaller TT (nothing: memory latency is not the bottleneck).

Evening 8 Sep, measured with the new compressed-clock A/B (`tools.ab --clock --clock-scale 6`: 120+0.5 played as 20+0.083 with every allocator constant scaled, 300 games each vs v21): MOVES_LEFT_START 45 -30 +/- 40, 50 -7 +/- 39. The 60/30 spread stays; the clock is closed as a topic.
Correction history (pawn-structure eval correction fed back from search results): -2 +/- 39 @60 ms, 49.5% over 200 games @200 ms. Cut-node LMR +1: +7 +/- 39. 6.95M-row net on the v21 engine: +24, +16, +7, +20 (+/- 40 each, 1200 games pooled +17, -3..+37); at 200 ms +21, +28 (+/- 56); miss test 44/76 vs v21's 50/76. Middlegame-weighted net (16-28-piece rows counted twice; hold194 0.007612): -13, +17 (+/- 40) - level, dropped.

## v16-v20 — 7-8 Sep

| version | change | measured |
|---|---|---|
| v16 | 256-wide net on 5.3M positions | +12 (-12..+35) vs v15 net, 600 games @60 ms, rated openings |
| v17 | +35% node speed (fast-math net kernels, packed TT, one-pass move sort); check evasions in quiescence; score-drop time extension | +81 +/- 41 and +48 +/- 40 vs v16 (300 games each) |
| v18 | flat-position clock economy removed (it cut thinking in balanced middlegames: rounds 51, 53) | - |
| v19 | time management restored to v14's (v17's extension over-spent early: round 56 reached move 40 with 28 s) | +21 +/- 40 and +54 +/- 40 vs v14; real-clock games: 41-44 s at move 40 |
| v20 | own-engine opening book, 30 s a move, 311 positions from the 45 rated starts | safe by construction; not Elo-tested |

Rejected 7-8 Sep: continuation history (-15 at 400 ms), quiet checks in quiescence (-19), lazy SEE (no speed), captures-only generator (+10% speed, 0 Elo), 6.95M-row net (-22), futility 130 (-4), LMP raised (+22 / 600), unreduced checks (+27, -20).
Miss test (76 positions from our games where we lost >= 80 cp, 2 s each): v14 41/76, v19 48/76.
Rejected 8 Sep afternoon: 512-wide net (hold194 0.007500 vs 0.007441; -119 +/- 89 after 70 games, stopped), quiescence TT (+31, +9; -5 with LMP), own-game-upweighted net (-10, -53), pawn-structure correction history (-2 +/- 39 @60 ms; 200 ms run in `ab_corr/ab200*.log`), cut-node LMR +1 (+7 +/- 39). The engine sits at a plateau where nothing in the +-20 band can be told from noise with the cores available.

## v15 — 7 Sep morning (tag `v15`)

| change | measured |
|---|---|
| Network: hidden 256 (was 128), 30 epochs, on 3.88M positions (3.42M + 440k rated-opening rows + Closed Sicilian and random-ending rows); unseen-holdout loss 0.00782 vs 0.00815 for the v14 net | +7 +/- 44 (240 games @60 ms); node speed unchanged |
| Clock spread 70/35 moves (was 60/30): round 44 lost a held position on a depth-11 move with 8 s left at move 90; half the rated games run past move 70. The top-10 spend 3.3 s/move early (site data), so the early cut is small | game evidence; not A/B-testable at fixed time |
| Flat-position reserve: after ten straight moves within 20 cp past ply 60, 60% of the budget (round 42: 30 s to 2 s over sixty moves of a dead draw) | game evidence |
| Stray engine.py.orig dropped from the zip; nnue.py can load king-bucketed nets | tidy |

Rejected tonight (all 300-600 games @60 ms vs v14): continuation history +23 then -52 (pooled ~-10); trade-down bonus +19; 128-net retrained on 3.88M +13 then -13 (pooled ~0); the three combined +12.
Site-review ladder analysis (day 3, Stockfish-16 d16 verdicts): our endgame error rate is now at top-3 level; the remaining gap is the opening (inaccuracies 3.3 vs 1.4 per 100 moves) and middlegame mistakes (0.78 vs 0.29).

## v14 — 6 Sep evening (tag `v14`)

| change | measured |
|---|---|
| Network retrained on 3.42M positions with left-right mirror augmentation (holdout 0.00878 from 0.00929) | +24 +/- 40 vs v12 net @60 ms (300 games); +3 +/- 69 @400 ms (100 games) |

## v13 — 6 Sep afternoon (tag `v13`)

| change | measured |
|---|---|
| Rules changed: a game still running at 600 plies is a draw (was a material adjudication at 300). The search returns a draw at the cap and fades the evaluation toward zero over the last 80 plies; the old material-adjudication terms are gone | rules; round 31 ran past ply 300 under the old logic |

The 3.42M-row net (104k Closed Sicilian rows) measured level with v12's (+0 +/- 51 after 180 games) and was not shipped.

## v12 — 6 Sep midday (tag `v12`)

| change | measured |
|---|---|
| Network output gain 0.85 instead of 0.7 (re-tuned for the v10 net: 0.85 +37 +/- 40, 1.0 +20 +/- 54, 0.55 −29 +/- 61 vs 0.7) | +37 +/- 40 (300 games @60 ms) |

## v11 — 6 Sep morning (tag `v11`)

| change | measured |
|---|---|
| Clock spread over max(30, 60 − ply/2) moves plus 0.6 × increment (was 45/22 and 0.8): round 31 with v9 ran 162 moves and finished with 2 s, having spent 3–6 s a move for the first 30 moves | modelled: 3 s/move to move 30, 2 s at move 40, 1 s at move 60, 8 s left at move 80 in a 160-move game |

## v10 — 6 Sep morning (tag `v10`)

| change | measured |
|---|---|
| Network retrained on 2.91M positions (1.1M endgame-weighted, 294k from the rated openings), with per-piece-count output buckets and pure engine labels | **+50 +/- 40** vs v9 (300 games @60 ms) |

Recipe experiments overnight: buckets alone +5/+17 +/- 40; pure engine labels +39 +/- 50; null-move eval reduction -26; history pruning -9; LMR/2.0 +3 (all rejected).

## v9 — 5 Sep night (tag `v9`)

| change | measured |
|---|---|
| Iterative deepening keeps the finished part of a cut-off iteration (moves whose score beat the pass's alpha), so an iteration may start whenever the soft budget is unspent; clock spread over max(22, 45 − ply/2) moves, hard ceiling 2.5× soft. Real-game spend rises from ~40% of the budget to ~130% | +53 +/- 167 (20 games at the real 120 s + 0.5 s control); v8 given 1.6× time beats v8 by +92 +/- 50 (200 games), which is the gain this policy is meant to collect |
| Singular extensions: when the hash move beats every alternative by 2 cp/ply at half depth, search it a ply deeper | +45 +/- 70 @400 ms (100 games), +7 +/- 40 @60 ms |

Rejected on the way: three time policies tested at 1/5 scale (24 s + 0.1 s) all measured slightly negative; the scaled clock exaggerates per-move overhead and the first version had a bound bug (it could switch to a move whose score was only an upper bound). The v9 net candidate (1.91M rows incl. openings, seeds 1+3) was -27 +/- 40 vs the v8 net despite a better holdout.

## v8 — 5 Sep evening (tag `v8`)

| change | measured |
|---|---|
| Network output damped to 0.7 (the residual net was over-confident; 0.8: +44 +/- 40, 0.65: +34 +/- 40 over v7, 300 games each @60 ms; 1.3: -112 +/- 82) | ~+40 |
| Clock spread over 40 moves (floor 20) instead of 50 (floor 22): round 25 was lost with 55 s unused in a 47-move game | ~4.4 s a move in the middlegame instead of 2.8 |

Shelved: singular extensions, +7 +/- 40 (data/singular_ext.patch).

## v7 — 5 Sep evening (tag `v7`)

| change | measured |
|---|---|
| Network retrained on 1.56M positions, ~250k of them from the endgame-weighted sampler | +38 +/- 40 vs v6 net (300 games @60 ms); +117 +/- 120 @400 ms (40 games) |

## v6 — 5 Sep afternoon (tag `v6`)

| change | measured |
|---|---|
| Time allocation by moves-to-go (clock spread over max(22, 50 − ply/2) moves + 0.8×increment) instead of a fixed fraction: about 3 s/move through the middlegame instead of 5 s early and 1 s from move 40 | +35 +/- 90 (60 clocked games at 24 s + 0.1 s) |
| Network retrained on 1.18M positions with a clean, contiguous holdout (earlier nets were over-trained by a leaky holdout) | +31 +/- 40 vs v5 net (300 games @60 ms) |
| Node-cap speed estimate floored at 150k nps so a wall-clock hiccup cannot shrink the next search | safety |
| **v6 vs v4 directly** | **+94 +/- 41** (300 games @60 ms), before the time-allocation gain |

Rejected on the way: opposite-coloured-bishop halving + unstoppable-passer rule (together -49 +/- 57 after 150 games; passer alone -2 +/- 48; bishops alone -7 +/- 48).

## v5 — 5 Sep midday (tag `v5`)

| change | measured |
|---|---|
| Network retrained on 876k laptop-labelled positions (from 740k) | +33 +/- 40 vs v4 net (300 games @60 ms) |
| Countermove ordering, history-aware late-move reductions, internal iterative reduction | +23 +/- 40 (300 games @60 ms) |
| Drawn-material scaling applied to the whole evaluation, network included | included above |
| All three together vs v4 | +14 +/- 40 (300 games @60 ms) |

## v4 — 5 Sep (tag `v4`)

| change | measured |
|---|---|
| Neural evaluation: a residual 2-net ensemble (hidden 256) trained on 740k positions labelled on the team's laptop, correcting the hand evaluation; incremental accumulators keep the search at ~95% of its old speed | **+40 +/- 28** (600 games @60 ms vs v3) |
| Pondering off: the frozen rules suspend the process while the opponent thinks | rules |
| Weights ship as `.safetensors` read with numpy alone; `training/` documents the run for the audit | compliance |

## Rejected

| change | measured | note |
|---|---|---|
| King-safety weight x2 | -6 +/- 66 | |
| Hand-set pawn shelter and openness terms | -61 +/- 67 | superseded by the data-tuned terms below |
| Do not prune or reduce checking moves | -57 +/- 55 | standard elsewhere, but here the extra work outweighed the mates found |

## In progress

Data-driven evaluation: pawn shelter, pawn storm, pawn threats, rook on
open/semi-open files, king distance to passed pawns and per-piece material
corrections, all fitted by Texel tuning on Stockfish-labelled self-play
positions (`tools/pipeline.py`, `tools/tune.py`). First fit on 33k positions:
+26 +/- 40. Refit on the overnight data before it ships.

## What the rated games said

Stockfish at 300k nodes per move, our moves only:

- Round 3 (draw): fair; our 39...h5 (-204) was not punished.
- Round 4 (loss, Black): Greek gift 10.Bxh7+; the evaluation liked ...O-O by +40
  where the reference had -237. King attack under-valued.
- Round 6 (loss, Black): no blunder, a 30-50 cp slide per move under a pawn
  storm from move 23 to 38. Same weakness.
- Round 7 (win, White, lucky): 21.Be3 / 23.d5 / 25.c4 turned +105 into -88;
  objectively lost from move 38 to 60; the opponent released it.
- Round 8 (loss, Black, v2): 19...b5 (-232; b6 held), then 43...Rb2 into a
  mate in seven the search only finds at depth 20.
