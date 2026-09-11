# Abhi's Chess Demon — an AI Chessathon engine in pure Python

A chess engine written for the AI Chessathon ladder (September 2026), where every
agent runs as plain Python on one CPU core with 120 s + 0.5 s per game, no native
code, and a 90-second import budget. The whole engine — bitboard move generation,
alpha-beta search and an NNUE-style evaluation — is Python compiled with numba at
import time. Twenty-four builds over nine days, twenty-one of them
uploaded and validated; every change was measured before it shipped, and the ones
that failed are recorded alongside the ones that worked. Final standing: **47th of
465 teams (top 10%)**, rating 2319, peak 2372, 41W 39D 33L.

## The engine (`nbchess/`)

- **Board**: magic bitboards, copy-make, Zobrist hashing, incremental material and
  piece-square sums (`core.py`).
- **Search** (`search.py`): principal-variation search with a packed transposition
  table, aspiration windows, null-move pruning, late-move reductions and pruning,
  reverse and forward futility, static-exchange pruning, singular extensions,
  killer/counter/history ordering, check extensions and a quiescence search with
  check evasions and a captures-only generator.
- **Evaluation**: a residual NNUE — 768 piece-square inputs per perspective, 256
  hidden units, three output buckets by piece count — trained on ~7M positions
  labelled by Stockfish, added on top of a hand-tuned tapered evaluation (mobility,
  pawn structure, king safety, threats). Accumulators are updated incrementally in
  one fused pass per move (`nnue.py`).
- **Clock** (`engine.py`): the remaining time is spread over `max(30, 60 − ply/2)`
  moves plus 60% of the increment, with a hard ceiling of half the clock.
- **Opening book** (`book.json`): 858 positions — every start and opening line
  seen in our rated games — each answered by the engine after 30–40 s of thought.
- **Contempt**: a draw is scored as −60 cp, so the engine plays on in level positions.

Node speed is roughly 600–700k nodes/second on one core; a middlegame move
reaches about depth 10–12 in the 2–3 seconds the clock allows.

## How changes were decided

Nothing shipped without beating the current build in a measured match:
`tools/ab.py` plays two builds against each other from the rated openings with both
colours, 300–1800 games at 60 ms/move, and reports Elo with a 95% interval.
Time-management changes were measured on a compressed real clock
(`--clock --clock-scale 6`), and every build got real-clock games, a compliance
check against the platform contract and a full harness game before upload.

`docs/VERSIONS.md` is the complete record: what each build changed, what it
measured, and the fifty-odd ideas that were rejected on the numbers — including
several that were shipped early on, lost games on the ladder, and were reverted.

Highlights, all vs the previous build unless stated:

| build | what | measured |
|---|---|---|
| v2 | static exchange evaluation | +75 ± 40 |
| v3 | pondering on the opponent's clock (allowed at the time) | +109 ± 51 |
| v10–v12 | first NNUE, output gain tuned | +37 ± 40 |
| v14 | 3.4M-position net with mirror augmentation | +24 ± 40 |
| v17 | +35% node speed (fast-math kernels, packed TT, one-pass sort) | +81 / +48 ± 40 |
| v19 | clock scheme restored after two failed variants | +21 / +54 ± 40 |
| v21 | +25% node speed, bit-identical search (four hot-path rewrites) | +33 / +45 ± 40 |
| v22 | 6.95M-position net | +20 (+4..+36), 1800 games |
| v23 | contempt 60; net with 800k rows relabelled at deeper Stockfish search | +78 vs +38 against a weaker build; net +10 |
| v24 | deep book: 858 positions from our own rated games, 40 s each | safe by construction |

Things that did not help, each measured: continuation history, correction
history, cut-node LMR, quiescence TT, king-bucketed inputs, a 512-wide net, a
non-residual net, LLVM popcount intrinsics, TT prefetch, int16 accumulators,
middlegame-weighted training, a cosine learning-rate schedule on its own, and
spending the clock faster early.

## Training (`tools/`, `training/`)

Positions came from Stockfish self-play pipelines run on a laptop
(`tools/pipeline.py` and successors), labelled at 15k nodes, later relabelled at
40k nodes for 1.5M middlegame rows. `tools/nnue_train_l10.py` trains the net with
a preallocated int16 feature loader, in-place mirror augmentation and a contiguous
holdout of whole games; `tools/nnue_ensemble.py` writes the weights in a
self-describing safetensors layout that `nnue.py` reads with numpy alone.

## Running it

```
make setup        # uv sync
make play         # one real-clock game against a baseline
uv run python -m tools.ab <dir_a> <dir_b> --games 300 --ms 60 --book data/start_fens.txt
make zip          # build the submission
```
