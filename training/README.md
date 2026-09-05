# How the shipped network was trained

Everything the engine ships in `nbchess/nnue.safetensors` was produced by the
code in this repository from data this team generated. Nothing was started
from a published chess network. This page exists so the training can be
demonstrated at the fair-play audit.

## Data

`tools/pipeline.py` plays reference-engine self-play games from randomised
openings on the team's own laptop (three processes, `--play-nodes 8000
--label-nodes 40000 --per-game 24`), samples up to 24 quiet positions per
game and labels each with a 40,000-node search. Rows are
`fen|result|engine_cp`. Using an existing engine to *label training data* is
explicitly permitted by the rules; the engine itself never ships.

The data files (`nn_mac1.txt`, `nn_mac2.txt`, `nn_mac3.txt`) live on the
laptop, not in the repository; they are tens of megabytes and can be shown
on request.

## Network

`tools/nnue_train.py`: 768 piece-square features per perspective (own pieces
first, board flipped for Black), one shared hidden layer of 128 with clipped
ReLU, linear output over the side-to-move accumulator followed by the other
side's. Trained in *residual* mode: the target is the label minus the hand
evaluation, so the net corrects `evaluate_w` rather than replacing it.
Loss: sigmoid (Texel K = 0.9) with a 0.7 / 0.3 blend of engine label and
game result; AdamW, weight decay 1e-4, batch 8192, lr 2e-3, 20 epochs, best
holdout epoch kept. Holdout is the last 5% of rows.

## Runs

### v4 net (5 Sep morning)

Three seeds on the same 740,151 positions, logs in this directory:

| run | seed | holdout loss | file |
|---|---|---|---|
| `train_e1.log` | 1 | 0.013987 | `nnue_e1.npz` |
| `train_e2.log` | 2 | 0.013776 | `nnue_e2.npz` |
| `train_e3.log` | 3 | 0.013922 | `nnue_e3.npz` |

Hand evaluation alone scores 0.0217 on the same holdout.

`tools/nnue_ensemble.py` merges `nnue_e1` and `nnue_e2` into one net of
hidden size 256 (the average of two clipped-ReLU nets is exactly one wider
net with halved output weights) and writes it as `.safetensors`, which
`nbchess/nnue.py` reads with numpy alone.

Match result at 60 ms/move, 600 games, both colours from the same openings:
**+40 +/- 28 Elo** over the v3 engine without the net; +42 +/- 70 at 400 ms.

### v5 net (5 Sep midday)

Same recipe on 876,397 positions (the three laptop files at 08:45 UTC),
seeds 1 and 2 (`train_r2_s1.log`, `train_r2_s2.log`; holdout 0.013818 and
0.013905), ensembled the same way. On a fresh 43,820-position holdout the
pair scores 0.00970 against 0.01669 for the v4 pair (hand evaluation
0.02093). Match result: **+33 +/- 40** over the v4 net, 300 games at 60 ms.

### v6 net (5 Sep afternoon) - the shipped net

The trainer's holdout was a random 5% of rows; with ~24 positions per game
that put siblings of every holdout position in the training set, so the
holdout flattered the net and "best epoch" picked an overfitted one. The
holdout is now the last 5% of rows as a contiguous block (whole games). On
that clean holdout the best epoch is 5 of 20 and the loss curve rises after
it, so every earlier net had been trained too long.

Same recipe on 1,181,564 positions (files at 12:10 UTC), seeds 1 and 2
(`train_r4_s1.log`, `train_r4_s2.log`; clean holdout 0.016743 and 0.016739),
ensembled: pair 0.016178 on the same 59,079-row holdout, against 0.016827
for the v5 pair and 0.017277 for the v4 pair (hand evaluation 0.020343).
Match result: **+31 +/- 40** over the v5 net, 300 games at 60 ms.
