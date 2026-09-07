# Ladder analysis after day 3 (6 Sep, 45 rounds)

Source: the site's own game reviews (Stockfish 16, depth 16) for all 833
games of the top 30 teams, scraped from the public game pages, plus the
rating history on each team page. Raw tables: `ladder_day3_analysis.txt`.

## Standing

11th of 338 at 2238 (19-8-13). Day-3 gain +327, the largest in the top 11
after test_bot. The top three (AlphaFish 2644, Anchoa 2506, test_bot 2430)
are separating from 4th-10th (2240-2426).

## Accuracy

Day-3 average centipawn loss per move: us 15.9; top ten 13.1-19.9. Blunders
per game 0.2 (top ten 0.0-0.3). Day 1 we were at 23.8 and day 2 at 33.9, so
the rating still carries the early losses: all-time we are 1-2-7 against
the top ten, but six of the seven losses came before round 31.

## Where the errors are (day 3, per 100 own moves, us / top three)

| phase | inaccuracies | mistakes | blunders |
|---|---|---|---|
| opening (first 8 moves) | 3.3 / 1.4 | 0.8 / 0.0 | 0.8 / 0.0 |
| middlegame | 3.1 / 2.0 | 0.8 / 0.3 | 0.2 / 0.0 |
| endgame | 1.3 / 1.3 | 0.1 / 0.2 | 0.1 / 0.2 |

The endgame, which lost rounds 21, 28, 30 and 38, is now at the level of the
top three. What remains is judgement in the opening and middlegame:
rounds 32 and 44 were lost to runs of positional mistakes in Closed
Sicilian structures, not to tactics.

## Clocks

The top ten spend 3.3 s a move over the first twenty moves (we spent 2.9)
and have a median 14.6 s left after move 80 (we had 11.2). Round 44 was
lost on a depth-11 move with 8 s left at move 90, so v15 spreads the clock
over 70 moves falling to 35 (was 60/30) and spends less in dead-level
positions.

## Void games

Games that end without a result are common for everyone (Anchoa 10,
AI Fellow 10, AlphaFish 5, us 5): a platform matter, not an agent bug.
