"""Self-generated opening book: our own engine at long think on the rated
start positions. For each start FEN (either side to move): the best move at
BOOK_MS; then for the top-K replies to it (by the engine's own ranking at
REPLY_MS) our answer at BOOK_MS; and, for the case where we move second,
our answer to each of the opponent's top-K first moves.
Writes book_<shard>.json as {fen: uci}; merge the shards into book.json keyed
by the first four FEN fields (see agent.py)."""
import sys, json, time, chess
from nbchess.engine import Engine
shard, nshards = int(sys.argv[1]), int(sys.argv[2])
BOOK_MS, REPLY_MS, K = 30000.0, 8000.0, 3
fens = [l.strip() for l in open("data/start_fens.txt") if l.strip()]
fens = [f for i, f in enumerate(fens) if i % nshards == shard]
e = Engine()
book = {}
def best(fen, ms, ply):
    e.tt[:] = 0
    r = e.think(fen, ms, ms * 1.5, game_ply=ply)
    return r
t0 = time.time()
for f in fens:
    b0 = chess.Board(f)
    # we move first from this start
    r = best(f, BOOK_MS, 0); m0 = r[0][0]; book[b0.fen()] = m0
    b1 = b0.copy(); b1.push_uci(m0)
    replies = best(b1.fen(), REPLY_MS, 1)[:K]
    for rep, _ in replies:
        b2 = b1.copy(); b2.push_uci(rep)
        if b2.is_game_over(): continue
        book[b2.fen()] = best(b2.fen(), BOOK_MS, 2)[0][0]
    # we move second from this start: opponent's likely first moves
    firsts = best(f, REPLY_MS, 0)[:K]
    for mv, _ in firsts:
        b1 = b0.copy(); b1.push_uci(mv)
        if b1.is_game_over(): continue
        book[b1.fen()] = best(b1.fen(), BOOK_MS, 1)[0][0]
    json.dump(book, open(f"book_{shard}.json", "w"), indent=0)
    print(f"{len(book)} entries after {time.time()-t0:.0f}s", flush=True)
print("done", len(book))
