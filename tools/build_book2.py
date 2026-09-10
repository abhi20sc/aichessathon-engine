"""Deepen the book from positions seen in our rated games: `fens2.json`
holds the positions (ordered by ply) where we were to move in the first
16 plies and which the book does not cover. Each gets a BOOK_MS think by
the current engine; results are written after every position so a partial
run is still usable. `python build_book2.py <shard> <nshards> <ms>`."""
import sys, json, time, chess
sys.path.insert(0, "/home/claude/abV23")
from nbchess_b.engine import Engine
shard, nshards, ms = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
fens = json.load(open("/home/claude/book/fens2.json"))
fens = [f for i, f in enumerate(fens) if i % nshards == shard]
out = f"/home/claude/book/book2_{shard}.json"
try:
    book = json.load(open(out))
except FileNotFoundError:
    book = {}
e = Engine()
t0 = time.time()
for f in fens:
    b = chess.Board(f)
    if b.fen() in book or b.is_game_over():
        continue
    e.tt[:] = 0
    r = e.think(f, ms, ms * 1.5, game_ply=0)
    mv = r[0][0]
    if chess.Move.from_uci(mv) in b.legal_moves:
        book[b.fen()] = mv
    json.dump(book, open(out, "w"), indent=0)
    print(f"{len(book)} done  {time.time()-t0:.0f}s", flush=True)
print("finished", len(book))
