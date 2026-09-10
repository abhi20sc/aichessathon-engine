"""Merge the deep-book shards into the repo's book.json (keys: first four
FEN fields) after checking every move is legal in its position."""
import json, glob, chess
book = json.load(open("/home/claude/starter/book.json"))
before = len(book)
added = 0; bad = 0
for p in sorted(glob.glob("/home/claude/book/book2_*.json")):
    for fen, uci in json.load(open(p)).items():
        b = chess.Board(fen)
        if chess.Move.from_uci(uci) not in b.legal_moves:
            bad += 1; continue
        key = " ".join(fen.split()[:4])
        book[key] = uci; added += 1
# final legality pass over the whole book
for key, uci in book.items():
    b = chess.Board(key + " 0 1")
    assert chess.Move.from_uci(uci) in b.legal_moves, key
json.dump(book, open("/home/claude/starter/book.json", "w"), indent=0)
print(f"book: {before} -> {len(book)} entries ({added} added, {bad} illegal skipped)")
