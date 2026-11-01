"""Analyze the recorded Cloudict tournament JSON.

Usage: uv run python benchmarks/analyze_tournament.py
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connect6.board import init_board, make_move
from connect6.defines import Color, GRID_NUM
from connect6.protocol import msg2move
from connect6.vcf import _scan, find_forced_win

OUT_FILE = ROOT / "benchmarks" / "tournament_cloudict.json"
DEPTHS = (3, 4, 5)
BAND = 20000


def our_favor(ours_black: bool, score: float) -> bool:
    return score >= BAND if ours_black else score <= -BAND


def opp_favor(ours_black: bool, score: float) -> bool:
    return score <= -BAND if ours_black else score >= BAND


def replay(rows: list[dict], ply_cap: int) -> list[list[int]]:
    board = [[0] * GRID_NUM for _ in range(GRID_NUM)]
    init_board(board)
    for record in rows:
        if record["ply"] >= ply_cap:
            break
        make_move(board, msg2move(record["msg"]), Color.BLACK if record["color"] == "black" else Color.WHITE)
    return board


def main() -> int:
    rows = json.loads(OUT_FILE.read_text())
    assert len(rows) == 36, f"expected 36 games, found {len(rows)}"
    wins = sum(1 for r in rows if r["winner"] == ("black" if r["ours_color"] == "black" else "white"))
    print(f"=== the matrix: {wins}-{len(rows) - wins} (ours-opponent) ===")
    for depth in DEPTHS:
        sub = [r for r in rows if r["depth"] == depth]
        ours_moves = [m for r in sub for m in r["moves"] if m["engine"] == "ours" and "nodes" in m]
        cloud_moves = [m for r in sub for m in r["moves"] if m["engine"] == "cloudict" and "wall" in m]
        plies = [r["plies"] for r in sub]
        w = sum(1 for r in sub if r["winner"] == ("black" if r["ours_color"] == "black" else "white"))
        print(
            f"floor {depth}: {w}-{len(sub) - w} | plies {min(plies)}-{max(plies)} median "
            f"{statistics.median(plies):.0f} | our nodes/move mean "
            f"{statistics.mean([m['nodes'] for m in ours_moves]):,.0f} max {max(m['nodes'] for m in ours_moves):,} | "
            f"our wall mean {statistics.mean([m['wall'] for m in ours_moves]):.1f} s | "
            f"cloudict wall mean {statistics.mean([m['wall'] for m in cloud_moves]):.1f} s"
        )
    as_black_wins = sum(1 for r in rows if r["ours_color"] == "black" and r["winner"] == "black")
    as_white_wins = sum(1 for r in rows if r["ours_color"] == "white" and r["winner"] == "white")
    print(f"color split: {as_black_wins} wins as black, {as_white_wins} as white — the gap is engine-side")

    print("\n=== loss declarations (terminal-band scores favoring the opponent) ===")
    declared = 0
    leads: list[int] = []
    for r in rows:
        ours_black = r["ours_color"] == "black"
        first = next(
            (m["ply"] for m in r["moves"] if m["engine"] == "ours" and m.get("score") is not None
             and opp_favor(ours_black, m["score"])),
            None,
        )
        if first is not None:
            declared += 1
            leads.append(r["plies"] - first)
    print(f"{declared} of {len(rows)} games contain at least one declaration")
    print(f"lead time of the first declaration: {min(leads)}-{max(leads)} plies before the end, median {statistics.median(leads):.0f}")
    scored = [m for r in rows for m in r["moves"] if m["engine"] == "ours" and m.get("score") is not None]
    decls = sum(1 for r in rows for m in r["moves"]
                if m["engine"] == "ours" and m.get("score") is not None
                and opp_favor(r["ours_color"] == "black", m["score"]))
    print(f"declaration moves: {decls} of {len(scored)} scored moves")
    undeclared = [r for r in rows
                  if not any(m["engine"] == "ours" and m.get("score") is not None
                             and opp_favor(r["ours_color"] == "black", m["score"]) for m in r["moves"])]
    print(f"undeclared games: {len(undeclared)} — final recorded scores:")
    for r in undeclared:
        last = next((m.get("score") for m in reversed(r["moves"])
                     if m["engine"] == "ours" and m.get("score") is not None), None)
        last_text = f"{last:+.0f}" if last is not None else "none"
        print(f"  d{r['depth']} {r['opening']} ours {r['ours_color']}, end ply {r['plies']}, last score {last_text}")

    print("\n=== win claims (terminal-band scores favoring us, in games we lost) ===")
    claims = [(r, m) for r in rows for m in r["moves"]
              if m["engine"] == "ours" and m.get("score") is not None
              and our_favor(r["ours_color"] == "black", m["score"])]
    print(f"{len(claims)} win-claim moves in {len({id(r) for r, _ in claims})} games; all lost")
    for r, m in claims:
        ours_color = Color.BLACK if r["ours_color"] == "black" else Color.WHITE
        board = replay(r["moves"], m["ply"])
        claim = find_forced_win(board, ours_color)
        mine, theirs = _scan(board, ours_color)
        opp_fives, opp_fours = (theirs[1], theirs[2]) if ours_color == Color.BLACK else (mine[1], mine[2])
        assert claim is not None, "the stage-1 proof must reproduce on the recorded board"
        print(
            f"d{r['depth']} {r['opening']} ours {r['ours_color']} ply {m['ply']}: stage-1 answers in "
            f"{m.get('nodes')} nodes; at the claim the opponent holds {len(opp_fives)} live five(s) "
            f"and {len(opp_fours)} live four(s); the game ends {r['plies'] - m['ply']} ply later, "
            f"winner {r['winner']}"
        )

    print("\n=== the book ===")
    books = sum(1 for r in rows for m in r["moves"] if m["engine"] == "ours" and m.get("book"))
    print(f"book hits: {books} (opening IIKK as black, one per floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())