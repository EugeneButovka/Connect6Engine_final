"""The verdict tournament: v4, v5 and v6 games started from the trajectory
prefixes where the engines differ — the verdict-flip prefixes (played by
every pairing) and the v5-vs-v6 budget-boundary prefixes.

`probe` first measures each engine's verdict on the flip starts at floors
3/4/5; every game is recorded move by move to
benchmarks/tournament_verdicts.json.

Usage:
  uv run python benchmarks/tournament_verdicts.py probe
  uv run python benchmarks/tournament_verdicts.py play --floor 4 --jobs 12
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

ROOT = Path(__file__).resolve().parent.parent
os.environ["V6_JOBS"] = "1"
os.environ["WEIGHTS_FILE"] = str(ROOT / "tuning" / "default.json")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tuning"))

import genetic_tuner as gt
import harness
from connect6.board import check_game_end, make_move
from connect6.defines import (
    MAXINT,
    MININT,
    Board,
    Color,
    GameResult,
    Move,
    SearchStats,
    SearchResult,
)
from connect6.opening_book import book_move, load_opening_book
from connect6.parallel import search_v6
from connect6.protocol import move2msg
from connect6.search import search_alpha_beta_with_quiescence
from connect6.vcf import search_v5

SearchFn = Callable[[Board, Color, int, Move, SearchStats], SearchResult]

FLIP_STARTS = (36, 42, 44, 45, 46, 47)
BOUND_STARTS = (3, 19, 21, 34, 37)
FLIP_PAIRINGS = (("v4", "v5"), ("v4", "v6"), ("v5", "v6"))
BOUND_PAIRINGS = (("v5", "v6"),)
OUT_FILE = ROOT / "benchmarks" / "tournament_verdicts.json"
TRAJECTORY_FILE = ROOT / "benchmarks" / "trajectory.json"

HISTORY: list[str] = json.loads(TRAJECTORY_FILE.read_text())["history"]
CHAMPION = harness.load_weights(ROOT / "tuning" / "winner.json")
BOOK = load_opening_book()


@dataclass(frozen=True, slots=True)
class Engine:
    name: str
    search: SearchFn
    book: bool


ENGINES: dict[str, Engine] = {
    "v4": Engine("v4 + quiescence, pruning, book", search_alpha_beta_with_quiescence, True),
    "v5": Engine("v5 + forced-win layer", search_v5, True),
    "v6": Engine("v6 + parallel split, budget", search_v6, True),
}


def start_position(prefix: int) -> tuple[Board, Move, Color]:
    board, pre_move, mover = harness.replay_history(HISTORY[:prefix])
    return board, pre_move, mover


def verdict(score: int, mover: Color) -> str:
    if (mover is Color.BLACK and score >= MAXINT) or (mover is Color.WHITE and score <= MININT):
        return "mover-win"
    if (mover is Color.BLACK and score <= MININT) or (mover is Color.WHITE and score >= MAXINT):
        return "mover-loss"
    return "quiet"


def probe(floors: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prefix in FLIP_STARTS:
        board, pre_move, mover = start_position(prefix)
        mover_name = "black" if mover is Color.BLACK else "white"
        for floor in floors:
            for key, engine in ENGINES.items():
                gt.apply_weights(CHAMPION)
                stats = SearchStats()
                start = time.perf_counter()
                result = engine.search(board, mover, floor, pre_move, stats)
                ms = (time.perf_counter() - start) * 1000
                rows.append(
                    {
                        "start": prefix,
                        "mover": mover_name,
                        "floor": floor,
                        "engine": key,
                        "move": move2msg(result.move),
                        "score": result.score,
                        "verdict": verdict(result.score, mover),
                        "nodes": stats.node_count,
                        "ms": round(ms, 1),
                    }
                )
                print(
                    f"start {prefix:2d} ({mover_name} to move) floor {floor}: "
                    f"{key} answers {move2msg(result.move)} at {result.score} "
                    f"({verdict(result.score, mover)}) in {stats.node_count:,} nodes, {ms:,.0f} ms",
                    flush=True,
                )
    return rows


def play(
    start: int,
    mover_engine: str,
    opponent_engine: str,
    floor: int,
    cap: int,
) -> dict[str, object]:
    board, pre_move, mover = start_position(start)
    mover_color = "black" if mover is Color.BLACK else "white"
    players = {
        mover: ENGINES[mover_engine],
        (Color.WHITE if mover is Color.BLACK else Color.BLACK): ENGINES[opponent_engine],
    }
    moves: list[dict[str, object]] = []
    tally: dict[str, dict[str, float]] = {
        key: {"nodes": 0, "ms": 0.0, "book_hits": 0} for key in ("v4", "v5", "v6")
    }
    winner: str | None = None
    plies = 0
    for ply in range(cap):
        engine = players[mover]
        key = next(k for k, e in ENGINES.items() if e is engine)
        stats = SearchStats()
        score: int | None = None
        start_clock = time.perf_counter()
        booked = book_move(board, BOOK) if engine.book else None
        if booked is not None:
            move = booked
        else:
            gt.apply_weights(CHAMPION)
            result = engine.search(board, mover, floor, pre_move, stats)
            move = result.move
            score = result.score
        ms = (time.perf_counter() - start_clock) * 1000
        make_move(board, move, mover)
        moves.append(
            {
                "ply": ply + 1,
                "color": "black" if mover is Color.BLACK else "white",
                "engine": key,
                "msg": move2msg(move),
                "nodes": stats.node_count,
                "ms": round(ms, 1),
                "score": score,
            }
        )
        tally[key]["nodes"] += stats.node_count
        tally[key]["ms"] += ms
        tally[key]["book_hits"] += 1 if stats.node_count == 0 else 0
        plies += 1
        end = check_game_end(board, move)
        if end is not None:
            winner = key if end is GameResult.WIN else None
            break
        pre_move = move
        mover = Color.WHITE if mover is Color.BLACK else Color.BLACK
    return {
        "start": start,
        "mover_color": mover_color,
        "mover_engine": mover_engine,
        "opponent_engine": opponent_engine,
        "floor": floor,
        "winner": winner,
        "plies_from_start": plies,
        "tally": tally,
        "moves": moves,
    }


def play_job(job: dict[str, int | str]) -> dict[str, object]:
    return play(
        cast(int, job["start"]),
        cast(str, job["mover_engine"]),
        cast(str, job["opponent_engine"]),
        cast(int, job["floor"]),
        cast(int, job["cap"]),
    )


def jobs_for(floor: int, cap: int) -> list[dict[str, int | str]]:
    jobs: list[dict[str, int | str]] = []
    for prefix in FLIP_STARTS:
        for left, right in FLIP_PAIRINGS:
            jobs.append({"start": prefix, "mover_engine": left, "opponent_engine": right, "floor": floor, "cap": cap})
            jobs.append({"start": prefix, "mover_engine": right, "opponent_engine": left, "floor": floor, "cap": cap})
    for prefix in BOUND_STARTS:
        for left, right in BOUND_PAIRINGS:
            jobs.append({"start": prefix, "mover_engine": left, "opponent_engine": right, "floor": floor, "cap": cap})
            jobs.append({"start": prefix, "mover_engine": right, "opponent_engine": left, "floor": floor, "cap": cap})
    return jobs


def load_record() -> dict[str, object]:
    if OUT_FILE.is_file():
        record: dict[str, object] = json.loads(OUT_FILE.read_text())
        return record
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="The verdict tournament: v4/v5/v6 from the positions where they differ.")
    parser.add_argument("command", choices=("probe", "play"))
    parser.add_argument("--floor", type=int, default=4)
    parser.add_argument("--cap", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    record = load_record()
    if args.command == "probe":
        record["probe"] = probe((3, 4, 5))
        OUT_FILE.write_text(json.dumps(record, indent=1))
        print(f"probe written to {OUT_FILE}")
        return 0

    jobs = jobs_for(args.floor, args.cap)
    games: list[dict[str, object]] = []
    started = time.perf_counter()
    if args.jobs <= 1:
        for number, job in enumerate(jobs, 1):
            game = play_job(job)
            games.append(game)
            print(
                f"game {number:2d}/{len(jobs)}: start {cast(int, game['start']):2d} "
                f"{cast(str, game['mover_engine'])} (mover) vs {cast(str, game['opponent_engine'])} — "
                f"winner: {cast('str', game['winner']) if game['winner'] else 'none (cap)'} "
                f"in {cast(int, game['plies_from_start'])} plies",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(play_job, job) for job in jobs]
            for number, future in enumerate(as_completed(futures), 1):
                game = future.result()
                games.append(game)
                print(
                    f"game {number:2d}/{len(futures)} done: start {cast(int, game['start']):2d} "
                    f"{cast(str, game['mover_engine'])} vs {cast(str, game['opponent_engine'])} — "
                    f"winner: {cast('str', game['winner']) if game['winner'] else 'none (cap)'} "
                    f"in {cast(int, game['plies_from_start'])} plies",
                    flush=True,
                )
    games.sort(key=lambda g: (cast(int, g["start"]), str(g["mover_engine"]), str(g["opponent_engine"])))
    record["configuration"] = {
        "floor": args.floor,
        "cap_from_start": args.cap,
        "v6_jobs": 1,
        "weights": "tuning/winner.json installed per search",
        "book": "v4/v5/v6 consult the opening book first",
        "starts": {
            "flips": list(FLIP_STARTS),
            "bounds": list(BOUND_STARTS),
        },
        "pairings": {
            "flips": [list(p) for p in FLIP_PAIRINGS],
            "bounds": [list(p) for p in BOUND_PAIRINGS],
        },
    }
    record["games"] = games
    OUT_FILE.write_text(json.dumps(record, indent=1))
    print(f"{len(games)} games in {time.perf_counter() - started:,.0f} s — written to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())