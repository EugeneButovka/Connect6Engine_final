"""The external tournament: the compiled engine against Cloudict.

Six openings, each played with our engine as black and as white, at three
depths (3, 4, 5). Every game is recorded move by move to
benchmarks/tournament_cloudict.json.

Usage: uv run python benchmarks/tournament_cloudict.py
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connect6.board import check_game_end, init_board, make_move
from connect6.defines import GRID_NUM, Color, GameResult
from connect6.protocol import msg2move

CLOUDICT_BIN = ROOT / "benchmarks" / "cloudict" / "gameEngine"
PATTERNS = ROOT / "benchmarks" / "cloudict" / "patterns.in"
OUR_BIN = ROOT / "dist" / "Connect6_Butovka_Ahmed"
OUT_FILE = ROOT / "benchmarks" / "tournament_cloudict.json"
DEPTHS = (3, 4, 5)
OPENINGS = ("IIKK", "JKLJ", "KJKI", "KLJI", "JLIJ", "JHLH")
DEADLINE_S = 600.0
MAX_PLIES = 60

MOVE_RE = re.compile(r"^move ([A-S]{2,4})\s*$")
NODE_RE = re.compile(r"^Node\s*:\s*(\d+)")
SCORE_RE = re.compile(r"^Score\s*:\s*(-?[\d.]+)")


class Engine:
    err: IO[str] | None

    def __init__(self, proc: subprocess.Popen[str], tag: str) -> None:
        assert proc.stdin is not None
        assert proc.stdout is not None
        self.proc = proc
        self.stdin = proc.stdin
        self.stdout = proc.stdout
        self.tag = tag
        self.err = None
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.cmd("name")

    def _reader(self) -> None:
        for line in self.stdout:
            self.lines.put(line.rstrip("\n"))

    def cmd(self, line: str) -> None:
        self.stdin.write(line + "\n")
        self.stdin.flush()

    def _next_line(self, timeout: float) -> str:
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{self.tag}: no reply within {timeout} s") from None

    def read_reply(self, timeout: float) -> tuple[str, list[str]]:
        lines, move = [], None
        while move is None:
            line = self._next_line(timeout)
            lines.append(line)
            matched = MOVE_RE.match(line)
            if matched:
                move = matched.group(1)
        return move, lines

    def close(self) -> None:
        try:
            self.cmd("exit")
            self.stdin.close()
            self.proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(self.proc.pid, 9)
            except ProcessLookupError:
                self.proc.kill()
        if self.err is not None:
            try:
                self.err.close()
            except OSError:
                pass


def spawn_ours(depth: int, errlog: Path) -> Engine:
    err = open(errlog, "w")
    proc = subprocess.Popen(
        [str(OUR_BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    engine = Engine(proc, "ours")
    engine.err = err
    engine.cmd(f"depth {depth}")
    return engine


def spawn_cloudict(workdir: str, depth: int, errlog: Path) -> Engine:
    err = open(errlog, "w")
    proc = subprocess.Popen(
        [str(CLOUDICT_BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        bufsize=1,
        cwd=workdir,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    engine = Engine(proc, "cloudict")
    engine.err = err
    engine.cmd(f"depth {depth}")
    return engine


def stats_from(lines: list[str]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for line in lines:
        matched = NODE_RE.match(line)
        if matched:
            stats["nodes"] = int(matched.group(1))
        matched = SCORE_RE.match(line)
        if matched:
            stats["score"] = float(matched.group(1))
        if line.startswith("Book:"):
            stats["book"] = True
    return stats


def play_game(ours: Engine, cloudict: Engine, ours_black: bool, opening: str, depth: int) -> dict[str, object]:
    board = [[0] * GRID_NUM for _ in range(GRID_NUM)]
    init_board(board)
    moves: list[dict[str, object]] = []
    black, white = (ours, cloudict) if ours_black else (cloudict, ours)
    black.cmd("new black")
    first, _ = black.read_reply(DEADLINE_S)
    assert first == "JJ", f"black must open the center, got {first}"
    white.cmd("new white")
    white.cmd(f"depth {depth}")
    white.cmd("black JJ")
    white.cmd(f"white {opening}")
    moves.append({"ply": 1, "color": "black", "engine": black.tag, "msg": "JJ"})
    moves.append({"ply": 2, "color": "white", "engine": white.tag, "msg": opening})
    make_move(board, msg2move("JJ"), Color.BLACK)
    make_move(board, msg2move(opening), Color.WHITE)
    cur, cur_color, ply, winner = black, Color.BLACK, 2, None
    while ply < MAX_PLIES:
        start = time.perf_counter()
        cur.cmd(f"move {moves[-1]['msg']}")
        reply, lines = cur.read_reply(DEADLINE_S)
        wall = time.perf_counter() - start
        ply += 1
        color = "black" if cur_color is Color.BLACK else "white"
        record = {"ply": ply, "color": color, "engine": cur.tag, "msg": reply, "wall": round(wall, 3)}
        record.update(stats_from(lines))
        moves.append(record)
        move = msg2move(reply)
        make_move(board, move, cur_color)
        end = check_game_end(board, move)
        if end == GameResult.WIN:
            winner = color
            break
        if end == GameResult.DRAW:
            winner = "draw"
            break
        cur = white if cur is black else black
        cur_color = Color.WHITE if cur_color is Color.BLACK else Color.BLACK
    return {
        "opening": opening,
        "ours_color": "black" if ours_black else "white",
        "depth": depth,
        "winner": winner,
        "plies": ply,
        "moves": moves,
    }


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="cloudict_run_")
    errdir = Path(tempfile.mkdtemp(prefix="cloudict_err_"))
    shutil.copy(PATTERNS, workdir)
    rows: list[dict[str, object]] = []
    game_no = 0
    for depth in DEPTHS:
        for opening in OPENINGS:
            for ours_black in (True, False):
                game_no += 1
                record = None
                for attempt in (1, 2, 3):
                    ours = spawn_ours(depth, errdir / f"ours_{game_no}_{attempt}.txt")
                    cloudict = spawn_cloudict(workdir, depth, errdir / f"cloudict_{game_no}_{attempt}.txt")
                    try:
                        record = play_game(ours, cloudict, ours_black, opening, depth)
                        break
                    except (BrokenPipeError, TimeoutError) as exc:
                        print(f"game {game_no} attempt {attempt} failed: {exc!r} — respawning", flush=True)
                        time.sleep(3.0)
                    finally:
                        ours.close()
                        cloudict.close()
                if record is None:
                    print(f"game {game_no} (depth {depth} {opening} "
                          f"{'black' if ours_black else 'white'}) forfeited after 3 attempts", flush=True)
                    continue
                ours_won = record["winner"] == ("black" if ours_black else "white")
                print(
                    f"depth {depth} opening {opening} ours as {'black' if ours_black else 'white'}: "
                    f"winner {record['winner']} ({record['plies']} plies) "
                    f"{'— OUR WIN' if ours_won else ''}",
                    flush=True,
                )
                rows.append(record)
                OUT_FILE.write_text(json.dumps(rows, indent=1))
                time.sleep(2.0)
    shutil.rmtree(workdir, ignore_errors=True)
    wins = sum(
        1
        for row in rows
        if row["winner"] == ("black" if row["ours_color"] == "black" else "white")
    )
    print(f"\n{wins}/{len(rows)} wins for the compiled engine across {len(DEPTHS)} depths")
    print(f"engine stderr captured in {errdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())