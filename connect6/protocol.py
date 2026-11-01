import sys
import time

from .defines import (
    Board,
    GRID_NUM,
    LOG_FILE,
    NOSTONE,
    Color,
    Move,
    Position,
)


def color_to_name(color: Color) -> str:
    if color == Color.BLACK:
        return "black"
    return "white"


def _timestamp() -> str:
    return time.ctime(time.time())[:-1]


def _append_log(line: str) -> None:
    try:
        with open(LOG_FILE, "a") as file:
            file.write(line)
    except OSError:
        print(f"Error: Can't open log file - {LOG_FILE}", file=sys.stderr)


def log_command(msg: str) -> None:
    _append_log(f"[{_timestamp()}] - {msg}\n")


def log_error(msg: str) -> None:
    _append_log(f"[{_timestamp()}] - ERROR: {msg}\n")


def move2msg(move: Move) -> str:
    first = move.positions[0]
    second = move.positions[1]
    if first.x == second.x and first.y == second.y:
        return f"{chr(ord('S') - first.x + 1)}{chr(first.y + ord('A') - 1)}"
    return (
        f"{chr(first.y + ord('A') - 1)}{chr(ord('S') - first.x + 1)}"
        f"{chr(second.y + ord('A') - 1)}{chr(ord('S') - second.x + 1)}"
    )


def msg2move(msg: str) -> Move:
    if len(msg) not in (2, 4):
        raise ValueError(f"invalid move message {msg!r}: expected 2 or 4 letters")
    for char in msg:
        if not "A" <= char <= "S":
            raise ValueError(f"invalid coordinate {char!r} in move message {msg!r}: expected A-S")
    if len(msg) == 2:
        x = ord('S') - ord(msg[1]) + 1
        y = ord(msg[0]) - ord('A') + 1
        return Move((Position(x, y), Position(x, y)))
    return Move(
        (
            Position(ord('S') - ord(msg[1]) + 1, ord(msg[0]) - ord('A') + 1),
            Position(ord('S') - ord(msg[3]) + 1, ord(msg[2]) - ord('A') + 1),
        )
    )


def print_board(board: Board) -> None:
    print("   " + "".join([chr(i + ord('A') - 1) + " " for i in range(1, GRID_NUM - 1)]))
    for i in range(1, GRID_NUM - 1):
        print(f"{chr(ord('A') - 1 + i)}", end=" ")
        for j in range(1, GRID_NUM - 1):
            x = GRID_NUM - 1 - j
            y = i
            stone = board[x][y]
            if stone == NOSTONE:
                print(" -", end="")
            elif stone == Color.BLACK:
                print(" O", end="")
            elif stone == Color.WHITE:
                print(" *", end="")
        print(" ", end="")
        print(f"{chr(ord('A') - 1 + i)}", end="\n")
    print("   " + "".join([chr(i + ord('A') - 1) + " " for i in range(1, GRID_NUM - 1)]))