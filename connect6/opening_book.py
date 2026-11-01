import json
import sys

from .defines import GRID_NUM, NOSTONE, OPENING_BOOK_FILE, Board, Move
from .protocol import msg2move


def load_opening_book() -> dict[str, str]:
    if not OPENING_BOOK_FILE.is_file():
        return {}
    try:
        data = json.loads(OPENING_BOOK_FILE.read_text())
    except (OSError, ValueError):
        print(f"Ignoring unreadable opening book: {OPENING_BOOK_FILE}", file=sys.stderr)
        return {}
    book: dict[str, str] = {}
    for key, move in data.items():
        if isinstance(key, str) and isinstance(move, str):
            book[key] = move
    return book


def position_key(board: Board) -> str:
    stones: list[str] = []
    for x in range(1, GRID_NUM - 1):
        for y in range(1, GRID_NUM - 1):
            value = board[x][y]
            if value != NOSTONE:
                stones.append(f"{x}.{y}.{value}")
    return ",".join(stones)


def book_move(board: Board, book: dict[str, str]) -> Move | None:
    entry = book.get(position_key(board))
    if entry is None:
        return None
    try:
        move = msg2move(entry)
    except ValueError:
        return None
    for position in move.positions:
        if board[position.x][position.y] != NOSTONE:
            return None
    return move