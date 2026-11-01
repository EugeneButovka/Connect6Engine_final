from .defines import (
    BORDER,
    DIRECTIONS,
    GRID_NUM,
    NOSTONE,
    Board,
    Color,
    Direction,
    THREAT_WIN,
)


def count_line_threats(cells: list[int], attacker: int) -> int:
    count = 0
    marked = [False] * len(cells)
    for start in range(len(cells) - 5):
        has_defender = False
        attacker_stones = 0
        for value in cells[start : start + 6]:
            if value == attacker:
                attacker_stones += 1
            elif value != NOSTONE:
                has_defender = True
                break
        if has_defender:
            continue
        if attacker_stones == 6:
            return THREAT_WIN
        if attacker_stones >= 4 and not any(marked[start : start + 6]):
            for offset in range(start, start + 6):
                if cells[offset] == NOSTONE:
                    marked[offset] = True
            count += 1
    return count


def count_move_threats(board: Board, x1: int, y1: int, x2: int, y2: int, color: Color) -> int:
    attacker = color.value
    board[x1][y1] = attacker
    board[x2][y2] = attacker
    total = 0
    for direction in DIRECTIONS:
        axis = _count_axis_threats(board, x1, y1, x2, y2, direction, attacker)
        if axis >= THREAT_WIN:
            total = THREAT_WIN
            break
        total += axis
    board[x1][y1] = NOSTONE
    board[x2][y2] = NOSTONE
    return total


def _count_axis_threats(
    board: Board,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    direction: Direction,
    attacker: int,
) -> int:
    dx, dy = direction
    if dx * (y2 - y1) != dy * (x2 - x1):
        first = count_line_threats(_extract_slice(board, x1, y1, direction, -5, 5), attacker)
        return first + count_line_threats(_extract_slice(board, x2, y2, direction, -5, 5), attacker)
    steps = x2 - x1 if dx else y2 - y1
    if steps == 0 or abs(steps) > 10:
        first = count_line_threats(_extract_slice(board, x1, y1, direction, -5, 5), attacker)
        return first + count_line_threats(_extract_slice(board, x2, y2, direction, -5, 5), attacker)
    lo = min(-5, steps - 5)
    hi = max(5, steps + 5)
    return count_line_threats(_extract_slice(board, x1, y1, direction, lo, hi), attacker)


def _extract_slice(
    board: Board,
    x: int,
    y: int,
    direction: Direction,
    lo: int,
    hi: int,
) -> list[int]:
    dx, dy = direction
    cells: list[int] = []
    for offset in range(lo, hi + 1):
        cx = x + dx * offset
        cy = y + dy * offset
        if 0 <= cx < GRID_NUM and 0 <= cy < GRID_NUM:
            cells.append(board[cx][cy])
        else:
            cells.append(BORDER)
    return cells