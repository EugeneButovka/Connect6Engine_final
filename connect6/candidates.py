from .defines import (
    CellPair,
    Color,
    CompletionPair,
    DIRECTIONS,
    GRID_NUM,
    MAX_CANDIDATE_CELLS,
    MAX_CANDIDATE_MOVES,
    THREAT_RANK_WIDTH,
    THREAT_WEIGHT,
    THREAT_WIN,
    LIVE_WEIGHTS,
    NOSTONE,
    BORDER,
    WINDOW_FACTOR,
    Board,
    Move,
    Position,
    RankedPair,
    ScoredCell,
    SetWeight,
)
from .board import copy_board, measure_line
from .threats import count_move_threats


def generate_candidate_moves(board: Board, limit: int, color: Color) -> list[Move]:
    moves, _ = _generate_candidates(board, limit, color, score_threats=False)
    return moves


def generate_scored_candidate_moves(board: Board, limit: int, color: Color) -> tuple[list[Move], list[int]]:
    return _generate_candidates(board, limit, color, score_threats=True)


def _generate_candidates(
    board: Board,
    limit: int,
    color: Color,
    score_threats: bool,
) -> tuple[list[Move], list[int]]:
    scratch = copy_board(board)
    cells, completion_pairs = _get_scored_cells(scratch)
    cells = cells[:MAX_CANDIDATE_CELLS]
    moves: list[Move] = []
    threats: list[int] = []
    seen: set[CellPair] = set()
    for x1, y1, x2, y2 in completion_pairs:
        before = len(moves)
        _try_add_candidate(moves, seen, limit, x1, y1, x2, y2)
        if score_threats and len(moves) > before:
            threats.append(count_move_threats(scratch, x1, y1, x2, y2, color))
    pairs: list[RankedPair] = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            pairs.append((_pair_correction(scratch, cells[i], cells[j]), i, j))
    pairs.sort(reverse=True)
    pairs, counted = _rank_by_threats(scratch, cells, pairs, color)
    for _, i, j in pairs:
        if len(moves) >= limit:
            break
        before = len(moves)
        _try_add_candidate(moves, seen, limit, cells[i][1], cells[i][2], cells[j][1], cells[j][2])
        if len(moves) > before:
            threats.append(counted.get((i, j), -1))
    return moves, threats


def _get_scored_cells(board: Board) -> tuple[list[ScoredCell], list[CompletionPair]]:
    scored: list[ScoredCell] = []
    fillers: list[ScoredCell] = []
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            if board[i][j] != NOSTONE:
                continue
            if _count_neighbor_stones(board, i, j):
                scored.append((_score_position(board, i, j), i, j))
            else:
                fillers.append((0, i, j))
    scored.sort(reverse=True)
    completion_pairs: list[CompletionPair] = []
    for score, x, y in scored:
        if score >= SetWeight.LIVE_FIVE:
            completion_pairs.extend(_find_completion_pairs(board, x, y))
    return scored + fillers, completion_pairs


def _score_position(board: Board, x: int, y: int) -> int:
    total = 0
    for color in (Color.BLACK, Color.WHITE):
        board[x][y] = color.value
        value = 0
        for direction in DIRECTIONS:
            length, free, _, _ = measure_line(board, x, y, direction, 6, 6)
            if length + free >= 6:
                weight = LIVE_WEIGHTS[min(length, 5)]
                if length < 5:
                    weight *= WINDOW_FACTOR[min(free - (6 - length), 2)]
                value += weight
        board[x][y] = NOSTONE
        total += value
    return total


def _rank_by_threats(
    board: Board,
    cells: list[ScoredCell],
    pairs: list[RankedPair],
    color: Color,
) -> tuple[list[RankedPair], dict[tuple[int, int], int]]:
    top = pairs[:THREAT_RANK_WIDTH]
    win_pairs: list[RankedPair] = []
    ranked: list[RankedPair] = []
    counted: dict[tuple[int, int], int] = {}
    for score, i, j in top:
        threats = count_move_threats(board, cells[i][1], cells[i][2], cells[j][1], cells[j][2], color)
        counted[(i, j)] = threats
        if threats >= THREAT_WIN:
            win_pairs.append((score, i, j))
        else:
            ranked.append((score + THREAT_WEIGHT * threats, i, j))
    win_pairs.sort(reverse=True)
    ranked.sort(reverse=True)
    return win_pairs + ranked + pairs[THREAT_RANK_WIDTH:], counted


def _pair_correction(board: Board, cell_a: ScoredCell, cell_b: ScoredCell) -> int:
    score_a, x1, y1 = cell_a
    score_b, x2, y2 = cell_b
    score_sum = score_a + score_b
    dx = x2 - x1
    dy = y2 - y1
    if dx != 0 and dy != 0 and abs(dx) != abs(dy):
        return score_sum
    distance = max(abs(dx), abs(dy))
    if distance >= 12:
        return score_sum
    step_x = (dx > 0) - (dx < 0)
    step_y = (dy > 0) - (dy < 0)
    walker_x = x1 + step_x
    walker_y = y1 + step_y
    while walker_x != x2 or walker_y != y2:
        if board[walker_x][walker_y] != NOSTONE:
            return score_sum
        walker_x += step_x
        walker_y += step_y
    smaller = score_a if score_a < score_b else score_b
    if distance <= 6:
        return score_sum - smaller // 2
    return score_sum - smaller // 4


def _find_completion_pairs(board: Board, x: int, y: int) -> list[CompletionPair]:
    pairs: list[CompletionPair] = []
    for color in (Color.BLACK, Color.WHITE):
        board[x][y] = color.value
        for direction in DIRECTIONS:
            length, _, end_a, end_b = measure_line(board, x, y, direction, 6, 0)
            if length == 5:
                for end_x, end_y in (end_a, end_b):
                    if board[end_x][end_y] == NOSTONE:
                        pairs.append((x, y, end_x, end_y))
        board[x][y] = NOSTONE
    return pairs


def _count_neighbor_stones(board: Board, x: int, y: int) -> int:
    count = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            stone = board[x + dx][y + dy]
            if stone != NOSTONE and stone != BORDER:
                count += 1
    return count


def _try_add_candidate(
    moves: list[Move],
    seen: set[CellPair],
    limit: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    if len(moves) >= limit:
        return
    a = (x1, y1)
    b = (x2, y2)
    key = (a, b) if a <= b else (b, a)
    if key in seen:
        return
    seen.add(key)
    moves.append(Move((Position(x1, y1), Position(x2, y2))))