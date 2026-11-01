from .defines import (
    DIRECTIONS,
    GRID_NUM,
    LIVE_WEIGHTS,
    MAXINT,
    MININT,
    WINDOW_FACTOR,
    Board,
    Color,
    GameResult,
    Move,
)
from .board import check_game_end, measure_line


def evaluate(board: Board, pre_move: Move) -> int:
    result = check_game_end(board, pre_move)
    if result == GameResult.WIN:
        winner = board[pre_move.positions[0].x][pre_move.positions[0].y]
        if winner == Color.BLACK:
            return MAXINT
        return MININT
    if result == GameResult.DRAW:
        return 0
    return _score_living_sets(board, Color.BLACK) - _score_living_sets(board, Color.WHITE)


def _score_living_sets(board: Board, color: Color) -> int:
    value = 0
    for x in range(1, GRID_NUM - 1):
        for y in range(1, GRID_NUM - 1):
            if board[x][y] != color:
                continue
            for direction in DIRECTIONS:
                if board[x - direction[0]][y - direction[1]] == color.value:
                    continue
                length, free, _, _ = measure_line(board, x, y, direction)
                if length + free >= 6:
                    weight = LIVE_WEIGHTS[min(length, 5)]
                    if length < 5:
                        weight *= WINDOW_FACTOR[min(free - (6 - length), 2)]
                    value += weight
    return value