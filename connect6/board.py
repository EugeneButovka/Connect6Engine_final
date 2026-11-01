from .defines import (
    BORDER,
    Cell,
    DIRECTIONS,
    GRID_NUM,
    NOSTONE,
    Board,
    Color,
    Direction,
    GameResult,
    Line,
    Move,
)


def init_board(board: Board) -> None:
    for i in range(GRID_NUM):
        board[i][0] = board[0][i] = board[i][GRID_NUM - 1] = board[GRID_NUM - 1][i] = BORDER
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            board[i][j] = NOSTONE


def copy_board(board: Board) -> Board:
    return [row[:] for row in board]


def make_move(board: Board, move: Move, color: Color) -> None:
    board[move.positions[0].x][move.positions[0].y] = color.value
    board[move.positions[1].x][move.positions[1].y] = color.value


def unmake_move(board: Board, move: Move) -> None:
    board[move.positions[0].x][move.positions[0].y] = NOSTONE
    board[move.positions[1].x][move.positions[1].y] = NOSTONE


def opponent(color: Color) -> Color:
    if color == Color.BLACK:
        return Color.WHITE
    return Color.BLACK


def measure_line(
    board: Board,
    x: int,
    y: int,
    direction: Direction,
    max_length: int = GRID_NUM,
    max_free: int = GRID_NUM,
) -> Line:
    stone = board[x][y]
    length = 0
    free = 0
    ends: list[Cell] = []
    for sign in (1, -1):
        step_x = direction[0] * sign
        step_y = direction[1] * sign
        px = x
        py = y
        if sign == -1:
            px += step_x
            py += step_y
        while length < max_length and board[px][py] == stone:
            px += step_x
            py += step_y
            length += 1
        while free < max_free and board[px][py] == NOSTONE:
            px += step_x
            py += step_y
            free += 1
        ends.append((px, py))
    return length, free, ends[0], ends[1]


def _is_win_by_move(board: Board, pre_move: Move) -> bool:
    for direction in DIRECTIONS:
        for position in pre_move.positions:
            stone = board[position.x][position.y]
            if stone == BORDER or stone == NOSTONE:
                continue
            if measure_line(board, position.x, position.y, direction)[0] >= 6:
                return True
    return False


def _is_board_full(board: Board) -> bool:
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            if board[i][j] == NOSTONE:
                return False
    return True


def is_board_empty(board: Board) -> bool:
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            if board[i][j] != NOSTONE:
                return False
    return True


def check_game_end(board: Board, pre_move: Move) -> GameResult | None:
    if _is_win_by_move(board, pre_move):
        return GameResult.WIN
    if _is_board_full(board):
        return GameResult.DRAW
    return None