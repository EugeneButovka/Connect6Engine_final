import time

from .board import copy_board, is_board_empty, make_move, opponent, unmake_move
from .candidates import generate_scored_candidate_moves
from .defines import (
    DIRECTIONS,
    FORWARD_WIDTHS,
    GRID_NUM,
    MAXINT,
    MININT,
    NOSTONE,
    Board,
    Cell,
    Color,
    Move,
    Position,
    SearchStats,
    SearchResult,
    V5_DEEPEN_STEPS,
    V5_NODE_BUDGET,
    V5_SCREEN_WIDTH,
    VCF_NODE_CAP,
    VCF_PLY_CAP,
)
from .search import search_alpha_beta_with_quiescence

SideScan = tuple[bool, list[Cell], list[tuple[Cell, Cell]], list[Cell]]


def search_v5(
    board: Board,
    color: Color,
    depth: int,
    pre_move: Move,
    stats: SearchStats,
    use_vcf: bool = True,
) -> SearchResult:
    if is_board_empty(board):
        return SearchResult(Move((Position(10, 10), Position(10, 10))), 0)
    if use_vcf:
        win = find_forced_win(board, color, stats)
        if win is not None:
            return SearchResult(win, MAXINT if color == Color.BLACK else MININT)
    best: SearchResult | None = None
    spent = 0
    for step in range(V5_DEEPEN_STEPS + 1):
        iteration = SearchStats()
        best = search_alpha_beta_with_quiescence(board, color, depth + step, pre_move, iteration)
        spent += iteration.node_count
        stats.node_count += iteration.node_count
        if spent >= V5_NODE_BUDGET or best.score in (MAXINT, MININT):
            break
    assert best is not None
    if best.score in (MAXINT, MININT) or not use_vcf:
        return best
    other = opponent(color)
    work = copy_board(board)
    make_move(work, best.move, color)
    loses = find_forced_win(work, other, stats) is not None
    armed = len(join_threats(work, other, stats))
    if not loses and armed == 0:
        return best
    quality = (1 if loses else 0, armed)
    return _screen_fallback(board, color, best, quality, stats)


def _screen_fallback(
    board: Board,
    color: Color,
    best: SearchResult,
    quality: tuple[int, int],
    stats: SearchStats,
) -> SearchResult:
    other = opponent(color)
    candidates, _ = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], color)
    pool: list[Move] = [move for move in candidates[:V5_SCREEN_WIDTH] if move != best.move]
    work = copy_board(board)
    make_move(work, best.move, color)
    cells: set[Cell] = set()
    for join in join_threats(work, other, stats):
        cells.update(((join.positions[0].x, join.positions[0].y), (join.positions[1].x, join.positions[1].y)))
    for first in sorted(cells):
        for second in sorted(cells):
            if first < second:
                pool.append(Move((Position(first[0], first[1]), Position(second[0], second[1]))))
    chosen = best
    for candidate in pool:
        probe = copy_board(board)
        make_move(probe, candidate, color)
        loses = find_forced_win(probe, other, stats) is not None
        armed = len(join_threats(probe, other, stats))
        if (1 if loses else 0, armed) < quality:
            quality = (1 if loses else 0, armed)
            chosen = SearchResult(candidate, best.score)
    return chosen


def join_threats(board: Board, attacker: Color, stats: SearchStats | None = None) -> list[Move]:
    armed: list[Move] = []
    for join in _split_joins(board, attacker):
        make_move(board, join, attacker)
        mine, _ = _scan(board, attacker)
        cost, _ = _cover_options(mine[1], mine[2])
        unmake_move(board, join)
        if stats is not None:
            stats.node_count += 1
        if cost >= 2:
            armed.append(join)
    return armed


def _split_joins(board: Board, attacker: Color) -> list[Move]:
    joins: set[tuple[Cell, Cell]] = set()
    for dx, dy in DIRECTIONS:
        for x in range(1, GRID_NUM - 1):
            for y in range(1, GRID_NUM - 1):
                if board[x][y] != attacker.value:
                    continue
                px, py = x + 3 * dx, y + 3 * dy
                if not (1 <= px < GRID_NUM - 1 and 1 <= py < GRID_NUM - 1):
                    continue
                if board[px][py] != attacker.value:
                    continue
                if board[x + dx][y + dy] != NOSTONE or board[x + 2 * dx][y + 2 * dy] != NOSTONE:
                    continue
                first = (x + dx, y + dy)
                second = (x + 2 * dx, y + 2 * dy)
                joins.add((first, second) if first < second else (second, first))
    return [Move((Position(a[0], a[1]), Position(b[0], b[1]))) for a, b in sorted(joins)]


def find_forced_win(
    board: Board,
    color: Color,
    stats: SearchStats | None = None,
    ply_cap: int = VCF_PLY_CAP,
    node_cap: int = VCF_NODE_CAP,
    deadline: float | None = None,
) -> Move | None:
    work = copy_board(board)
    counter = [0]
    win = _attack(work, color, ply_cap, counter, node_cap, deadline)
    if stats is not None:
        stats.node_count += counter[0]
    return win


def _attack(
    board: Board,
    color: Color,
    plies: int,
    counter: list[int],
    node_cap: int,
    deadline: float | None = None,
) -> Move | None:
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    counter[0] += 1
    if counter[0] > node_cap or plies <= 0:
        return None
    mine, theirs = _scan(board, color)
    if mine[0] or theirs[0]:
        return None
    if mine[1]:
        return _completion(board, mine[1], mine[3])
    if mine[2]:
        first, second = mine[2][0]
        return Move((Position(first[0], first[1]), Position(second[0], second[1])))
    cost, covers = _cover_options(theirs[1], theirs[2])
    if cost > 2:
        return None
    pairs = set(_candidate_pairs(cost, covers, mine[3]))
    for join in _split_joins(board, color):
        pairs.add(((join.positions[0].x, join.positions[0].y), (join.positions[1].x, join.positions[1].y)))
    for pair in sorted(pairs):
        move = Move((Position(pair[0][0], pair[0][1]), Position(pair[1][0], pair[1][1])))
        make_move(board, move, color)
        win = None
        new_mine, new_theirs = _scan(board, color)
        if new_mine[0]:
            win = move
        elif not new_theirs[0]:
            reply_cost, replies = _cover_options(new_mine[1], new_mine[2])
            if reply_cost > 2:
                win = move
            elif reply_cost == 2:
                win = move if _defender_loses(board, color, replies, plies, counter, node_cap, deadline) else None
        unmake_move(board, move)
        if win is not None:
            return win
    return None


def _defender_loses(
    board: Board,
    color: Color,
    replies: list[tuple[Cell, Cell]],
    plies: int,
    counter: list[int],
    node_cap: int,
    deadline: float | None = None,
) -> bool:
    for reply in replies:
        move = Move((Position(reply[0][0], reply[0][1]), Position(reply[1][0], reply[1][1])))
        make_move(board, move, opponent(color))
        refuted = _attack(board, color, plies - 2, counter, node_cap, deadline) is None
        unmake_move(board, move)
        if refuted:
            return False
    return True


def _completion(board: Board, fives: list[Cell], cells: list[Cell]) -> Move:
    first = fives[0]
    partner = next((cell for cell in fives[1:] if cell != first), None)
    if partner is None:
        partner = next((cell for cell in cells if cell != first), None)
    if partner is None:
        partner = _any_empty(board, first)
    return Move((Position(first[0], first[1]), Position(partner[0], partner[1])))


def _any_empty(board: Board, exclude: Cell) -> Cell:
    for x in range(1, GRID_NUM - 1):
        for y in range(1, GRID_NUM - 1):
            if board[x][y] == NOSTONE and (x, y) != exclude:
                return (x, y)
    return exclude


def _cover_options(
    fives: list[Cell],
    fours: list[tuple[Cell, Cell]],
) -> tuple[int, list[tuple[Cell, Cell]]]:
    sets = [frozenset((cell,)) for cell in fives] + [frozenset(pair) for pair in fours]
    if not sets:
        return 0, []
    union = sorted(set().union(*sets))
    singles = [cell for cell in union if all(cell in group for group in sets)]
    if singles:
        return 1, [(cell, cell) for cell in singles]
    pairs: list[tuple[Cell, Cell]] = []
    for i, first in enumerate(union):
        for second in union[i + 1 :]:
            if all(first in group or second in group for group in sets):
                pairs.append((first, second))
    if pairs:
        return 2, pairs
    return 3, []


def _candidate_pairs(
    cost: int,
    covers: list[tuple[Cell, Cell]],
    cells: list[Cell],
) -> list[tuple[Cell, Cell]]:
    pairs: set[tuple[Cell, Cell]] = set()
    if cost == 0:
        for i, first in enumerate(cells):
            for second in cells[i + 1 :]:
                pairs.add((first, second))
    elif cost == 1:
        for cover, _ in covers:
            for second in cells:
                if second != cover:
                    pairs.add((cover, second) if cover < second else (second, cover))
    else:
        for first, second in covers:
            pairs.add((first, second) if first < second else (second, first))
    return sorted(pairs)


def _scan(board: Board, color: Color) -> tuple[SideScan, SideScan]:
    other = color.value ^ 3
    me_fives: list[Cell] = []
    me_fours: list[tuple[Cell, Cell]] = []
    me_cells: set[Cell] = set()
    them_fives: list[Cell] = []
    them_fours: list[tuple[Cell, Cell]] = []
    them_cells: set[Cell] = set()
    me_done = False
    them_done = False
    for dx, dy in DIRECTIONS:
        x_lo = 1 if dx >= 0 else 6
        x_hi = GRID_NUM - 7 if dx > 0 else GRID_NUM - 2
        y_lo = 1 if dy >= 0 else 6
        y_hi = GRID_NUM - 7 if dy > 0 else GRID_NUM - 2
        for x in range(x_lo, x_hi + 1):
            for y in range(y_lo, y_hi + 1):
                me = 0
                them = 0
                empties: list[Cell] = []
                for step in range(6):
                    value = board[x + dx * step][y + dy * step]
                    if value == color.value:
                        me += 1
                    elif value == other:
                        them += 1
                    elif value == NOSTONE:
                        empties.append((x + dx * step, y + dy * step))
                if me and them:
                    continue
                if me >= 6:
                    me_done = True
                elif them >= 6:
                    them_done = True
                elif me == 5:
                    me_fives.append(empties[0])
                    me_cells.update(empties)
                elif me == 4:
                    me_fours.append((empties[0], empties[1]))
                    me_cells.update(empties)
                elif me == 3:
                    me_cells.update(empties)
                elif them == 5:
                    them_fives.append(empties[0])
                    them_cells.update(empties)
                elif them == 4:
                    them_fours.append((empties[0], empties[1]))
                    them_cells.update(empties)
                elif them == 3:
                    them_cells.update(empties)
    mine: SideScan = (me_done, me_fives, me_fours, sorted(me_cells))
    theirs: SideScan = (them_done, them_fives, them_fours, sorted(them_cells))
    return mine, theirs