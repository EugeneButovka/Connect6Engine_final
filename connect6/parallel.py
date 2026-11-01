import os
import time
from concurrent.futures import Future, ProcessPoolExecutor

from .board import copy_board, is_board_empty, make_move, opponent
from .candidates import generate_scored_candidate_moves
from .defines import (
    FORWARD_WIDTHS,
    MAXINT,
    MININT,
    V5_DEEPEN_STEPS,
    V5_SCREEN_WIDTH,
    V6_GROWTH_FACTOR,
    V6_JOBS_ENV,
    V6_NODE_BUDGET,
    V6_SCREEN_RESERVE,
    V6_TIME_BUDGET,
    V6_TIME_BUDGET_ENV,
    Board,
    Cell,
    Color,
    Move,
    Position,
    SearchStats,
    SearchResult,
)
from .evaluation import evaluate
from .search import search_candidate_subtree
from .vcf import find_forced_win, join_threats

_pool: ProcessPoolExecutor | None = None
_pool_jobs: int = 0
_pool_stale: bool = False


def worker_count() -> int:
    raw = os.environ.get(V6_JOBS_ENV, "")
    try:
        configured = int(raw)
    except ValueError:
        configured = 0
    if configured > 0:
        return configured
    return os.cpu_count() or 1


def configured_budget() -> float:
    raw = os.environ.get(V6_TIME_BUDGET_ENV, "")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0.0:
            return value
    return V6_TIME_BUDGET


def _drop_pool() -> None:
    global _pool, _pool_stale
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
    _pool_stale = False


def _ensure_pool(jobs: int) -> ProcessPoolExecutor:
    global _pool, _pool_jobs, _pool_stale
    if _pool is not None and (_pool_stale or _pool_jobs != jobs):
        _drop_pool()
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=jobs)
        _pool_jobs = jobs
        _pool_stale = False
    return _pool


def _mark_stale() -> None:
    global _pool_stale
    if _pool is not None:
        _pool_stale = True


def _worker_subtree(board: Board, color: Color, depth: int, cells: tuple[int, int, int, int], incoming: int) -> tuple[int, int]:
    candidate = Move((Position(cells[0], cells[1]), Position(cells[2], cells[3])))
    stats = SearchStats()
    result = search_candidate_subtree(board, color, depth, candidate, incoming, stats)
    return result.score, stats.node_count


def _worker_defense(board: Board, other: Color, deadline: float | None = None) -> tuple[int, int, int]:
    stats = SearchStats()
    loses = 1 if find_forced_win(board, other, stats, deadline=deadline) is not None else 0
    armed = len(join_threats(board, other, stats))
    return loses, armed, stats.node_count


def _move_cells(move: Move) -> tuple[int, int, int, int]:
    return (move.positions[0].x, move.positions[0].y, move.positions[1].x, move.positions[1].y)


def _split_root(
    board: Board,
    color: Color,
    depth: int,
    pre_move: Move,
    stats: SearchStats,
    deadline: float | None,
) -> tuple[SearchResult, bool]:
    candidates, threats = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], color)
    if not candidates:
        stats.node_count += 1
        return SearchResult(pre_move, evaluate(board, pre_move)), True
    futures: list[Future[tuple[int, int]]] | None = None
    if worker_count() > 1:
        executor = _ensure_pool(worker_count())
        futures = [
            executor.submit(_worker_subtree, board, color, depth, _move_cells(candidate), threat)
            for candidate, threat in zip(candidates, threats)
        ]
    maximizing = color == Color.BLACK
    best_score = MININT if maximizing else MAXINT
    best_move = candidates[0]
    nodes = 1
    complete = True
    for index, candidate in enumerate(candidates):
        if futures is None:
            if deadline is not None and time.perf_counter() >= deadline:
                complete = False
                break
            score, spent = _worker_subtree(board, color, depth, _move_cells(candidate), threats[index])
        else:
            try:
                if deadline is None:
                    score, spent = futures[index].result()
                else:
                    score, spent = futures[index].result(timeout=max(deadline - time.perf_counter(), 0.0))
            except TimeoutError:
                complete = False
                _mark_stale()
                break
        nodes += spent
        if maximizing:
            if score > best_score:
                best_score, best_move = score, candidate
        elif score < best_score:
            best_score, best_move = score, candidate
        if best_score == (MAXINT if maximizing else MININT):
            break
    stats.node_count += nodes
    return SearchResult(best_move, best_score), complete


def _fixed_deepen(
    board: Board,
    color: Color,
    depth: int,
    pre_move: Move,
    stats: SearchStats,
) -> tuple[SearchResult, int]:
    best: SearchResult | None = None
    spent = 0
    reached = depth
    for step in range(V5_DEEPEN_STEPS + 1):
        iteration = SearchStats()
        best, _ = _split_root(board, color, depth + step, pre_move, iteration, None)
        spent += iteration.node_count
        stats.node_count += iteration.node_count
        reached = depth + step
        if spent >= V6_NODE_BUDGET or best.score in (MAXINT, MININT):
            break
    assert best is not None
    return best, reached


def _budget_deepen(
    board: Board,
    color: Color,
    depth: int,
    pre_move: Move,
    stats: SearchStats,
    deadline: float,
) -> tuple[SearchResult, int]:
    best: SearchResult | None = None
    reached = depth
    current = depth
    while True:
        iteration = SearchStats()
        started = time.perf_counter()
        result, complete = _split_root(board, color, current, pre_move, iteration, deadline)
        wall = time.perf_counter() - started
        stats.node_count += iteration.node_count
        if complete:
            best = result
            reached = current
            if result.score in (MAXINT, MININT):
                break
            if deadline - time.perf_counter() < wall * V6_GROWTH_FACTOR:
                break
            current += 1
        else:
            if best is None or result.score == (MAXINT if color == Color.BLACK else MININT):
                best = result
                reached = current
            break
    assert best is not None
    return best, reached


def _probe_defenses(
    board: Board,
    color: Color,
    other: Color,
    pool: list[Move],
    stats: SearchStats,
    deadline: float | None,
) -> list[tuple[int, int]]:
    results: list[tuple[int, int]] = []
    if worker_count() > 1 and len(pool) > 1:
        executor = _ensure_pool(worker_count())
        futures = []
        for candidate in pool:
            probe = copy_board(board)
            make_move(probe, candidate, color)
            futures.append(executor.submit(_worker_defense, probe, other, deadline))
        for future in futures:
            if deadline is None:
                loses, armed, nodes = future.result()
            else:
                try:
                    loses, armed, nodes = future.result(timeout=max(deadline - time.perf_counter(), 0.0))
                except TimeoutError:
                    _mark_stale()
                    break
            stats.node_count += nodes
            results.append((loses, armed))
    else:
        for candidate in pool:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            probe = copy_board(board)
            make_move(probe, candidate, color)
            loses, armed, nodes = _worker_defense(probe, other, deadline)
            stats.node_count += nodes
            results.append((loses, armed))
    return results


def _screen(
    board: Board,
    color: Color,
    best: SearchResult,
    stats: SearchStats,
    deadline: float | None = None,
) -> SearchResult:
    other = opponent(color)
    work = copy_board(board)
    make_move(work, best.move, color)
    loses = find_forced_win(work, other, stats, deadline=deadline) is not None
    armed = join_threats(work, other, stats)
    quality = (1 if loses else 0, len(armed))
    if quality == (0, 0):
        return best
    candidates, _ = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], color)
    pool: list[Move] = [move for move in candidates[:V5_SCREEN_WIDTH] if move != best.move]
    cells: set[Cell] = set()
    for join in armed:
        cells.update(
            ((join.positions[0].x, join.positions[0].y), (join.positions[1].x, join.positions[1].y))
        )
    for first in sorted(cells):
        for second in sorted(cells):
            if first < second:
                pool.append(Move((Position(first[0], first[1]), Position(second[0], second[1]))))
    probed = _probe_defenses(board, color, other, pool, stats, deadline)
    chosen = best
    for candidate, quality_probed in zip(pool, probed):
        if quality_probed < quality:
            quality = quality_probed
            chosen = SearchResult(candidate, best.score)
    return chosen


def search_v6(
    board: Board,
    color: Color,
    depth: int,
    pre_move: Move,
    stats: SearchStats,
    use_vcf: bool = True,
    budget: float | None = None,
) -> SearchResult:
    if is_board_empty(board):
        return SearchResult(Move((Position(10, 10), Position(10, 10))), 0)
    start = time.perf_counter()
    deadline = None if budget is None else start + budget
    if use_vcf:
        scan_deadline = None if deadline is None else start + (deadline - start) * 0.5
        win = find_forced_win(board, color, stats, deadline=scan_deadline)
        if win is not None:
            return SearchResult(win, MAXINT if color == Color.BLACK else MININT)
    if deadline is None:
        best, reached = _fixed_deepen(board, color, depth, pre_move, stats)
    else:
        reserve = min(V6_SCREEN_RESERVE, (deadline - start) * 0.25) if use_vcf else 0.0
        best, reached = _budget_deepen(board, color, depth, pre_move, stats, deadline - reserve)
    stats.depth_reached = reached
    if best.score in (MAXINT, MININT) or not use_vcf:
        return best
    return _screen(board, color, best, stats, deadline)