from .defines import (
    FORWARD_WIDTHS,
    MAXINT,
    MININT,
    MAX_CANDIDATE_MOVES,
    QUIESCENCE_DEPTH,
    QUIESCENCE_MIN_THREATS,
    QUIESCENCE_WIDTH,
    THREAT_WIN,
    Board,
    Color,
    Move,
    Position,
    SearchStats,
    SearchResult,
    SetWeight,
)
from .board import copy_board, is_board_empty, make_move, opponent, unmake_move
from .candidates import generate_candidate_moves, generate_scored_candidate_moves
from .evaluation import evaluate


def search_min_max(board: Board, color: Color, depth: int, pre_move: Move, stats: SearchStats) -> SearchResult:
    return _search(board, color, depth, pre_move, stats, prune=False)


def search_alpha_beta(board: Board, color: Color, depth: int, pre_move: Move, stats: SearchStats) -> SearchResult:
    return _search(board, color, depth, pre_move, stats, prune=True)


def search_alpha_beta_with_quiescence(board: Board, color: Color, depth: int, pre_move: Move, stats: SearchStats) -> SearchResult:
    return _search(board, color, depth, pre_move, stats, prune=True, enhanced=True)


def search_candidate_subtree(
    board: Board,
    color: Color,
    depth: int,
    candidate: Move,
    incoming: int,
    stats: SearchStats,
) -> SearchResult:
    """One split-root candidate of v6: its own full-window enhanced subtree,
    searched with principal variation search."""
    work = copy_board(board)
    make_move(work, candidate, color)
    return _alpha_beta(
        depth - 1,
        work,
        opponent(color),
        candidate,
        stats,
        MININT,
        MAXINT,
        True,
        True,
        1,
        incoming if incoming > 0 else 0,
        True,
    )


def _search(board: Board, color: Color, depth: int, pre_move: Move, stats: SearchStats, prune: bool, enhanced: bool = False) -> SearchResult:
    working_board = copy_board(board)
    if is_board_empty(working_board):
        center = Move((Position(10, 10), Position(10, 10)))
        return SearchResult(center, 0)
    return _alpha_beta(depth, working_board, color, pre_move, stats, MININT, MAXINT, prune, enhanced)


def _alpha_beta(
    depth: int,
    board: Board,
    color: Color,
    pre_move: Move,
    stats: SearchStats,
    alpha: int,
    beta: int,
    prune: bool,
    enhanced: bool = False,
    ply: int = 0,
    incoming: int = 0,
    pvs: bool = False,
) -> SearchResult:
    if enhanced and depth <= 0:
        return _quiescence(board, color, pre_move, stats, alpha, beta, incoming, QUIESCENCE_DEPTH)
    stats.node_count += 1
    score = evaluate(board, pre_move)
    if score == MAXINT or score == MININT:
        return SearchResult(pre_move, score)
    if depth <= 0:
        return SearchResult(pre_move, score)

    threats: list[int] = []
    if enhanced:
        width = FORWARD_WIDTHS[min(ply, len(FORWARD_WIDTHS) - 1)]
        candidates, threats = generate_scored_candidate_moves(board, width, color)
    else:
        candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, color)
    if len(candidates) == 0:
        return SearchResult(pre_move, score)

    maximizing = color == Color.BLACK
    best_score = MININT if maximizing else MAXINT
    best_move = candidates[0]
    for index, candidate in enumerate(candidates):
        child_incoming = threats[index] if enhanced else 0
        make_move(board, candidate, color)
        if pvs and prune and index > 0 and alpha + 1 < beta:
            child = _scouted_child(
                depth - 1, board, color, candidate, stats, alpha, beta, prune, enhanced, ply + 1, child_incoming
            )
        else:
            child = _alpha_beta(
                depth - 1,
                board,
                opponent(color),
                candidate,
                stats,
                alpha,
                beta,
                prune,
                enhanced,
                ply + 1,
                child_incoming if child_incoming > 0 else 0,
                pvs,
            )
        unmake_move(board, candidate)
        best_score, best_move, alpha, beta, cutoff = _advance(
            maximizing, child.score, candidate, best_score, best_move, alpha, beta, prune
        )
        if cutoff:
            return SearchResult(best_move, best_score)
    return SearchResult(best_move, best_score)


def _scouted_child(
    depth: int,
    board: Board,
    color: Color,
    candidate: Move,
    stats: SearchStats,
    alpha: int,
    beta: int,
    prune: bool,
    enhanced: bool,
    ply: int,
    child_incoming: int,
) -> SearchResult:
    """A non-first child under principal variation search: scouted with a
    null window, re-searched wide when the scout fails high."""
    incoming = child_incoming if child_incoming > 0 else 0
    if color == Color.BLACK:
        scout = _alpha_beta(
            depth, board, opponent(color), candidate, stats, alpha, alpha + 1, prune, enhanced, ply, incoming, True
        )
        if scout.score <= alpha:
            return scout
    else:
        scout = _alpha_beta(
            depth, board, opponent(color), candidate, stats, beta - 1, beta, prune, enhanced, ply, incoming, True
        )
        if scout.score >= beta:
            return scout
    return _alpha_beta(
        depth, board, opponent(color), candidate, stats, alpha, beta, prune, enhanced, ply, incoming, True
    )


def _advance(
    maximizing: bool,
    child_score: int,
    candidate: Move,
    best_score: int,
    best_move: Move,
    alpha: int,
    beta: int,
    prune: bool,
) -> tuple[int, Move, int, int, bool]:
    if maximizing:
        if child_score > best_score:
            best_score = child_score
            best_move = candidate
        if prune:
            if best_score >= beta:
                return best_score, best_move, alpha, beta, True
            if best_score > alpha:
                alpha = best_score
    else:
        if child_score < best_score:
            best_score = child_score
            best_move = candidate
        if prune:
            if best_score <= alpha:
                return best_score, best_move, alpha, beta, True
            if best_score < beta:
                beta = best_score
    return best_score, best_move, alpha, beta, False


def _quiescence(
    board: Board,
    color: Color,
    pre_move: Move,
    stats: SearchStats,
    alpha: int,
    beta: int,
    incoming: int,
    qdepth: int,
) -> SearchResult:
    stats.node_count += 1
    maximizing = color == Color.BLACK
    stand = evaluate(board, pre_move)
    if stand == MAXINT or stand == MININT:
        return SearchResult(pre_move, stand)
    loud = incoming > 0 or abs(stand) >= int(SetWeight.LIVE_FIVE)
    if not loud:
        if maximizing and stand >= beta:
            return SearchResult(pre_move, stand)
        if not maximizing and stand <= alpha:
            return SearchResult(pre_move, stand)
    if qdepth <= 0:
        return SearchResult(pre_move, stand)

    moves, threats = generate_scored_candidate_moves(board, QUIESCENCE_WIDTH, color)
    if len(moves) == 0:
        return SearchResult(pre_move, stand)
    for move, threat in zip(moves, threats):
        if threat >= THREAT_WIN:
            return SearchResult(move, MAXINT if maximizing else MININT)

    if loud:
        pool = list(zip(moves, threats))
        best_score = MININT if maximizing else MAXINT
        best_move = moves[0]
    else:
        pool = [(move, threat) for move, threat in zip(moves, threats) if threat >= QUIESCENCE_MIN_THREATS]
        if len(pool) == 0:
            return SearchResult(pre_move, stand)
        best_score = stand
        best_move = pre_move
        if maximizing and stand > alpha:
            alpha = stand
        if not maximizing and stand < beta:
            beta = stand

    for move, threat in pool:
        make_move(board, move, color)
        child = _quiescence(
            board,
            opponent(color),
            move,
            stats,
            alpha,
            beta,
            threat if threat > 0 else 0,
            qdepth - 1,
        )
        unmake_move(board, move)
        best_score, best_move, alpha, beta, cutoff = _advance(
            maximizing, child.score, move, best_score, best_move, alpha, beta, True
        )
        if cutoff:
            return SearchResult(best_move, best_score)
    return SearchResult(best_move, best_score)