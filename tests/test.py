import io
import json
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ["WEIGHTS_FILE"] = str(Path(__file__).resolve().parents[1] / "tuning" / "default.json")
os.environ["BOOK_FILE"] = str(Path(__file__).resolve().parents[1] / "tuning" / "no_book.json")
os.environ["V6_TIME_BUDGET"] = "0.5"
os.environ["V6_JOBS"] = "2"

from connect6.board import init_board, _is_board_full, _is_win_by_move, check_game_end, copy_board, opponent, make_move
from connect6.game_engine import GameEngine
from connect6.protocol import color_to_name, move2msg, msg2move
from connect6.evaluation import evaluate, _score_living_sets
from connect6.candidates import generate_candidate_moves, generate_scored_candidate_moves, _pair_correction
from connect6.opening_book import book_move, position_key
from connect6.threats import count_line_threats, count_move_threats
from connect6.search import (
    SearchDeadlineExceeded,
    search_alpha_beta,
    search_alpha_beta_with_quiescence,
    search_candidate_subtree,
    search_min_max,
)
from connect6.vcf import find_forced_win, search_v5
from connect6.parallel import search_v6
from connect6.defines import (
    ENGINE_NAME,
    FORWARD_WIDTHS,
    GRID_NUM,
    SetWeight,
    THREAT_WIN,
    V6_JOBS_ENV,
    MAXINT,
    MAX_CANDIDATE_MOVES,
    MININT,
    NOSTONE,
    Color,
    DIRECTIONS,
    GameResult,
    Move,
    Position,
    SearchStats,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_empty_board():
    board = [[0] * GRID_NUM for _ in range(GRID_NUM)]
    init_board(board)
    return board


def create_move(x1, y1, x2, y2):
    return Move((Position(x1, y1), Position(x2, y2)))


def create_move_at(x, y):
    return create_move(x, y, x, y)


def fill_board_no_line(board):
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            board[i][j] = Color.BLACK if (i + 2 * j) % 5 < 2 else Color.WHITE


# ============================================================
# BOARD TESTS
# ============================================================

def test_empty_board():
    board = create_empty_board()
    assert _is_board_full(board) == False
    print("Board Test 1 - Empty board: PASS")


def test_one_stone():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    assert _is_board_full(board) == False
    print("Board Test 2 - One stone: PASS")


def test_almost_full_board():
    board = create_empty_board()
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            board[i][j] = Color.BLACK
    board[10][10] = NOSTONE
    assert _is_board_full(board) == False
    print("Board Test 3 - Almost full board: PASS")


def test_full_board():
    board = create_empty_board()
    for i in range(1, GRID_NUM - 1):
        for j in range(1, GRID_NUM - 1):
            board[i][j] = Color.BLACK
    assert _is_board_full(board) == True
    print("Board Test 4 - Full board: PASS")


# ============================================================
# WIN TESTS
# ============================================================

def test_horizontal_win():
    board = create_empty_board()
    for y in range(5, 11):
        board[10][y] = Color.BLACK
    assert check_game_end(board, create_move_at(10, 10)) == GameResult.WIN
    print("Win Test 1 - Horizontal 6: PASS")


def test_vertical_win():
    board = create_empty_board()
    for x in range(5, 11):
        board[x][10] = Color.BLACK
    assert check_game_end(board, create_move_at(10, 10)) == GameResult.WIN
    print("Win Test 2 - Vertical 6: PASS")


def test_diagonal_win():
    board = create_empty_board()
    for i in range(6):
        board[5 + i][5 + i] = Color.BLACK
    assert check_game_end(board, create_move_at(5, 5)) == GameResult.WIN
    print("Win Test 3 - Diagonal 6: PASS")


def test_anti_diagonal_win():
    board = create_empty_board()
    for i in range(6):
        board[5 + i][10 - i] = Color.BLACK
    assert check_game_end(board, create_move_at(5, 10)) == GameResult.WIN
    print("Win Test 4 - Anti-diagonal 6: PASS")


def test_five_not_win():
    board = create_empty_board()
    for y in range(5, 10):
        board[10][y] = Color.BLACK
    assert check_game_end(board, create_move_at(10, 9)) is None
    print("Win Test 5 - Five in a row: PASS")


def test_seven_win():
    board = create_empty_board()
    for y in range(4, 11):
        board[10][y] = Color.BLACK
    assert check_game_end(board, create_move_at(10, 10)) == GameResult.WIN
    print("Win Test 6 - Seven in a row: PASS")


def test_broken_sequence():
    board = create_empty_board()
    for y in range(5, 8):
        board[10][y] = Color.BLACK
    board[10][8] = Color.WHITE
    for y in range(9, 11):
        board[10][y] = Color.BLACK
    assert check_game_end(board, create_move_at(10, 10)) is None
    print("Win Test 7 - Broken sequence: PASS")


def test_white_win():
    board = create_empty_board()
    for y in range(5, 11):
        board[10][y] = Color.WHITE
    assert check_game_end(board, create_move_at(10, 10)) == GameResult.WIN
    assert _is_win_by_move(board, create_move_at(10, 10)) == True
    print("Win Test 8 - White win: PASS")


# ============================================================
# DRAW TESTS
# ============================================================

def test_full_board_draw():
    board = create_empty_board()
    fill_board_no_line(board)
    assert check_game_end(board, create_move_at(10, 10)) == GameResult.DRAW
    print("Draw Test 1 - Full board: PASS")


def test_win_beats_draw():
    board = create_empty_board()
    fill_board_no_line(board)
    for y in range(2, 8):
        board[5][y] = Color.BLACK
    assert check_game_end(board, create_move_at(5, 7)) == GameResult.WIN
    print("Draw Test 2 - Win beats draw: PASS")


# ============================================================
# COLOR TESTS
# ============================================================

def test_color_names():
    assert color_to_name(Color.BLACK) == "black"
    assert color_to_name(Color.WHITE) == "white"
    print("Color Test 1 - Names: PASS")


def test_opponent():
    assert opponent(Color.BLACK) == Color.WHITE
    assert opponent(Color.WHITE) == Color.BLACK
    print("Color Test 2 - Opponent: PASS")


# ============================================================
# MOVE GENERATION TESTS
# ============================================================

def candidates_are_valid(board, candidates):
    for move in candidates:
        for p in move.positions:
            assert 0 < p.x < GRID_NUM - 1
            assert 0 < p.y < GRID_NUM - 1
            assert board[p.x][p.y] == NOSTONE
        assert move.positions[0].x != move.positions[1].x or move.positions[0].y != move.positions[1].y
    return True


def test_candidate_limit():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.WHITE)
    assert len(candidates) == MAX_CANDIDATE_MOVES
    assert candidates_are_valid(board, candidates)
    print("Search Test 1 - Fixed limit of candidates: PASS")


def test_candidates_skip_occupied():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.WHITE)
    for move in candidates:
        assert board[move.positions[0].x][move.positions[0].y] == NOSTONE
        assert board[move.positions[1].x][move.positions[1].y] == NOSTONE
    print("Search Test 2 - Skip occupied positions: PASS")


def test_candidates_near_stones():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.WHITE)
    first = candidates[0]
    assert abs(first.positions[0].x - 10) <= 1 and abs(first.positions[0].y - 10) <= 1
    print("Search Test 3 - Candidates ordered near stones: PASS")


def test_candidates_few_empties():
    board = create_empty_board()
    fill_board_no_line(board)
    board[5][2] = NOSTONE
    board[5][3] = NOSTONE
    board[5][4] = NOSTONE
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.WHITE)
    assert len(candidates) == 3
    print("Search Test 4 - Few empties produce all pairs: PASS")


def test_candidates_full_board():
    board = create_empty_board()
    fill_board_no_line(board)
    assert generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.WHITE) == []
    print("Search Test 5 - No candidates on full board: PASS")


def test_candidates_custom_limit():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    assert len(generate_candidate_moves(board, 5, Color.WHITE)) == 5
    print("Search Test 6 - Custom limit: PASS")


# ============================================================
# EVALUATION TESTS
# ============================================================

def test_evaluate_black_win():
    board = create_empty_board()
    for y in range(5, 11):
        board[10][y] = Color.BLACK
    assert evaluate(board, create_move_at(10, 10)) == MAXINT
    print("Eval Test 1 - Black win returns MAXINT: PASS")


def test_evaluate_white_win():
    board = create_empty_board()
    for y in range(5, 11):
        board[10][y] = Color.WHITE
    assert evaluate(board, create_move_at(10, 10)) == MININT
    print("Eval Test 2 - White win returns MININT: PASS")


def test_evaluate_draw():
    board = create_empty_board()
    fill_board_no_line(board)
    assert evaluate(board, create_move_at(10, 10)) == 0
    print("Eval Test 3 - Draw returns 0: PASS")


def test_evaluate_empty_board():
    board = create_empty_board()
    assert evaluate(board, create_move_at(10, 10)) == 0
    print("Eval Test 4 - Empty board is balanced: PASS")


def test_evaluate_black_advantage():
    board = create_empty_board()
    for y in range(3, 7):
        board[10][y] = Color.BLACK
    for y in range(3, 5):
        board[15][y] = Color.WHITE
    score = evaluate(board, create_move_at(10, 10))
    assert score > 0
    assert score > SetWeight.TRIPLE
    print("Eval Test 5 - Black advantage is positive: PASS")


def test_evaluate_white_advantage():
    board = create_empty_board()
    for y in range(3, 7):
        board[10][y] = Color.WHITE
    for y in range(3, 5):
        board[15][y] = Color.BLACK
    score = evaluate(board, create_move_at(10, 10))
    assert score < 0
    assert score < -SetWeight.TRIPLE
    print("Eval Test 6 - White advantage is negative: PASS")


def test_evaluate_mirrored_board():
    board = create_empty_board()
    for y in range(3, 6):
        board[5][y] = Color.BLACK
        board[15][y] = Color.WHITE
    assert evaluate(board, create_move_at(10, 10)) == 0
    print("Eval Test 7 - Mirrored board is balanced: PASS")


def test_dead_set_is_worthless():
    dead = create_empty_board()
    for y in range(8, 12):
        dead[10][y] = Color.BLACK
    dead[10][7] = Color.WHITE
    dead[10][12] = Color.WHITE
    open_board = create_empty_board()
    for y in range(8, 12):
        open_board[10][y] = Color.BLACK
    assert _score_living_sets(open_board, Color.BLACK) - _score_living_sets(dead, Color.BLACK) == 3 * SetWeight.LIVE_FOUR
    print("Eval Test 8 - Dead set contributes nothing: PASS")


def test_diagonal_equals_horizontal():
    horizontal = create_empty_board()
    for y in range(3, 6):
        horizontal[10][y] = Color.BLACK
    diagonal = create_empty_board()
    for i in range(3):
        diagonal[10 + i][4 + i] = Color.BLACK
    assert _score_living_sets(horizontal, Color.BLACK) == _score_living_sets(diagonal, Color.BLACK)
    assert _score_living_sets(horizontal, Color.BLACK) > SetWeight.TRIPLE
    print("Eval Test 9 - All line axes are covered equally: PASS")


def test_longer_set_worth_more():
    three = create_empty_board()
    for y in range(3, 6):
        three[10][y] = Color.BLACK
    two_pairs = create_empty_board()
    for y in range(3, 5):
        two_pairs[10][y] = Color.BLACK
        two_pairs[15][y] = Color.BLACK
    assert _score_living_sets(three, Color.BLACK) > _score_living_sets(two_pairs, Color.BLACK)
    print("Eval Test 10 - Longer sets outweigh more shorter ones: PASS")


def test_spacious_set_outvalues_cramped_set():
    spacious = create_empty_board()
    for y in range(8, 12):
        spacious[10][y] = Color.BLACK
    cramped = create_empty_board()
    for y in range(8, 12):
        cramped[10][y] = Color.BLACK
    cramped[10][7] = Color.WHITE
    cramped[10][14] = Color.WHITE
    assert _score_living_sets(spacious, Color.BLACK) - _score_living_sets(cramped, Color.BLACK) == 2 * SetWeight.LIVE_FOUR
    print("Eval Test 11 - Spacious sets outweigh cramped ones: PASS")


def test_window_factor_caps():
    three_windows = create_empty_board()
    for y in range(8, 12):
        three_windows[10][y] = Color.BLACK
    three_windows[10][7] = Color.WHITE
    three_windows[10][16] = Color.WHITE
    many_windows = create_empty_board()
    for y in range(8, 12):
        many_windows[10][y] = Color.BLACK
    assert _score_living_sets(three_windows, Color.BLACK) == _score_living_sets(many_windows, Color.BLACK)
    print("Eval Test 12 - The window factor caps at three windows: PASS")


# ============================================================
# CANDIDATE RANKING TESTS
# ============================================================

def pair_set(move):
    return frozenset(placed_cells(move))


def test_colinear_pair_ranks_below_split_pair():
    board = create_empty_board()
    for y in list(range(3, 6)) + list(range(9, 12)):
        board[10][y] = Color.WHITE
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.BLACK)
    pairs_in_order = [pair_set(move) for move in candidates]
    clear_gap = frozenset({(10, 6), (10, 8)})
    assert pairs_in_order.index(frozenset({(10, 2), (10, 12)})) < pairs_in_order.index(clear_gap)
    assert pairs_in_order.index(frozenset({(10, 2), (10, 6)})) < pairs_in_order.index(clear_gap)
    assert pairs_in_order.index(frozenset({(10, 12), (10, 8)})) < pairs_in_order.index(clear_gap)
    print("Candidate Ranking Test 1 - Co-linear pairs rank below split pairs: PASS")


def test_colinear_correction_requires_clear_gap():
    board = create_empty_board()
    clear = _pair_correction(board, (100, 10, 4), (60, 10, 6))
    diagonal = _pair_correction(board, (100, 10, 10), (60, 13, 13))
    board[10][5] = Color.BLACK
    blocked = _pair_correction(board, (100, 10, 4), (60, 10, 6))
    assert clear == 130
    assert diagonal == 130
    assert blocked == 160
    print("Candidate Ranking Test 2 - Correction requires a clear gap: PASS")


def test_colinear_correction_stops_at_twelve():
    board = create_empty_board()
    near = _pair_correction(board, (100, 10, 2), (60, 10, 8))
    mid = _pair_correction(board, (100, 10, 2), (60, 10, 9))
    late = _pair_correction(board, (100, 10, 2), (60, 10, 13))
    far = _pair_correction(board, (100, 10, 2), (60, 10, 14))
    off_axis = _pair_correction(board, (100, 10, 2), (60, 12, 9))
    assert near == 130
    assert mid == 145
    assert late == 145
    assert far == 160
    assert off_axis == 160
    print("Candidate Ranking Test 3 - Correction fades with distance: PASS")


# ============================================================
# THREAT TESTS
# ============================================================

BLACK_STONE = Color.BLACK.value
DEFENDER = Color.WHITE.value


def test_live_five_open_is_two_threats():
    assert count_line_threats([0, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0], BLACK_STONE) == 2
    assert count_line_threats([0, 0, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0, 0], BLACK_STONE) == 2
    print("Threat Test 1 - An open live five forces two blocks: PASS")


def test_live_five_blocked_is_one_threat():
    assert count_line_threats([DEFENDER, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0], BLACK_STONE) == 1
    print("Threat Test 2 - A blocked live five forces one block: PASS")


def test_open_live_four_is_two_threats():
    assert count_line_threats([0, 0, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0, 0], BLACK_STONE) == 2
    print("Threat Test 3 - An open live four forces two blocks: PASS")


def test_dead_live_four_is_one_threat():
    assert count_line_threats([DEFENDER, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0, 0], BLACK_STONE) == 1
    print("Threat Test 4 - A dead live four forces one block: PASS")


def test_completed_six_is_threat_win():
    assert count_line_threats([0, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0], BLACK_STONE) == THREAT_WIN
    print("Threat Test 5 - A completed six is the win sentinel: PASS")


def test_overlapping_windows_count_once():
    assert count_line_threats([0, 0, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, BLACK_STONE, 0, 0], BLACK_STONE) == 2
    print("Threat Test 6 - Overlapping windows share one defence: PASS")


def test_quiet_slice_has_no_threats():
    assert count_line_threats([0] * 11, BLACK_STONE) == 0
    assert count_line_threats([DEFENDER, DEFENDER, 0, DEFENDER, DEFENDER, 0, DEFENDER], BLACK_STONE) == 0
    print("Threat Test 7 - Quiet lines hold no threats: PASS")


def create_bridge_board():
    board = create_empty_board()
    for y in (3, 4, 5, 9, 10, 11):
        board[10][y] = Color.BLACK
    return board


def test_joint_threats_share_defence():
    board = create_bridge_board()
    alone_near = count_move_threats(board, 10, 6, 15, 15, Color.BLACK)
    alone_far = count_move_threats(board, 10, 7, 15, 15, Color.BLACK)
    joint = count_move_threats(board, 10, 6, 10, 7, Color.BLACK)
    assert alone_near == 2
    assert alone_far == 2
    assert joint == 2
    print("Threat Test 8 - Joint threats share one defence, not two: PASS")


def create_fork_board():
    board = create_empty_board()
    for k in range(3, 6):
        board[10][k] = Color.BLACK
        board[k][10] = Color.BLACK
    for k in range(3, 6):
        board[15][k] = Color.WHITE
    return board


def test_fork_creates_four_threats():
    board = create_fork_board()
    assert count_move_threats(board, 10, 6, 6, 10, Color.BLACK) == 4
    print("Threat Test 9 - A perpendicular fork is four threats: PASS")


def test_double_threat_pair_outranks_higher_sum_pair():
    board = create_empty_board()
    for k in range(3, 6):
        board[10][k] = Color.BLACK
        board[k][10] = Color.BLACK
        board[15][k] = Color.WHITE
    board[10][2] = Color.WHITE
    board[2][10] = Color.WHITE
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.BLACK)
    pairs_in_order = [pair_set(move) for move in candidates]
    fork = frozenset({(10, 6), (6, 10)})
    quiet = frozenset({(15, 2), (15, 6)})
    assert pairs_in_order.index(fork) < pairs_in_order.index(quiet)
    print("Threat Test 10 - A double-threat pair outranks a higher-sum pair: PASS")


# ============================================================
# MIN-MAX SEARCH TESTS
# ============================================================

def create_winnable_board():
    board = create_empty_board()
    for y in range(3, 8):
        board[10][y] = Color.BLACK
    board[9][7] = Color.BLACK
    board[11][7] = Color.BLACK
    board[9][8] = Color.BLACK
    board[11][8] = Color.BLACK
    return board


def placed_cells(move):
    return {
        (move.positions[0].x, move.positions[0].y),
        (move.positions[1].x, move.positions[1].y),
    }


def test_search_finds_winning_move():
    board = create_winnable_board()
    result = search_min_max(board, Color.BLACK, 2, create_move_at(10, 7), SearchStats())
    assert result.score == MAXINT
    assert (10, 8) in placed_cells(result.move)
    print("Search Test 7 - Min-max finds the winning move: PASS")


def test_search_blocks_threat():
    board = create_winnable_board()
    result = search_min_max(board, Color.WHITE, 2, create_move_at(10, 7), SearchStats())
    assert result.score < MAXINT
    assert (10, 8) in placed_cells(result.move)
    print("Search Test 8 - Min-max blocks the opponent's winning move: PASS")


def test_search_depth_grows_nodes():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    counts = []
    for depth in (1, 2, 3):
        stats = SearchStats()
        search_min_max(board, Color.WHITE, depth, create_move_at(10, 10), stats)
        counts.append(stats.node_count)
    assert counts[0] < counts[1] < counts[2]
    print("Search Test 9 - Deeper searches explore more nodes: PASS")


def create_threat_board(run_length):
    board = create_empty_board()
    for y in range(9, 9 + run_length):
        board[9][y] = Color.WHITE
    board[12][12] = Color.BLACK
    board[12][13] = Color.BLACK
    return board


def color_can_complete_six(board, color):
    for x in range(1, GRID_NUM - 6):
        for y in range(1, GRID_NUM - 6):
            for dx, dy in DIRECTIONS:
                if not (1 <= x + 5 * dx <= GRID_NUM - 2 and 1 <= y + 5 * dy <= GRID_NUM - 2):
                    continue
                stones = empties = 0
                blocked = False
                for k in range(6):
                    cell = board[x + k * dx][y + k * dy]
                    if cell == color:
                        stones += 1
                    elif cell == NOSTONE:
                        empties += 1
                    else:
                        blocked = True
                        break
                if not blocked and stones >= 4 and stones + empties == 6:
                    return True
    return False


def test_candidates_include_completion_pairs():
    board = create_threat_board(4)
    candidates = generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.BLACK)
    pairs = {frozenset(placed_cells(move)) for move in candidates}
    assert frozenset({(9, 13), (9, 14)}) in pairs
    print("Search Test 10 - Two-stone completion pairs are generated: PASS")


def test_search_blocks_double_threat():
    board = create_threat_board(5)
    result = search_min_max(board, Color.BLACK, 2, create_move_at(9, 13), SearchStats())
    make_move(board, result.move, Color.BLACK)
    assert not color_can_complete_six(board, Color.WHITE)
    print("Search Test 11 - Min-max blocks a live five on both ends: PASS")


def test_search_blocks_two_stone_threat():
    board = create_threat_board(4)
    result = search_min_max(board, Color.BLACK, 2, create_move_at(9, 12), SearchStats())
    make_move(board, result.move, Color.BLACK)
    assert not color_can_complete_six(board, Color.WHITE)
    print("Search Test 12 - Min-max blocks a live four: PASS")


def test_search_and_candidates_do_not_mutate_board():
    board = create_threat_board(4)
    snapshot = copy_board(board)
    generate_candidate_moves(board, MAX_CANDIDATE_MOVES, Color.BLACK)
    assert board == snapshot
    search_min_max(board, Color.BLACK, 2, create_move_at(9, 13), SearchStats())
    assert board == snapshot
    print("Search Test 13 - Search and candidates leave the input board untouched: PASS")


def test_alpha_beta_finds_winning_move():
    board = create_winnable_board()
    result = search_alpha_beta(board, Color.BLACK, 2, create_move_at(10, 7), SearchStats())
    assert result.score == MAXINT
    assert (10, 8) in placed_cells(result.move)
    print("Search Test 14 - Alpha-beta finds the winning move: PASS")


def test_alpha_beta_matches_min_max():
    for threat, pre in ((4, (9, 12)), (5, (9, 13))):
        board = create_threat_board(threat)
        pre_move = create_move_at(*pre)
        naive = search_min_max(board, Color.BLACK, 2, pre_move, SearchStats())
        pruned = search_alpha_beta(board, Color.BLACK, 2, pre_move, SearchStats())
        assert pruned.score == naive.score
        make_move(board, pruned.move, Color.BLACK)
        assert not color_can_complete_six(board, Color.WHITE)
    print("Search Test 15 - Alpha-beta matches min-max on score and forced blocks: PASS")


def test_alpha_beta_visits_fewer_nodes():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    pre_move = create_move_at(10, 10)
    naive_stats = SearchStats()
    pruned_stats = SearchStats()
    search_min_max(board, Color.WHITE, 3, pre_move, naive_stats)
    search_alpha_beta(board, Color.WHITE, 3, pre_move, pruned_stats)
    assert pruned_stats.node_count < naive_stats.node_count
    print("Search Test 16 - Alpha-beta visits fewer nodes: PASS")


def create_forked_board():
    board = create_fork_board()
    board[10][6] = Color.BLACK
    board[6][10] = Color.BLACK
    return board


def create_quiet_board():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    board[3][3] = Color.BLACK
    board[10][3] = Color.WHITE
    board[3][10] = Color.WHITE
    return board


# ============================================================
# ENHANCED SEARCH TESTS (QUIESCENCE + FORWARD PRUNING)
# ============================================================

def test_quiescence_finds_win_beyond_horizon():
    board = create_fork_board()
    pre_move = create_move_at(15, 5)
    plain = search_alpha_beta(board, Color.BLACK, 2, pre_move, SearchStats())
    enhanced = search_alpha_beta_with_quiescence(board, Color.BLACK, 2, pre_move, SearchStats())
    assert plain.score < MAXINT
    assert enhanced.score == MAXINT
    fork = enhanced.move
    threats = count_move_threats(
        board,
        fork.positions[0].x,
        fork.positions[0].y,
        fork.positions[1].x,
        fork.positions[1].y,
        Color.BLACK,
    )
    assert threats >= 3
    print("Enhanced Search Test 1 - Quiescence finds the fork win one ply past the horizon: PASS")


def test_quiescence_prices_hopeless_defence():
    board = create_forked_board()
    pre_move = create_move_at(6, 10)
    plain = search_alpha_beta(board, Color.WHITE, 1, pre_move, SearchStats())
    enhanced = search_alpha_beta_with_quiescence(board, Color.WHITE, 1, pre_move, SearchStats())
    assert plain.score < MAXINT
    assert enhanced.score == MAXINT
    print("Enhanced Search Test 2 - Quiescence prices a lost defence as a win for the forker: PASS")


def test_enhanced_finds_winning_move():
    board = create_winnable_board()
    result = search_alpha_beta_with_quiescence(board, Color.BLACK, 2, create_move_at(10, 7), SearchStats())
    assert result.score == MAXINT
    assert (10, 8) in placed_cells(result.move)
    print("Enhanced Search Test 3 - Enhanced search finds the winning move: PASS")


def test_enhanced_blocks_opponent_win():
    board = create_winnable_board()
    result = search_alpha_beta_with_quiescence(board, Color.WHITE, 2, create_move_at(10, 7), SearchStats())
    assert result.score == MAXINT
    assert placed_cells(result.move) == {(10, 8), (10, 2)}
    print("Enhanced Search Test 4 - Enhanced search blocks both ends and still prices the position lost: PASS")


def test_enhanced_blocks_live_four():
    board = create_threat_board(4)
    result = search_alpha_beta_with_quiescence(board, Color.BLACK, 2, create_move_at(9, 12), SearchStats())
    make_move(board, result.move, Color.BLACK)
    assert not color_can_complete_six(board, Color.WHITE)
    print("Enhanced Search Test 5 - Enhanced search blocks a live four: PASS")


def test_enhanced_blocks_live_five():
    board = create_threat_board(5)
    result = search_alpha_beta_with_quiescence(board, Color.BLACK, 2, create_move_at(9, 13), SearchStats())
    make_move(board, result.move, Color.BLACK)
    assert not color_can_complete_six(board, Color.WHITE)
    print("Enhanced Search Test 6 - Enhanced search blocks a live five on both ends: PASS")


def test_forward_pruning_reduces_nodes_on_quiet_board():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    plain_stats = SearchStats()
    enhanced_stats = SearchStats()
    search_alpha_beta(board, Color.WHITE, 4, pre_move, plain_stats)
    search_alpha_beta_with_quiescence(board, Color.WHITE, 4, pre_move, enhanced_stats)
    assert enhanced_stats.node_count < plain_stats.node_count
    print("Enhanced Search Test 7 - Forward pruning visits fewer nodes at depth four on a quiet board: PASS")


# ============================================================
# V5 SEARCH TESTS (VCF + BUDGETED DEEPENING)
# ============================================================

def test_vcf_finds_fork_win():
    board = create_fork_board()
    stats = SearchStats()
    win = find_forced_win(board, Color.BLACK, stats)
    assert win is not None
    threats = count_move_threats(
        board,
        win.positions[0].x,
        win.positions[0].y,
        win.positions[1].x,
        win.positions[1].y,
        Color.BLACK,
    )
    assert threats >= 3
    assert stats.node_count <= 100
    print("V5 Test 1 - VCF proves the fork position a forced win without a tree search: PASS")


def test_vcf_completes_won_position():
    board = create_forked_board()
    stats = SearchStats()
    win = find_forced_win(board, Color.BLACK, stats)
    assert win is not None
    assert stats.node_count == 1
    make_move(board, win, Color.BLACK)
    first = win.positions[0]
    assert check_game_end(board, create_move_at(first.x, first.y)) == GameResult.WIN
    print("V5 Test 2 - VCF finishes the completed fork in one node: PASS")


def test_vcf_rejects_quiet_position():
    board = create_quiet_board()
    for color in (Color.BLACK, Color.WHITE):
        stats = SearchStats()
        assert find_forced_win(board, color, stats) is None
        assert stats.node_count == 1
    print("V5 Test 3 - VCF rejects a quiet position after one scan: PASS")


def test_search_v5_plays_the_forced_win():
    board = create_fork_board()
    stats = SearchStats()
    result = search_v5(board, Color.BLACK, 3, create_move_at(15, 5), stats)
    assert result.score == MAXINT
    assert stats.node_count <= 100
    win = find_forced_win(board, Color.BLACK, SearchStats())
    assert win is not None
    assert placed_cells(result.move) == placed_cells(win)
    print("V5 Test 4 - The v5 root plays the VCF win without building a tree: PASS")


def test_search_v5_keeps_the_forced_block():
    board = create_winnable_board()
    stats = SearchStats()
    result = search_v5(board, Color.WHITE, 3, create_move_at(10, 7), stats)
    assert placed_cells(result.move) == {(10, 8), (10, 2)}
    work = copy_board(board)
    make_move(work, result.move, Color.WHITE)
    assert find_forced_win(work, Color.BLACK, SearchStats()) is None
    print("V5 Test 5 - The v5 root keeps the forced five-run block and the screen confirms it: PASS")


def test_search_v5_deepens_under_budget():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    single = SearchStats()
    search_alpha_beta_with_quiescence(board, Color.WHITE, 2, pre_move, single)
    deep = SearchStats()
    search_v5(board, Color.WHITE, 2, pre_move, deep)
    assert deep.node_count > single.node_count
    print("V5 Test 6 - Budgeted deepening looks past the floor depth: PASS")


def test_search_v5_is_deterministic():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    first_stats, second_stats = SearchStats(), SearchStats()
    first = search_v5(board, Color.WHITE, 2, pre_move, first_stats)
    second = search_v5(board, Color.WHITE, 2, pre_move, second_stats)
    assert placed_cells(first.move) == placed_cells(second.move)
    assert first.score == second.score
    assert first_stats.node_count == second_stats.node_count
    print("V5 Test 7 - Identical inputs give identical v5 results: PASS")


def test_search_v5_without_vcf_still_finds_the_fork():
    board = create_fork_board()
    stats = SearchStats()
    result = search_v5(board, Color.BLACK, 3, create_move_at(15, 5), stats, use_vcf=False)
    assert result.score == MAXINT
    threats = count_move_threats(
        board,
        result.move.positions[0].x,
        result.move.positions[0].y,
        result.move.positions[1].x,
        result.move.positions[1].y,
        Color.BLACK,
    )
    assert threats >= 3
    print("V5 Test 8 - With VCF off the deepened search still finds the fork: PASS")


def test_vcf_toggles_flag():
    engine = GameEngine()
    assert engine.use_vcf is True
    engine._cmd_unvcf("")
    assert engine.use_vcf is False
    engine._cmd_vcf("")
    assert engine.use_vcf is True
    print("V5 Test 9 - vcf/unvcf toggle the forced-win stage: PASS")


def create_split_board():
    board = create_empty_board()
    board[10][4] = Color.WHITE
    board[10][7] = Color.WHITE
    board[13][13] = Color.BLACK
    board[13][14] = Color.BLACK
    return board


def test_search_v5_preempts_armed_split():
    board = create_split_board()
    stats = SearchStats()
    result = search_v5(board, Color.BLACK, 3, create_move_at(13, 14), stats)
    cells = placed_cells(result.move)
    assert cells & {(10, 5), (10, 6)}, cells
    work = copy_board(board)
    make_move(work, result.move, Color.BLACK)
    assert find_forced_win(work, Color.WHITE, SearchStats()) is None
    print("V5 Test 10 - The screen preempts an armed split pair instead of developing: PASS")


# ============================================================
# V6 SEARCH TESTS (PARALLEL ROOT SPLIT + WALL-TIME BUDGET)
# ============================================================

def test_v6_split_matches_shared_window():
    cases = [
        (create_quiet_board(), Color.WHITE, create_move_at(3, 10)),
        (create_fork_board(), Color.BLACK, create_move_at(15, 5)),
        (create_split_board(), Color.BLACK, create_move_at(13, 14)),
        (create_winnable_board(), Color.WHITE, create_move_at(10, 7)),
    ]
    for board, color, pre_move in cases:
        shared = search_alpha_beta_with_quiescence(board, color, 3, pre_move, SearchStats())
        candidates, threats = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], color)
        maximizing = color == Color.BLACK
        best_score = MININT if maximizing else MAXINT
        best_move = candidates[0]
        for candidate, threat in zip(candidates, threats):
            result = search_candidate_subtree(board, color, 3, candidate, threat, SearchStats())
            if maximizing and result.score > best_score:
                best_score, best_move = result.score, candidate
            if not maximizing and result.score < best_score:
                best_score, best_move = result.score, candidate
        assert placed_cells(best_move) == placed_cells(shared.move), (color, pre_move)
        assert best_score == shared.score, (color, pre_move)
    print("V6 Test 1 - The root split picks the shared-window root's move and score at equal depth: PASS")


def test_search_v6_plays_the_forced_win():
    board = create_fork_board()
    stats = SearchStats()
    result = search_v6(board, Color.BLACK, 3, create_move_at(15, 5), stats)
    win = find_forced_win(board, Color.BLACK, SearchStats())
    assert result.score == MAXINT
    assert stats.node_count <= 100
    assert win is not None
    assert placed_cells(result.move) == placed_cells(win)
    print("V6 Test 2 - The v6 root plays the VCF win without building a tree: PASS")


def test_search_v6_fixed_agrees_with_v5():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    v5_stats, v6_stats = SearchStats(), SearchStats()
    v5 = search_v5(board, Color.WHITE, 2, pre_move, v5_stats)
    os.environ[V6_JOBS_ENV] = "1"
    try:
        v6 = search_v6(board, Color.WHITE, 2, pre_move, v6_stats)
    finally:
        os.environ[V6_JOBS_ENV] = "2"
    assert placed_cells(v6.move) == placed_cells(v5.move)
    assert v6.score == v5.score
    assert v6_stats.depth_reached == 4
    print("V6 Test 3 - Fixed v6 agrees with v5 where both deepening budgets allow the same depths: PASS")


def test_search_v6_is_jobs_invariant():
    board = create_winnable_board()
    pre_move = create_move_at(10, 7)
    stats_one, stats_again, stats_pooled = SearchStats(), SearchStats(), SearchStats()
    os.environ[V6_JOBS_ENV] = "1"
    try:
        first = search_v6(board, Color.WHITE, 3, pre_move, stats_one)
        second = search_v6(board, Color.WHITE, 3, pre_move, stats_again)
    finally:
        os.environ[V6_JOBS_ENV] = "3"
    try:
        third = search_v6(board, Color.WHITE, 3, pre_move, stats_pooled)
    finally:
        os.environ[V6_JOBS_ENV] = "2"
    assert placed_cells(first.move) == placed_cells(second.move) == placed_cells(third.move)
    assert first.score == second.score == third.score
    assert stats_one.node_count == stats_again.node_count == stats_pooled.node_count
    print("V6 Test 4 - The answer is identical at one, three and pooled workers: PASS")


def test_search_v6_budget_answers_within_the_cap():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    stats = SearchStats()
    budget = 0.25
    started = time.perf_counter()
    result = search_v6(board, Color.WHITE, 3, pre_move, stats, True, budget)
    wall = time.perf_counter() - started
    assert wall <= budget + 1.0, wall
    work = copy_board(board)
    assert all(work[x][y] == NOSTONE for x, y in placed_cells(result.move))
    assert stats.depth_reached >= 3
    print("V6 Test 5 - A tight budget still answers with a legal move within the cap: PASS")


def test_search_v6_budget_deepens_past_the_floor():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    stats = SearchStats()
    started = time.perf_counter()
    search_v6(board, Color.WHITE, 2, pre_move, stats, True, 8.0)
    wall = time.perf_counter() - started
    assert stats.depth_reached >= 3, stats.depth_reached
    assert wall <= 12.0, wall
    print("V6 Test 6 - The wall-time budget deepens past the floor depth while it fits: PASS")


def test_search_v6_without_vcf_still_finds_the_fork():
    board = create_fork_board()
    stats = SearchStats()
    result = search_v6(board, Color.BLACK, 3, create_move_at(15, 5), stats, use_vcf=False)
    assert result.score == MAXINT
    threats = count_move_threats(
        board,
        result.move.positions[0].x,
        result.move.positions[0].y,
        result.move.positions[1].x,
        result.move.positions[1].y,
        Color.BLACK,
    )
    assert threats >= 3
    print("V6 Test 7 - With VCF off the deepened search still finds the fork: PASS")


def test_search_v6_preempts_armed_split():
    board = create_split_board()
    stats = SearchStats()
    result = search_v6(board, Color.BLACK, 3, create_move_at(13, 14), stats)
    work = copy_board(board)
    make_move(work, result.move, Color.BLACK)
    assert find_forced_win(work, Color.WHITE, SearchStats()) is None
    print("V6 Test 8 - The parallel screen never hands the opponent a proven forced win: PASS")


def test_search_v6_budget_enforces_the_deadline_on_fork_heavy_positions():
    battery_path = Path(__file__).resolve().parents[1] / "benchmarks" / "battery.json"
    fixture_ids = ("win-2-seedJJ", "win-3-seedHIKK", "defend-2-double-threat", "defend-3-fork-build")
    picked = [entry for entry in json.loads(battery_path.read_text()) if entry["id"] in fixture_ids]
    assert len(picked) == len(fixture_ids)
    budget = 2.0
    slowest = 0.0
    os.environ[V6_JOBS_ENV] = "1"
    try:
        for entry in picked:
            board = create_empty_board()
            pre_move = Move((Position(0, 0), Position(0, 0)))
            color = Color.BLACK
            for msg in entry["history"]:
                move = msg2move(msg)
                make_move(board, move, color)
                pre_move = move
                color = opponent(color)
            stats = SearchStats()
            started = time.perf_counter()
            result = search_v6(board, color, 2, pre_move, stats, True, budget)
            wall = time.perf_counter() - started
            work = copy_board(board)
            assert all(work[x][y] == NOSTONE for x, y in placed_cells(result.move)), entry["id"]
            assert wall <= budget + 1.0, (entry["id"], wall)
            slowest = max(slowest, wall)
    finally:
        os.environ[V6_JOBS_ENV] = "2"
    print(f"V6 Test 9 - Fork-heavy battery positions answer within the deadline cap (slowest {slowest:.2f}s): PASS")


def test_search_v6_budget_ladders_to_a_deep_floor():
    board = create_quiet_board()
    pre_move = create_move_at(3, 10)
    stats = SearchStats()
    budget = 0.5
    started = time.perf_counter()
    result = search_v6(board, Color.WHITE, 9, pre_move, stats, True, budget)
    wall = time.perf_counter() - started
    work = copy_board(board)
    assert all(work[x][y] == NOSTONE for x, y in placed_cells(result.move))
    assert wall <= budget + 1.0, wall
    assert 2 <= stats.depth_reached <= 9, stats.depth_reached
    assert MININT < result.score < MAXINT, result.score
    print("V6 Test 10 - A floor too deep for the budget is climbed to, not started from: PASS")


def test_search_candidate_subtree_aborts_at_the_deadline():
    board = create_quiet_board()
    candidates, threats = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], Color.WHITE)
    candidate, incoming = candidates[0], threats[0]
    stats = SearchStats()
    started = time.perf_counter()
    raised = False
    try:
        search_candidate_subtree(board, Color.WHITE, 9, candidate, incoming, stats, time.perf_counter() + 0.001)
    except SearchDeadlineExceeded:
        raised = True
    wall = time.perf_counter() - started
    assert raised
    assert wall < 1.0, wall
    fresh = SearchStats()
    result = search_candidate_subtree(board, Color.WHITE, 3, candidate, incoming, fresh)
    assert MININT < result.score < MAXINT, result.score
    print("V6 Test 11 - The subtree recursion aborts at an expired deadline and completes without one: PASS")


# ============================================================
# OPENING BOOK TESTS
# ============================================================

def test_position_key_is_stone_inventory():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    board[3][3] = Color.WHITE
    assert position_key(board) == "3.3.2,10.10.1"
    twin = create_empty_board()
    twin[3][3] = Color.WHITE
    twin[10][10] = Color.BLACK
    assert position_key(twin) == position_key(board)
    print("Opening Book Test 1 - Position key is a stone inventory: PASS")


def test_book_move_returns_recorded_move():
    board = create_empty_board()
    board[10][10] = Color.BLACK
    book = {position_key(board): "AABB"}
    move = book_move(board, book)
    assert move is not None
    assert placed_cells(move) == {(19, 1), (18, 2)}
    print("Opening Book Test 2 - Book lookup returns the recorded move: PASS")


def test_book_move_rejects_occupied_and_invalid():
    board = create_empty_board()
    first = msg2move("AABB").positions[0]
    board[first.x][first.y] = Color.BLACK
    assert book_move(board, {}) is None
    assert book_move(board, {"nope": "AABB"}) is None
    assert book_move(board, {position_key(board): "ZZZZ"}) is None
    assert book_move(board, {position_key(board): "AABB"}) is None
    print("Opening Book Test 3 - Book lookup rejects missing, malformed and occupied entries: PASS")


def test_engine_plays_book_move():
    scratch = create_empty_board()
    for stone in msg2move("JILL").positions:
        scratch[stone.x][stone.y] = Color.BLACK
    engine = GameEngine()
    engine._cmd_new("new white")
    engine._book = {position_key(scratch): "AABB"}
    result, output = capture_output(lambda: engine._cmd_move("move JILL"))
    assert result is True
    assert "Book:\tAABB" in output
    assert "move AABB" in output
    assert "Search Time" not in output
    assert engine.board[19][1] == Color.WHITE
    assert engine.board[18][2] == Color.WHITE
    print("Opening Book Test 4 - The engine plays a recorded book move: PASS")


def test_engine_falls_back_to_search_without_book_entry():
    engine = GameEngine()
    engine._cmd_new("new white")
    engine._book = {}
    result, output = capture_output(lambda: engine._cmd_move("move JILL"))
    assert result is True
    assert "Search Time" in output
    reply = next(line for line in output.splitlines() if line.startswith("move "))
    assert len(reply[5:]) == 4
    assert all("A" <= c <= "S" for c in reply[5:])
    print("Opening Book Test 5 - Without a book entry the engine searches: PASS")


# ============================================================
# PROTOCOL TESTS
# ============================================================

class ProtocolEngine:
    """Drives the engine binary over raw pipes: line-based commands,
    replies found by prefix scanning."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(PROJECT_ROOT, "main.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=PROJECT_ROOT,
            start_new_session=True,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self.stdin = self.proc.stdin
        self.stdout = self.proc.stdout
        self.stderr = self.proc.stderr
        self.lines = []
        self.sent = []

    def send(self, cmd):
        self.sent.append(cmd)
        self.stdin.write((cmd + "\n").encode())

    def next_line(self):
        ready, _, _ = select.select([self.stdout], [], [], 15.0)
        assert ready, f"no engine output within 15s after commands: {self.sent[-3:]}"
        line = self.stdout.readline().decode()
        assert line, "engine exited unexpectedly"
        line = line.rstrip("\n")
        self.lines.append(line)
        return line

    def read_until(self, prefix):
        while True:
            line = self.next_line()
            if line.startswith(prefix):
                return line

    def drain_startup(self):
        self.send("name")
        return self.read_until("name ")

    def close(self):
        self.stdin.close()
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait(timeout=10)
        return self.stderr.read().decode()


def capture_output(fn):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = fn()
    return result, buffer.getvalue()


def test_protocol_name_handshake():
    engine = ProtocolEngine()
    engine.send("name")
    reply = engine.read_until("name ")
    assert reply == f"name {ENGINE_NAME}"
    assert all(
        not line.startswith("name ") and not line.startswith("move ")
        for line in engine.lines[:-1]
    )
    engine.close()
    print("Protocol Test 1 - name handshake replies without prefix collisions: PASS")


def test_protocol_depth_configuration():
    engine = ProtocolEngine()
    engine.send("depth 2")
    assert engine.read_until("Set the search depth") == "Set the search depth to 2."
    engine.send("depth 0")
    assert engine.read_until("Set the search depth") == "Set the search depth to 2."
    engine.send("depth 99")
    assert engine.read_until("Set the search depth") == "Set the search depth to 2."
    engine.send("depth 1")
    assert engine.read_until("Set the search depth") == "Set the search depth to 1."
    engine.close()
    print("Protocol Test 2 - depth command configures the search depth: PASS")


def test_protocol_vcf_commands_are_silent():
    engine = ProtocolEngine()
    engine.drain_startup()
    engine.send("vcf")
    engine.send("unvcf")
    engine.send("name")
    assert engine.next_line() == f"name {ENGINE_NAME}"
    engine.close()
    print("Protocol Test 3 - vcf/unvcf are accepted without any output: PASS")


def test_protocol_gui_replay_flow():
    engine = ProtocolEngine()
    engine.send("depth 1")
    engine.read_until("Set the search depth")
    engine.send("new xxx")
    engine.send("black JIJK")
    engine.send("white KJKJ")
    engine.send("next")
    reply = engine.read_until("move ")
    coords = reply[5:]
    assert len(coords) == 4
    assert all("A" <= c <= "S" for c in coords)
    assert all(not line.startswith("move ") for line in engine.lines[:-1])
    engine.close()
    print("Protocol Test 4 - GUI replay (new xxx + places + next) gets a move: PASS")


def test_protocol_engine_first_center_opening():
    engine = ProtocolEngine()
    engine.send("new xxx")
    engine.send("next")
    assert engine.read_until("move ") == "move JJ"
    engine.close()
    print("Protocol Test 5 - engine-first game opens with 2-char 'move JJ': PASS")


def test_protocol_move_command_loop():
    engine = ProtocolEngine()
    engine.send("depth 1")
    engine.read_until("Set the search depth")
    engine.send("new white")
    engine.send("move JILL")
    first = engine.read_until("move ")
    assert all("A" <= c <= "S" for c in first[5:])
    engine.send("move ABAF")
    second = engine.read_until("move ")
    assert all("A" <= c <= "S" for c in second[5:])
    engine.close()
    print("Protocol Test 6 - move command loop keeps the game going: PASS")


def test_protocol_announces_human_win():
    engine = GameEngine()
    engine._cmd_new("new white")
    for y in range(5, 10):
        engine.board[10][y] = Color.BLACK.value
    human = move2msg(Move((Position(10, 4), Position(15, 15))))
    result, output = capture_output(lambda: engine._cmd_move(f"move {human}"))
    assert result is False
    assert "Human wins with black!" in output
    print("Protocol Test 7 - human win ends the game with announcement: PASS")


def test_protocol_announces_ai_win():
    engine = GameEngine()
    human = move2msg(Move((Position(15, 15), Position(16, 16))))

    def scenario():
        engine._cmd_new("new black")
        engine._cmd_depth("depth 1")
        for y in range(5, 9):
            engine.board[5][y] = Color.BLACK.value
        return engine._cmd_move(f"move {human}")

    result, output = capture_output(scenario)
    assert result is False
    assert "AI wins with black!" in output
    print("Protocol Test 8 - AI win ends the game with announcement: PASS")


def test_msg2move_rejects_invalid_messages():
    assert msg2move("JJ").positions[0] == Position(10, 10)
    assert msg2move("JILL").positions[1] == Position(8, 12)
    for bad in ("", "X", "XYZ", "JJJ", "ZJ", "JT", "AAAAA"):
        try:
            msg2move(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"msg2move accepted {bad!r}")
    print("Protocol Test 9 - msg2move validates message format and coordinates: PASS")


def test_protocol_malformed_commands_do_not_kill_engine():
    log_path = os.path.join(PROJECT_ROOT, "tia-engine.log")
    log_size_before = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    engine = ProtocolEngine()
    engine.send("depth 1")
    engine.read_until("Set the search depth")
    assert engine.next_line() == ""
    engine.send("new white")
    for bad in ("depth abc", "move XYZ", "move ZZZZ", "move", "black", "nonsense"):
        engine.send(bad)
    engine.send("name")
    assert engine.next_line() == f"name {ENGINE_NAME}"
    engine.send("move JILL")
    reply = engine.read_until("move ")
    assert all("A" <= c <= "S" for c in reply[5:])
    errors = engine.close()
    for bad in ("depth abc", "move XYZ", "move ZZZZ", "move", "black"):
        assert f"Error processing '{bad}'" in errors
    assert "nonsense" not in errors
    with open(log_path) as log_file:
        log_file.seek(log_size_before)
        session_log = log_file.read()
    for bad in ("depth abc", "move XYZ", "move ZZZZ", "move", "black"):
        assert f"ERROR: Error processing '{bad}'" in session_log
    assert "- move JILL" in session_log
    print("Protocol Test 10 - malformed commands never kill the engine: PASS")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    print("\n========== BOARD TESTS ==========\n")
    test_empty_board()
    test_one_stone()
    test_almost_full_board()
    test_full_board()

    print("\n========== WIN TESTS ==========\n")
    test_horizontal_win()
    test_vertical_win()
    test_diagonal_win()
    test_anti_diagonal_win()
    test_five_not_win()
    test_seven_win()
    test_broken_sequence()
    test_white_win()

    print("\n========== DRAW TESTS ==========\n")
    test_full_board_draw()
    test_win_beats_draw()

    print("\n========== COLOR TESTS ==========\n")
    test_color_names()
    test_opponent()

    print("\n========== EVALUATION TESTS ==========\n")
    test_evaluate_black_win()
    test_evaluate_white_win()
    test_evaluate_draw()
    test_evaluate_empty_board()
    test_evaluate_black_advantage()
    test_evaluate_white_advantage()
    test_evaluate_mirrored_board()
    test_dead_set_is_worthless()
    test_diagonal_equals_horizontal()
    test_longer_set_worth_more()
    test_spacious_set_outvalues_cramped_set()
    test_window_factor_caps()

    print("\n========== CANDIDATE RANKING TESTS ==========\n")
    test_colinear_pair_ranks_below_split_pair()
    test_colinear_correction_requires_clear_gap()
    test_colinear_correction_stops_at_twelve()

    print("\n========== THREAT TESTS ==========\n")
    test_live_five_open_is_two_threats()
    test_live_five_blocked_is_one_threat()
    test_open_live_four_is_two_threats()
    test_dead_live_four_is_one_threat()
    test_completed_six_is_threat_win()
    test_overlapping_windows_count_once()
    test_quiet_slice_has_no_threats()
    test_joint_threats_share_defence()
    test_fork_creates_four_threats()
    test_double_threat_pair_outranks_higher_sum_pair()

    print("\n========== MIN-MAX SEARCH TESTS ==========\n")
    test_search_finds_winning_move()
    test_search_blocks_threat()
    test_search_depth_grows_nodes()
    test_candidates_include_completion_pairs()
    test_search_blocks_double_threat()
    test_search_blocks_two_stone_threat()
    test_search_and_candidates_do_not_mutate_board()
    test_alpha_beta_finds_winning_move()
    test_alpha_beta_matches_min_max()
    test_alpha_beta_visits_fewer_nodes()

    print("\n========== MOVE GENERATION TESTS ==========\n")
    test_candidate_limit()
    test_candidates_skip_occupied()
    test_candidates_near_stones()
    test_candidates_few_empties()
    test_candidates_full_board()
    test_candidates_custom_limit()

    print("\n========== ENHANCED SEARCH TESTS (QUIESCENCE + FORWARD PRUNING) ==========\n")
    test_quiescence_finds_win_beyond_horizon()
    test_quiescence_prices_hopeless_defence()
    test_enhanced_finds_winning_move()
    test_enhanced_blocks_opponent_win()
    test_enhanced_blocks_live_four()
    test_enhanced_blocks_live_five()
    test_forward_pruning_reduces_nodes_on_quiet_board()

    print("\n========== V5 SEARCH TESTS (VCF + BUDGETED DEEPENING) ==========\n")
    test_vcf_finds_fork_win()
    test_vcf_completes_won_position()
    test_vcf_rejects_quiet_position()
    test_search_v5_plays_the_forced_win()
    test_search_v5_keeps_the_forced_block()
    test_search_v5_deepens_under_budget()
    test_search_v5_is_deterministic()
    test_search_v5_without_vcf_still_finds_the_fork()
    test_vcf_toggles_flag()
    test_search_v5_preempts_armed_split()

    print("\n========== V6 SEARCH TESTS (PARALLEL ROOT SPLIT + WALL-TIME BUDGET) ==========\n")
    test_v6_split_matches_shared_window()
    test_search_v6_plays_the_forced_win()
    test_search_v6_fixed_agrees_with_v5()
    test_search_v6_is_jobs_invariant()
    test_search_v6_budget_answers_within_the_cap()
    test_search_v6_budget_deepens_past_the_floor()
    test_search_v6_without_vcf_still_finds_the_fork()
    test_search_v6_preempts_armed_split()
    test_search_v6_budget_enforces_the_deadline_on_fork_heavy_positions()
    test_search_v6_budget_ladders_to_a_deep_floor()
    test_search_candidate_subtree_aborts_at_the_deadline()

    print("\n========== OPENING BOOK TESTS ==========\n")
    test_position_key_is_stone_inventory()
    test_book_move_returns_recorded_move()
    test_book_move_rejects_occupied_and_invalid()
    test_engine_plays_book_move()
    test_engine_falls_back_to_search_without_book_entry()

    print("\n========== PROTOCOL TESTS ==========\n")
    test_protocol_name_handshake()
    test_protocol_depth_configuration()
    test_protocol_vcf_commands_are_silent()
    test_protocol_gui_replay_flow()
    test_protocol_engine_first_center_opening()
    test_protocol_move_command_loop()
    test_protocol_announces_human_win()
    test_protocol_announces_ai_win()
    test_msg2move_rejects_invalid_messages()
    test_protocol_malformed_commands_do_not_kill_engine()

    print("\n========== ALL TESTS PASSED ==========\n")