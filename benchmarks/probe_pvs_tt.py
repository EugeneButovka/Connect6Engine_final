"""Measure whether PVS and a transposition table pay: base, pvs, tt, warm
and both variants of the split-root subtrees, per trajectory decision point.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("WEIGHTS_FILE", str(REPO / "tuning" / "winner.json"))
os.environ["V6_JOBS"] = "1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tuning"))

from connect6.board import copy_board, make_move, opponent, unmake_move
from connect6.candidates import generate_scored_candidate_moves
from connect6.defines import (
    FORWARD_WIDTHS,
    MAXINT,
    MININT,
    QUIESCENCE_DEPTH,
    QUIESCENCE_MIN_THREATS,
    QUIESCENCE_WIDTH,
    THREAT_WIN,
    Color,
    Move,
    SearchStats,
    SetWeight,
)
from connect6.evaluation import evaluate
from connect6.search import search_candidate_subtree

EXACT, LOWER, UPPER = 0, 1, 2
V6_NODE_BUDGET = 36000
FLOOR = 3


@dataclass
class Counters:
    nodes: int = 0
    q_nodes: int = 0
    hits: int = 0
    scouts: int = 0
    researches: int = 0


def _advance(maximizing, child_score, candidate, best_score, best_move, alpha, beta):
    if maximizing:
        if child_score > best_score:
            best_score = child_score
            best_move = candidate
        if best_score >= beta:
            return best_score, best_move, alpha, beta, True
        if best_score > alpha:
            alpha = best_score
    else:
        if child_score < best_score:
            best_score = child_score
            best_move = candidate
        if best_score <= alpha:
            return best_score, best_move, alpha, beta, True
        if best_score < beta:
            beta = best_score
    return best_score, best_move, alpha, beta, False


def _quiescence(board, color, pre_move, alpha, beta, incoming, qdepth, ctr):
    ctr.q_nodes += 1
    maximizing = color == Color.BLACK
    stand = evaluate(board, pre_move)
    if stand == MAXINT or stand == MININT:
        return pre_move, stand
    loud = incoming > 0 or abs(stand) >= int(SetWeight.LIVE_FIVE)
    if not loud:
        if maximizing and stand >= beta:
            return pre_move, stand
        if not maximizing and stand <= alpha:
            return pre_move, stand
    if qdepth <= 0:
        return pre_move, stand

    from connect6.candidates import generate_scored_candidate_moves as _gen

    moves, threats = _gen(board, QUIESCENCE_WIDTH, color)
    if len(moves) == 0:
        return pre_move, stand
    for move, threat in zip(moves, threats):
        if threat >= THREAT_WIN:
            return move, MAXINT if maximizing else MININT

    if loud:
        pool = list(zip(moves, threats))
        best_score = MININT if maximizing else MAXINT
        best_move = moves[0]
    else:
        pool = [(m, t) for m, t in zip(moves, threats) if t >= QUIESCENCE_MIN_THREATS]
        if len(pool) == 0:
            return pre_move, stand
        best_score = stand
        best_move = pre_move
        if maximizing and stand > alpha:
            alpha = stand
        if not maximizing and stand < beta:
            beta = stand

    for move, threat in pool:
        make_move(board, move, color)
        child_move, child_score = _quiescence(
            board,
            opponent(color),
            move,
            alpha,
            beta,
            threat if threat > 0 else 0,
            qdepth - 1,
            ctr,
        )
        unmake_move(board, move)
        best_score, best_move, alpha, beta, cutoff = _advance(
            maximizing, child_score, move, best_score, best_move, alpha, beta
        )
        if cutoff:
            return best_move, best_score
    return best_move, best_score


def _alpha_beta(
    depth,
    board,
    color,
    pre_move,
    alpha,
    beta,
    ply,
    incoming,
    pvs,
    tt,
    ctr,
):
    if depth <= 0:
        m, s = _quiescence(board, color, pre_move, alpha, beta, incoming, QUIESCENCE_DEPTH, ctr)
        return m, s
    ctr.nodes += 1
    score = evaluate(board, pre_move)
    if score == MAXINT or score == MININT:
        return pre_move, score

    key = None
    if tt is not None:
        key = (b"".join(map(bytes, board)), color, incoming)
        entry = tt.get(key)
        if entry is not None:
            e_depth, e_flag, e_score = entry
            if e_depth >= depth:
                if (
                    e_flag == EXACT
                    or (e_flag == LOWER and e_score >= beta)
                    or (e_flag == UPPER and e_score <= alpha)
                ):
                    ctr.hits += 1
                    return pre_move, e_score

    width = FORWARD_WIDTHS[min(ply, len(FORWARD_WIDTHS) - 1)]
    candidates, threats = generate_scored_candidate_moves(board, width, color)
    if len(candidates) == 0:
        return pre_move, score

    maximizing = color == Color.BLACK
    best_score = MININT if maximizing else MAXINT
    best_move = candidates[0]
    alpha_in, beta_in = alpha, beta
    for index, candidate in enumerate(candidates):
        child_incoming = threats[index]
        make_move(board, candidate, color)
        if pvs and index > 0 and alpha + 1 < beta:
            if maximizing:
                if alpha >= MAXINT - 1:
                    child_move, child_score = _alpha_beta(
                        depth - 1, board, opponent(color), candidate, alpha, beta,
                        ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
                    )
                else:
                    ctr.scouts += 1
                    _, scout = _alpha_beta(
                        depth - 1, board, opponent(color), candidate, alpha, alpha + 1,
                        ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
                    )
                    if scout <= alpha:
                        child_move, child_score = candidate, scout
                    else:
                        ctr.researches += 1
                        child_move, child_score = _alpha_beta(
                            depth - 1, board, opponent(color), candidate, alpha, beta,
                            ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
                        )
            else:
                if beta <= MININT + 1:
                    child_move, child_score = _alpha_beta(
                        depth - 1, board, opponent(color), candidate, alpha, beta,
                        ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
                    )
                else:
                    ctr.scouts += 1
                    _, scout = _alpha_beta(
                        depth - 1, board, opponent(color), candidate, beta - 1, beta,
                        ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
                    )
                    if scout >= beta:
                        child_move, child_score = candidate, scout
                    else:
                        ctr.researches += 1
                        child_move, child_score = _alpha_beta(
                            depth - 1, board, opponent(color), candidate, alpha, beta,
                            ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
                        )
        else:
            child_move, child_score = _alpha_beta(
                depth - 1, board, opponent(color), candidate, alpha, beta,
                ply + 1, child_incoming if child_incoming > 0 else 0, pvs, tt, ctr,
            )
        unmake_move(board, candidate)
        best_score, best_move, alpha, beta, cutoff = _advance(
            maximizing, child_score, candidate, best_score, best_move, alpha, beta
        )
        if cutoff:
            best_score_for_store = best_score
            break
    else:
        best_score_for_store = best_score

    if tt is not None and key is not None:
        if best_score_for_store <= alpha_in:
            flag = UPPER
        elif best_score_for_store >= beta_in:
            flag = LOWER
        else:
            flag = EXACT
        old = tt.get(key)
        if old is None or old[0] <= depth:
            tt[key] = (depth, flag, best_score_for_store)
    return best_move, best_score


def subtree(board, color, depth, candidate, incoming, pvs=False, tt=None):
    work = copy_board(board)
    make_move(work, candidate, color)
    ctr = Counters()
    _, score = _alpha_beta(
        depth - 1, work, opponent(color), candidate, MININT, MAXINT, 1,
        incoming if incoming > 0 else 0, pvs, tt, ctr,
    )
    return score, ctr


@dataclass
class VariantRun:
    score_sum: int
    nodes: int = 0
    q_nodes: int = 0
    hits: int = 0
    scouts: int = 0
    researches: int = 0
    best_move: str = ""
    best_score: int = 0


def _argmax(candidates, scores, color):
    maximizing = color == Color.BLACK
    best = scores[0]
    best_move = candidates[0]
    for move, score in zip(candidates[1:], scores[1:]):
        if maximizing and score > best:
            best, best_move = score, move
        elif not maximizing and score < best:
            best, best_move = score, move
    return best_move, best


def _cells(move):
    return tuple(sorted((p.x, p.y) for p in move.positions))


def run_iteration(board, color, depth, candidates, threats, pvs, tables):
    scores = []
    total = Counters()
    for index, (candidate, threat) in enumerate(zip(candidates, threats)):
        table = tables[index] if tables is not None else None
        score, ctr = subtree(board, color, depth, candidate, threat, pvs, table)
        scores.append(score)
        total.nodes += ctr.nodes
        total.q_nodes += ctr.q_nodes
        total.hits += ctr.hits
        total.scouts += ctr.scouts
        total.researches += ctr.researches
    best_move, best_score = _argmax(candidates, scores, color)
    return best_move, best_score, total


def probe_prefix(prefix):
    import harness

    record_path = REPO / "benchmarks" / "trajectory.json"
    import json

    history = json.loads(record_path.read_text())["history"]
    board, pre_move, color = harness.replay_history(history[:prefix])
    candidates, threats = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], color)
    if not candidates:
        return {"prefix": prefix, "skipped": "no candidates"}

    out = {"prefix": prefix, "candidates": len(candidates)}

    # baseline, calibrated against the engine's own subtrees
    real_nodes = 0
    for candidate, threat in list(zip(candidates, threats))[:3]:
        stats = SearchStats()
        real = search_candidate_subtree(board, color, FLOOR, candidate, threat, stats)
        score, ctr = subtree(board, color, FLOOR, candidate, threat, pvs=True)
        assert score == real.score, (prefix, "calibration score")
        assert ctr.nodes + ctr.q_nodes == stats.node_count, (prefix, "calibration nodes")
        real_nodes += stats.node_count
    base_move, base_score, base_d3 = run_iteration(board, color, FLOOR, candidates, threats, False, None)
    out["base_d3"] = {"nodes": base_d3.nodes, "q": base_d3.q_nodes, "move": str(_cells(base_move)), "score": base_score}

    # pvs d3
    pvs_move, pvs_score, pvs_d3 = run_iteration(board, color, FLOOR, candidates, threats, True, None)
    assert _cells(pvs_move) == _cells(base_move) and pvs_score == base_score, (prefix, "pvs d3 exactness")
    out["pvs_d3"] = {"nodes": pvs_d3.nodes, "q": pvs_d3.q_nodes, "scouts": pvs_d3.scouts, "researches": pvs_d3.researches}

    # tt d3
    fresh = [dict() for _ in candidates]
    tt_move, tt_score, tt_d3 = run_iteration(board, color, FLOOR, candidates, threats, False, fresh)
    assert _cells(tt_move) == _cells(base_move) and tt_score == base_score, (prefix, "tt d3 exactness")
    out["tt_d3"] = {"nodes": tt_d3.nodes, "q": tt_d3.q_nodes, "hits": tt_d3.hits}

    # both d3
    fresh2 = [dict() for _ in candidates]
    both_move, both_score, both_d3 = run_iteration(board, color, FLOOR, candidates, threats, True, fresh2)
    assert _cells(both_move) == _cells(base_move) and both_score == base_score, (prefix, "both d3 exactness")
    out["both_d3"] = {"nodes": both_d3.nodes, "q": both_d3.q_nodes, "hits": both_d3.hits,
                      "scouts": both_d3.scouts, "researches": both_d3.researches}

    # the deepening chains, mirroring _fixed_deepen's budget rule
    base_scores = {FLOOR: base_score}
    base_moves = {FLOOR: _cells(base_move)}
    base_chain_nodes = base_d3.nodes + base_d3.q_nodes
    spent = base_chain_nodes
    depth = FLOOR + 1
    while spent < V6_NODE_BUDGET and depth <= FLOOR + 2:
        base_move_i, base_score_i, ctr_i = run_iteration(board, color, depth, candidates, threats, False, None)
        base_scores[depth] = base_score_i
        base_moves[depth] = _cells(base_move_i)
        spent += ctr_i.nodes + ctr_i.q_nodes
        base_chain_nodes += ctr_i.nodes + ctr_i.q_nodes
        depth += 1

    warm_tables = fresh  # reuse the d3 tables
    warm_chain_nodes = (tt_d3.nodes + tt_d3.q_nodes) + tt_d3.hits
    warm_spent = warm_chain_nodes
    warm_hits = tt_d3.hits
    warm_scores = {FLOOR: tt_score}
    verify_nodes = 0
    depth = FLOOR + 1
    while warm_spent < V6_NODE_BUDGET and depth <= FLOOR + 2:
        if depth not in base_scores:
            base_move_i, base_score_i, ctr_i = run_iteration(
                board, color, depth, candidates, threats, False, None
            )
            base_scores[depth] = base_score_i
            base_moves[depth] = _cells(base_move_i)
            verify_nodes += ctr_i.nodes + ctr_i.q_nodes
        w_move, w_score, w_ctr = run_iteration(board, color, depth, candidates, threats, False, warm_tables)
        assert w_score == base_scores[depth], (prefix, depth, "warm chain exactness")
        assert _cells(w_move) == base_moves[depth], (prefix, depth, "warm chain move exactness")
        warm_spent += (w_ctr.nodes + w_ctr.q_nodes) + w_ctr.hits
        warm_chain_nodes += (w_ctr.nodes + w_ctr.q_nodes) + w_ctr.hits
        warm_hits += w_ctr.hits
        warm_scores[depth] = w_score
        depth += 1

    both_tables = fresh2
    both_chain_nodes = (both_d3.nodes + both_d3.q_nodes) + both_d3.hits
    both_spent = both_chain_nodes
    both_hits = both_d3.hits
    depth = FLOOR + 1
    while both_spent < V6_NODE_BUDGET and depth <= FLOOR + 2:
        if depth not in base_scores:
            base_move_i, base_score_i, ctr_i = run_iteration(
                board, color, depth, candidates, threats, False, None
            )
            base_scores[depth] = base_score_i
            base_moves[depth] = _cells(base_move_i)
            verify_nodes += ctr_i.nodes + ctr_i.q_nodes
        b_move, b_score, b_ctr = run_iteration(board, color, depth, candidates, threats, True, both_tables)
        assert b_score == base_scores[depth], (prefix, depth, "both chain exactness")
        assert _cells(b_move) == base_moves[depth], (prefix, depth, "both chain move exactness")
        both_spent += (b_ctr.nodes + b_ctr.q_nodes) + b_ctr.hits
        both_chain_nodes += (b_ctr.nodes + b_ctr.q_nodes) + b_ctr.hits
        both_hits += b_ctr.hits
        depth += 1
    out["verify_nodes"] = verify_nodes
    out["chain"] = {
        "base_depths": sorted(base_scores),
        "warm_depths": sorted(warm_scores),
    }
    out["base_chain"] = base_chain_nodes
    out["warm_chain"] = warm_chain_nodes
    out["both_chain"] = both_chain_nodes
    out["warm_hits"] = warm_hits
    out["both_hits"] = both_hits
    out["depths_base"] = len(base_scores)
    out["depths_warm"] = len(warm_scores)
    return out


PREFIXES = [1, 3, 19, 23, 25, 27, 29, 34, 37, 38, 41, 42]


def main():
    print(f"probe: {len(PREFIXES)} trajectory prefixes, floor={FLOOR}, budget={V6_NODE_BUDGET}")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(probe_prefix, PREFIXES))
    print(f"\n{'prefix':>6} {'cand':>4} | {'base d3':>8} {'pvs':>8} {'tt':>8} {'both':>8} | "
          f"{'tt hits':>7} {'scouts':>7} {'resch':>6} | {'depths b/w/x':>12} {'chain base':>10} {'warm':>9} {'both':>9}")
    tot_base_d3 = tot_pvs = tot_tt = tot_both = tot_hits = 0
    tot_base_chain = tot_warm_chain = tot_both_chain = 0
    for row in results:
        if "skipped" in row:
            print(f"{row['prefix']:>6}  {row['skipped']}")
            continue
        base = row["base_d3"]["nodes"] + row["base_d3"]["q"]
        pvs = row["pvs_d3"]["nodes"] + row["pvs_d3"]["q"]
        tt = row["tt_d3"]["nodes"] + row["tt_d3"]["q"] + row["tt_d3"]["hits"]
        both = row["both_d3"]["nodes"] + row["both_d3"]["q"] + row["both_d3"]["hits"]
        tot_base_d3 += base
        tot_pvs += pvs
        tot_tt += tt
        tot_both += both
        tot_hits += row["tt_d3"]["hits"]
        tot_base_chain += row["base_chain"]
        tot_warm_chain += row["warm_chain"]
        tot_both_chain += row["both_chain"]
        depths = f"{row['depths_base']}/{row['depths_warm']}/{'x'}"
        print(f"{row['prefix']:>6} {row['candidates']:>4} | {base:>8,} {pvs:>8,} {tt:>8,} {both:>8,} | "
              f"{row['tt_d3']['hits']:>7,} {row['pvs_d3']['scouts']:>7,} {row['pvs_d3']['researches']:>6,} | "
              f"{depths:>12} {row['base_chain']:>10,} {row['warm_chain']:>9,} {row['both_chain']:>9,}")
    n = len(PREFIXES)
    print(f"\ntotals over {n} prefixes (d3 floor iteration):")
    print(f"  base {tot_base_d3:>10,}   pvs {tot_pvs:>10,} ({tot_pvs / tot_base_d3:.3f}x)"
          f"   tt {tot_tt:>10,} ({tot_tt / tot_base_d3:.3f}x)"
          f"   both {tot_both:>10,} ({tot_both / tot_base_d3:.3f}x)")
    print(f"  tt hit rate at d3: {tot_hits:,} hits")
    print(f"\ndeepening chains (mirror of _fixed_deepen under the 36k budget):")
    print(f"  base {tot_base_chain:>10,}   warm-tt {tot_warm_chain:>10,} ({tot_warm_chain / tot_base_chain:.3f}x)"
          f"   both {tot_both_chain:>10,} ({tot_both_chain / tot_base_chain:.3f}x)")
    print("\nall exactness asserts passed: every variant reproduced the baseline move + score")


if __name__ == "__main__":
    main()