"""Measure PVS economics over the trajectory prefixes: base, pvs and
tt-probe variants of the split-root subtrees, chained under the budget rule.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
    SearchStats,
    SetWeight,
)
from connect6.evaluation import evaluate
from connect6.search import search_candidate_subtree

EXACT, LOWER, UPPER = 0, 1, 2
V6_NODE_BUDGET = 36000
FLOOR = 3
MAX_STEPS = 2  # floor + 0..2


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

    moves, threats = generate_scored_candidate_moves(board, QUIESCENCE_WIDTH, color)
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
            board, opponent(color), move, alpha, beta,
            threat if threat > 0 else 0, qdepth - 1, ctr,
        )
        unmake_move(board, move)
        best_score, best_move, alpha, beta, cutoff = _advance(
            maximizing, child_score, move, best_score, best_move, alpha, beta
        )
        if cutoff:
            return best_move, best_score
    return best_move, best_score


def _alpha_beta(depth, board, color, pre_move, alpha, beta, ply, incoming, pvs, tt, ctr):
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
        child_incoming = threats[index] if threats[index] > 0 else 0
        make_move(board, candidate, color)
        if pvs and index > 0 and alpha + 1 < beta:
            if maximizing and alpha < MAXINT - 1:
                ctr.scouts += 1
                _, scout = _alpha_beta(
                    depth - 1, board, opponent(color), candidate, alpha, alpha + 1,
                    ply + 1, child_incoming, pvs, tt, ctr,
                )
                if scout <= alpha:
                    child_score = scout
                else:
                    ctr.researches += 1
                    _, child_score = _alpha_beta(
                        depth - 1, board, opponent(color), candidate, alpha, beta,
                        ply + 1, child_incoming, pvs, tt, ctr,
                    )
            elif not maximizing and beta > MININT + 1:
                ctr.scouts += 1
                _, scout = _alpha_beta(
                    depth - 1, board, opponent(color), candidate, beta - 1, beta,
                    ply + 1, child_incoming, pvs, tt, ctr,
                )
                if scout >= beta:
                    child_score = scout
                else:
                    ctr.researches += 1
                    _, child_score = _alpha_beta(
                        depth - 1, board, opponent(color), candidate, alpha, beta,
                        ply + 1, child_incoming, pvs, tt, ctr,
                    )
            else:
                _, child_score = _alpha_beta(
                    depth - 1, board, opponent(color), candidate, alpha, beta,
                    ply + 1, child_incoming, pvs, tt, ctr,
                )
        else:
            _, child_score = _alpha_beta(
                depth - 1, board, opponent(color), candidate, alpha, beta,
                ply + 1, child_incoming, pvs, tt, ctr,
            )
        unmake_move(board, candidate)
        best_score, best_move, alpha, beta, cutoff = _advance(
            maximizing, child_score, candidate, best_score, best_move, alpha, beta
        )
        if cutoff:
            break

    if tt is not None and key is not None:
        if best_score <= alpha_in:
            flag = UPPER
        elif best_score >= beta_in:
            flag = LOWER
        else:
            flag = EXACT
        old = tt.get(key)
        if old is None or old[0] <= depth:
            tt[key] = (depth, flag, best_score)
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


def _cells(move):
    return tuple(sorted((p.x, p.y) for p in move.positions))


def run_iteration(board, color, depth, candidates, threats, pvs, tables):
    from connect6.defines import MININT as _MIN, MAXINT as _MAX

    maximizing = color == Color.BLACK
    best_score = _MIN if maximizing else _MAX
    best_move = candidates[0]
    total = Counters()
    proven = False
    for index, (candidate, threat) in enumerate(zip(candidates, threats)):
        table = tables[index] if tables is not None else None
        score, ctr = subtree(board, color, depth, candidate, threat, pvs, table)
        if maximizing and score > best_score:
            best_score, best_move = score, candidate
        elif not maximizing and score < best_score:
            best_score, best_move = score, candidate
        total.nodes += ctr.nodes
        total.q_nodes += ctr.q_nodes
        total.hits += ctr.hits
        total.scouts += ctr.scouts
        total.researches += ctr.researches
        if best_score == (_MAX if maximizing else _MIN):
            proven = True
            break
    return best_move, best_score, total, proven


def probe_prefix(prefix):
    import json

    import harness

    history = json.loads((REPO / "benchmarks" / "trajectory.json").read_text())["history"]
    board, pre_move, color = harness.replay_history(history[:prefix])
    candidates, threats = generate_scored_candidate_moves(board, FORWARD_WIDTHS[0], color)
    if not candidates:
        return {"prefix": prefix, "skipped": "no candidates"}

    # calibration against the deployed engine
    for candidate, threat in list(zip(candidates, threats))[:2]:
        stats = SearchStats()
        real = search_candidate_subtree(board, color, FLOOR, candidate, threat, stats)
        score, ctr = subtree(board, color, FLOOR, candidate, threat, pvs=True)
        assert score == real.score, (prefix, "calibration score")
        assert ctr.nodes + ctr.q_nodes == stats.node_count, (prefix, "calibration nodes")

    base_move, base_score, base_d3, _ = run_iteration(board, color, FLOOR, candidates, threats, False, None)
    pvs_move, pvs_score, pvs_d3, _ = run_iteration(board, color, FLOOR, candidates, threats, True, None)
    assert _cells(pvs_move) == _cells(base_move) and pvs_score == base_score, (prefix, "pvs floor exactness")

    # chains mirroring _fixed_deepen
    def chain(pvs, with_tt):
        from connect6.defines import MAXINT, MININT

        spent = 0
        depths = []
        total = Counters()
        for step in range(MAX_STEPS + 1):
            depth = FLOOR + step
            tables = [dict() for _ in candidates] if with_tt else None
            move, score, ctr, proven = run_iteration(board, color, depth, candidates, threats, pvs, tables)
            spent += ctr.nodes + ctr.q_nodes
            total.nodes += ctr.nodes
            total.q_nodes += ctr.q_nodes
            total.hits += ctr.hits
            total.scouts += ctr.scouts
            total.researches += ctr.researches
            depths.append(depth)
            if spent >= V6_NODE_BUDGET or score in (MAXINT, MININT):
                break
        return depths, total

    base_depths, base_chain = chain(False, False)
    pvs_depths, pvs_chain = chain(True, False)
    tt_depths, tt_chain = chain(True, True)

    return {
        "prefix": prefix,
        "candidates": len(candidates),
        "base_d3": base_d3.nodes + base_d3.q_nodes,
        "pvs_d3": pvs_d3.nodes + pvs_d3.q_nodes,
        "pvs_scouts": pvs_d3.scouts,
        "pvs_researches": pvs_d3.researches,
        "base_chain": base_chain.nodes + base_chain.q_nodes,
        "pvs_chain": pvs_chain.nodes + pvs_chain.q_nodes,
        "base_depths": base_depths,
        "pvs_depths": pvs_depths,
        "tt_chain": tt_chain.nodes + tt_chain.q_nodes,
        "tt_hits": tt_chain.hits,
        "tt_depths": tt_depths,
        "tt_scouts": tt_chain.scouts,
        "tt_researches": tt_chain.researches,
    }


def main():
    prefixes = list(range(1, 49))
    print(f"decision run: all {len(prefixes)} trajectory prefixes, floor={FLOOR}, budget={V6_NODE_BUDGET}")
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(probe_prefix, prefixes))
    rows = [r for r in results if "skipped" not in r]
    print(f"\n{'prefix':>6} | {'base d3':>8} {'pvs d3':>8} | {'base chain':>10} {'pvs chain':>10} "
          f"| {'depths base':>11} {'pvs':>9} | {'tt hits':>7}")
    tot = {k: 0 for k in ("base_d3", "pvs_d3", "base_chain", "pvs_chain", "tt_hits",
                          "pvs_scouts", "pvs_researches")}
    deeper = same = 0
    for row in rows:
        for k in tot:
            tot[k] += row[k]
        b = len(row["base_depths"])
        p = len(row["pvs_depths"])
        if p > b:
            deeper += 1
        elif p == b:
            same += 1
        print(f"{row['prefix']:>6} | {row['base_d3']:>8,} {row['pvs_d3']:>8,} | "
              f"{row['base_chain']:>10,} {row['pvs_chain']:>10,} | "
              f"{','.join(map(str, row['base_depths'])):>11} {','.join(map(str, row['pvs_depths'])):>9} | "
              f"{row['tt_hits']:>7,}")
    n = len(rows)
    print(f"\ntotals over {n} prefixes:")
    print(f"  floor d3 : base {tot['base_d3']:>10,}   pvs {tot['pvs_d3']:>10,} ({tot['pvs_d3'] / tot['base_d3']:.3f}x)")
    print(f"  chain    : base {tot['base_chain']:>10,}   pvs {tot['pvs_chain']:>10,} ({tot['pvs_chain'] / tot['base_chain']:.3f}x)")
    print(f"  pvs scouts {tot['pvs_scouts']:,}, re-searches {tot['pvs_researches']:,} "
          f"({tot['pvs_researches'] / max(tot['pvs_scouts'], 1):.1%} of scouts)")
    print(f"  depth reach: pvs deeper at {deeper}/{n} prefixes, equal at {same}/{n}")
    print(f"  within-tree tt hits across all chain depths: {tot['tt_hits']:,} (zero means no transpositions to harvest)")


if __name__ == "__main__":
    main()