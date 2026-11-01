"""Measure the split-vs-shared cost per iteration and the budget comparison.

Part 1 — the per-iteration cost of the shared-window search against the
split root, over the trajectory prefixes. Part 2 — the whole trajectory
re-run with V6_NODE_BUDGET patched to 54,000 against the default 36,000.
"""

import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ["WEIGHTS_FILE"] = str(REPO / "tuning" / "winner.json")
os.environ["V6_JOBS"] = "1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tuning"))

import harness


def factor_worker(prefix):
    from connect6.defines import SearchStats
    from connect6.parallel import _split_root
    from connect6.search import search_alpha_beta_with_quiescence

    history = json.loads((REPO / "benchmarks" / "trajectory.json").read_text())["history"]
    board, pre_move, color = harness.replay_history(history[:prefix])
    row = {"prefix": prefix}
    for depth in (3, 4):
        seq = SearchStats()
        search_alpha_beta_with_quiescence(board, color, depth, pre_move, seq)
        split = SearchStats()
        _split_root(board, color, depth, pre_move, split, None)
        row[f"seq{depth}"] = seq.node_count
        row[f"split{depth}"] = split.node_count
    return row


def budget_worker(task):
    from connect6.defines import SearchStats
    from connect6.protocol import move2msg
    import connect6.parallel as parallel
    from connect6.parallel import search_v6

    history, prefix, budget = task
    setattr(parallel, "V6_NODE_BUDGET", budget)
    board, pre_move, color = harness.replay_history(history[:prefix])
    stats = SearchStats()
    result = search_v6(board, color, 3, pre_move, stats)
    return {"prefix": prefix, "nodes": stats.node_count, "score": result.score,
            "move": move2msg(result.move)}


def main():
    record = json.loads((REPO / "benchmarks" / "trajectory.json").read_text())
    history = record["history"]

    print("=== part 1: the factor table (split with PVS vs shared) ===")
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(factor_worker, range(1, len(history))))
    # one-node stage-1 answers build no tree
    ones = {42, 44, 45, 46, 47}
    table = [r for r in rows if r["prefix"] not in ones]
    for depth in (3, 4):
        ratios = sorted(r[f"split{depth}"] / r[f"seq{depth}"] for r in table)
        print(f"depth {depth}: ratio {ratios[0]:.2f}x .. {ratios[-1]:.2f}x, "
              f"median {statistics.median(ratios):.2f}x over {len(table)} prefixes")
    worst = max(table, key=lambda r: r["split3"])
    print(f"worst floor split3: {worst['split3']:,} nodes (prefix {worst['prefix']})")
    floor_stops = [r["prefix"] for r in table if r["split3"] >= 36000]
    print(f"floor stops (split3 >= 36000): {len(floor_stops)}/{len(table)} -> {floor_stops}")

    print("\n=== part 2: the B=54,000 budget comparison ===")
    tasks = [(history, prefix, 54000) for prefix in range(1, len(history))]
    with ProcessPoolExecutor(max_workers=12) as pool:
        big = {row["prefix"]: row for row in pool.map(budget_worker, tasks)}

    # the default budget side, re-run in the same pass
    tasks = [(history, prefix, 36000) for prefix in range(1, len(history))]
    with ProcessPoolExecutor(max_workers=12) as pool:
        small = {row["prefix"]: row for row in pool.map(budget_worker, tasks)}

    v5_rows = harness.analyze_history(history, [("v5", harness_search_v5())], 3, jobs=12)
    v5 = {row.prefix: row for row in v5_rows}
    positions = sorted(small)
    for name, side in (("36000", small), ("54000", big)):
        total = sum(side[p]["nodes"] for p in positions)
        changes = sum(1 for p in positions if side[p]["move"] != v5[p].move)
        flips = sum(
            1 for p in positions
            if (abs(side[p]["score"]) >= 20000) != (abs(v5[p].score) >= 20000)
        )
        divergences = [p for p in positions if side[p]["move"] != v5[p].move]
        peak = max(side[p]["nodes"] for p in positions)
        peak_at = max(positions, key=lambda p: side[p]["nodes"])
        v5_total = sum(v5[p].nodes for p in positions)
        print(f"B={name}: total {total:,} ({total / v5_total:.2f}x v5), "
              f"changes {changes}/48, flips {flips}, peak {peak:,} (prefix {peak_at})")
        print(f"  divergences: {divergences}")
    both_moves = sum(1 for p in positions if small[p]["move"] == big[p]["move"])
    print(f"36k vs 54k: identical moves at {both_moves}/48 positions")


def harness_search_v5():
    from connect6.vcf import search_v5

    return search_v5


if __name__ == "__main__":
    main()