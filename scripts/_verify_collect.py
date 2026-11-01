import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODE = sys.argv[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tuning"))
if MODE == "compiled":
    sys.path.insert(0, str(ROOT / "dist" / "lib"))
os.environ.setdefault("V6_JOBS", "1")

import harness
from connect6.defines import SearchStats
from connect6.parallel import search_v6
from connect6.protocol import move2msg


def main() -> None:
    rows = []
    started = time.perf_counter()
    for label, board, color, pre_move in harness.documented_positions():
        stats = SearchStats()
        result = search_v6(board, color, 3, pre_move, stats)
        rows.append(
            {
                "label": label,
                "move": move2msg(result.move),
                "score": result.score,
                "nodes": stats.node_count,
            }
        )
    wall = time.perf_counter() - started
    print(json.dumps({"rows": rows, "wall": wall}))


if __name__ == "__main__":
    main()