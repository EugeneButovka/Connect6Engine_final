"""Verify the compiled engine: identical moves, scores and node counts,
plus the wall speedup.

Usage: uv run --group build python scripts/verify_compiled.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "dist" / "lib"
COLLECT = ROOT / "scripts" / "_verify_collect.py"


def collect(mode: str) -> tuple[list[dict[str, object]], float]:
    env = dict(os.environ)
    env["WEIGHTS_FILE"] = str(ROOT / "tuning" / "winner.json")
    env["BOOK_FILE"] = str(ROOT / "tuning" / "no_book.json")
    if mode == "compiled":
        env["PYTHONPATH"] = str(LIB)
    else:
        env.pop("PYTHONPATH", None)
    probe = subprocess.run(
        [sys.executable, str(COLLECT), mode],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=1200,
    )
    payload = json.loads(probe.stdout)
    return payload["rows"], payload["wall"]


def main() -> int:
    interp_rows, interp_wall = collect("interpreted")
    comp_rows, comp_wall = collect("compiled")
    assert len(interp_rows) == len(comp_rows)
    for interp, comp in zip(interp_rows, comp_rows):
        assert interp == comp, f"{interp} != {comp}"
        print(
            f"{interp['label']:34s} {interp['move']:<5s} score {interp['score']:6d} "
            f"{interp['nodes']:7d} nodes — identical"
        )
    print(f"interpreted {interp_wall:6.2f} s, compiled {comp_wall:6.2f} s "
          f"— {interp_wall / comp_wall:.2f}x on the same fixed-depth work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())