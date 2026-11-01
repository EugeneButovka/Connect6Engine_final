"""Assemble the compiled engine under dist/.

Usage: uv run --group build python scripts/build_executable.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIB = ROOT / "dist" / "lib"
MODULES = [
    "defines",
    "board",
    "threats",
    "opening_book",
    "evaluation",
    "candidates",
    "search",
    "vcf",
    "parallel",
    "protocol",
    "game_engine",
]

SETUP = """\
from setuptools import setup

from mypyc.build import mypycify

setup(
    name="connect6",
    packages=["connect6"],
    ext_modules=mypycify([{files}]),
)
"""

LAUNCHER = """\
#!{python}
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("WEIGHTS_FILE", os.path.join(here, "lib", "tuning", "winner.json"))
os.environ.setdefault("BOOK_FILE", os.path.join(here, "lib", "tuning", "opening_book.json"))
os.environ["PYTHONPATH"] = os.path.join(here, "lib") + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, os.path.join(here, "lib"))

from connect6.game_engine import GameEngine

if __name__ == "__main__":
    raise SystemExit(GameEngine().run())
"""


def main() -> int:
    from connect6.defines import ENGINE_NAME

    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    (LIB / "connect6").mkdir(parents=True)
    for source in (ROOT / "connect6").glob("*.py"):
        shutil.copy2(source, LIB / "connect6" / source.name)
    (LIB / "tuning").mkdir()
    for name in ("winner.json", "default.json", "opening_book.json"):
        shutil.copy2(ROOT / "tuning" / name, LIB / "tuning" / name)
    files = ", ".join(f'"connect6/{module}.py"' for module in MODULES)
    (LIB / "setup.py").write_text(SETUP.format(files=files))
    print(f"compiling {len(MODULES)} modules with mypyc ...")
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=LIB, check=True)
    for module in MODULES:
        (LIB / "connect6" / f"{module}.py").unlink()
    (LIB / "setup.py").unlink()
    shutil.rmtree(LIB / "build", ignore_errors=True)
    for cache in (LIB / "connect6").glob("__pycache__"):
        shutil.rmtree(cache)
    launcher = ROOT / "dist" / ENGINE_NAME
    launcher.write_text(LAUNCHER.format(python=sys.executable))
    launcher.chmod(0o755)
    probe = subprocess.run(
        [str(launcher)],
        input="name\nexit\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    name_line = next(line for line in probe.stdout.splitlines() if line.startswith("name "))
    assert name_line == f"name {ENGINE_NAME}", probe.stdout + probe.stderr
    print(f"built dist/{ENGINE_NAME}: {name_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())