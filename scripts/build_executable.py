"""Assemble the compiled engine under dist/.

Usage:
  uv run --group build python scripts/build_executable.py
  uv run --group build python scripts/build_executable.py --launcher
"""

import argparse
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
TUNING = ("winner.json", "default.json", "opening_book.json")
WINDOWS = sys.platform == "win32"
EXE_SUFFIX = ".exe" if WINDOWS else ""

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

LAUNCHER_BAT = """\
@echo off
"{python}" "%~dp0{name}.py" %*
"""

ONEFILE_ENTRY = """\
import multiprocessing

from connect6.game_engine import GameEngine


def main():
    gameEngine = GameEngine()
    gameEngine.run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
"""

STDLIB_HIDDEN = ("json", "time", "pathlib", "dataclasses", "enum", "concurrent.futures")


def bundle_onefile(engine_name: str) -> None:
    work = ROOT / "build"
    work.mkdir(exist_ok=True)
    entry = work / "_onefile_entry.py"
    entry.write_text(ONEFILE_ENTRY)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        engine_name,
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(work),
        "--specpath",
        str(work),
        "--paths",
        str(LIB),
        "--clean",
        "--noconfirm",
    ]
    for module in MODULES:
        command += ["--hidden-import", f"connect6.{module}"]
    # the mypyc shared runtime suffix varies by platform
    for runtime in list(LIB.glob("*__mypyc*.so")) + list(LIB.glob("*__mypyc*.pyd")):
        command += ["--hidden-import", runtime.name.split(".")[0]]
    for hidden in STDLIB_HIDDEN:
        command += ["--hidden-import", hidden]
    for name in TUNING:
        command += ["--add-data", str(ROOT / "tuning" / name) + os.pathsep + "tuning"]
    command.append(str(entry))
    print("bundling the compiled build into a single executable ...")
    subprocess.run(command, cwd=ROOT, check=True)


def probe_engine(launcher: Path) -> str:
    probe = subprocess.run(
        [str(launcher)],
        input="name\nexit\n",
        capture_output=True,
        text=True,
        timeout=180,
    )
    name_lines = [line for line in probe.stdout.splitlines() if line.startswith("name ")]
    assert name_lines, probe.stdout + probe.stderr
    return name_lines[0]


def write_launcher(engine_name: str) -> Path:
    if not WINDOWS:
        launcher = ROOT / "dist" / engine_name
        launcher.write_text(LAUNCHER.format(python=sys.executable))
        launcher.chmod(0o755)
        return launcher
    script = ROOT / "dist" / f"{engine_name}.py"
    script.write_text(LAUNCHER.format(python=sys.executable))
    batch = ROOT / "dist" / f"{engine_name}.bat"
    batch.write_text(LAUNCHER_BAT.format(python=sys.executable, name=engine_name), newline="\r\n")
    return batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the compiled engine under dist/.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launcher", action="store_true", help="stage dist/lib and write a launcher instead of the single executable")
    mode.add_argument("--onefile", action="store_true", help="bundle the compiled build into a single executable (default)")
    args = parser.parse_args()

    from connect6.defines import ENGINE_NAME

    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    (LIB / "connect6").mkdir(parents=True)
    for source in (ROOT / "connect6").glob("*.py"):
        shutil.copy2(source, LIB / "connect6" / source.name)
    (LIB / "tuning").mkdir()
    for name in TUNING:
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
    if args.launcher:
        launcher = write_launcher(ENGINE_NAME)
    else:
        bundle_onefile(ENGINE_NAME)
        launcher = ROOT / "dist" / (ENGINE_NAME + EXE_SUFFIX)
    name_line = probe_engine(launcher)
    assert name_line == f"name {ENGINE_NAME}", name_line
    print(f"built dist/{launcher.name}: {name_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())