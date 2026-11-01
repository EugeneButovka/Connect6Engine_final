# Connect6 Engine (TIA.Connect6)

A [Connect6](https://en.wikipedia.org/wiki/Connect6) game engine written in pure Python (standard library only). It communicates over stdin/stdout with a simple text protocol, making it compatible with the standard Connect6 game interface.

Authors: Evgenii Butovka, Hasnaat

## Highlights

- **Search.** Min-max and alpha-beta baselines; the deployed `search_v6()` adds a proven forced-win layer, quiescence, forward pruning and a parallel split root.
- **Tuned weights.** A self-play genetic algorithm found the weights that ship as `tuning/winner.json`; the hand-tuned `tuning/default.json` is the fallback and baseline.
- **Opening book.** The first moves are answered instantly from `tuning/opening_book.json`.
- **Threat-counting candidates.** Wins, forks and forced blocks are always in the candidate pool.
- **Position battery.** Nineteen curated positions are re-verified end-to-end by `tuning/harness.py`.
- **Reproducible.** The searches are deterministic, so benchmark reruns are exact.

## Contents

[Quick start](#quick-start) · [Build an executable](#build-an-executable) · [Commands](#commands) · [Architecture](#architecture) · [Tests](#tests) · [Type checking](#type-checking) · [Benchmarks](#benchmarks) · [Tuning](#tuning)

## Quick start

[uv](https://docs.astral.sh/uv/) manages Python 3.14 and the virtual environment automatically:

```sh
uv run game6        # installed console script
uv run main.py      # or run the entry point directly
```

A session looks like this (`>` marks what you type):

```
On help for GameEngine Connect6_Butovka_Ahmed
 name        - print the name of the Game Engine.
 ...
> name
name Connect6_Butovka_Ahmed
> new white
> move JJ
Book:	IIKK
move IIKK
> move LKKK
Search Time:	0.240
Node:	1433
Score:	-60.000
move LLIJ
```

`new white` makes the engine White; `move JJ` is the opponent's center opening, answered from the opening book in zero time; a reply that leaves the book lines (here `LKKK`) falls back to the search.

**Configuration.** The engine reads its weights, at import, from the file named by the `WEIGHTS_FILE` environment variable — default: `tuning/winner.json`; if that file is absent, the hand-tuned `tuning/default.json` applies. The opening book works the same way: `tuning/opening_book.json` by default, or the file named by `BOOK_FILE` — a missing or unreadable book simply disables the book. The deployment budget is `V6_TIME_BUDGET` seconds of wall time per move (default 28), and `V6_JOBS` sets the number of root-split worker processes (default: all logical cores; `V6_JOBS=1` runs the same split inline). Every command is appended to `tia-engine.log`.

## Build an executable

To build the compiled engine — the mypyc build of the whole `connect6` package bundled into a single executable, under `dist/`:

```sh
uv run --group build python scripts/build_executable.py
```

The command compiles the package with mypyc, stages the extension modules as `dist/lib/connect6`, copies the shipped weights and opening book to `dist/lib/tuning`, and packs the modules, their shared runtime and the three `tuning/` artifacts with PyInstaller into `dist/Connect6_Butovka_Ahmed` (the name comes from `ENGINE_NAME`) — a self-contained executable that needs nothing else on disk. The build targets the current operating system — macOS, Windows or Linux; a C compiler is required for mypyc (Xcode tools, MSVC Build Tools or gcc), and on Windows the executable is `Connect6_Butovka_Ahmed.exe`. The bundled entry mirrors `main.py`, and both call `multiprocessing.freeze_support()` at the top of `__main__`: a frozen executable's spawned workers re-execute the app's own binary, and the call diverts each child into worker mode instead of booting the whole engine again — a no-op in interpreted runs. The build ends with a protocol smoke test. To additionally verify the compilation: `uv run --group build python scripts/verify_compiled.py` asserts identical moves, scores and node counts between the interpreted and compiled builds, and reports the wall speedup.

The `--launcher` alternative produces the same compiled build but writes a `dist/Connect6_Butovka_Ahmed` launcher script instead of the bundle (on Windows, a `Connect6_Butovka_Ahmed.bat` wrapper that runs the same launcher script with the recorded interpreter): it pins `WEIGHTS_FILE` and `BOOK_FILE` to the staged files — explicit environment values win — and exports `PYTHONPATH`, so the engine's own worker processes import the compiled package too.

The legacy alternative — a PyInstaller one-file build (also captured by the `Connect6_Butovka_Ahmed.spec` at the repository root) — **supported on all our operating systems: macOS, Windows and Linux**:

```sh
uv run pyinstaller --onefile --name Connect6_Butovka_Ahmed --add-data "tuning/winner.json:tuning" --add-data "tuning/default.json:tuning" --add-data "tuning/opening_book.json:tuning" main.py
```

On Windows the `--add-data` separator is `;` instead of `:`, and the executable is `dist/Connect6_Butovka_Ahmed.exe`:

```sh
uv run pyinstaller --onefile --name Connect6_Butovka_Ahmed --add-data "tuning/winner.json;tuning" --add-data "tuning/default.json;tuning" --add-data "tuning/opening_book.json;tuning" main.py
```

It bundles the interpreter, the package and the three `tuning/` artifacts into a single `dist/Connect6_Butovka_Ahmed` executable — the same path the mypyc build writes, so the last build wins; the artifacts land beside the bundled package, so the default weight and book lookups (`connect6/defines.py` resolves them relative to the package) keep working. Unlike the mypyc build, this one ships interpreted bytecode — no compilation speedup.

## Commands

Once running, the engine accepts the following commands:

| Command      | Description                                                            |
|--------------|------------------------------------------------------------------------|
| `name`       | Print the name of the game engine.                                      |
| `print`      | Print the board (`O` = black, `*` = white).                             |
| `exit`/`quit`| Quit the game.                                                          |
| `black XXXX` | Place black stones on the given positions.                              |
| `white XXXX` | Place white stones on the given positions.                             |
| `next`       | The engine searches a move for the next step.                          |
| `move XXXX`  | Tell the engine the opponent made the move `XXXX`; it replies with its own move. |
| `new black`  | Start a new game; the engine plays Black and opens in the center.       |
| `new white`  | Start a new game; the engine plays White.                               |
| `depth d`    | Set the search depth floor (1–9, default 3); the budget deepens past it while the wall budget lasts. |
| `vcf`        | Enable the forced-win search stage.                                      |
| `unvcf`      | Disable the forced-win search stage.                                     |
| `help`       | Print the command list.                                                 |

*Legend — `X` is a letter `A`–`S` naming a column/row, two letters per stone: `JJ` is the centre, `CCDD` a pair of stones; `d` is the search depth (1–9, default 3).*

When the game ends, the engine announces the winner with their color (`AI wins with white!` / `Human wins with black!`) or a draw, and exits. Malformed commands are reported on stderr and ignored — the engine keeps running.

## Architecture

```
main.py                             entry point (thin: creates and runs the engine)
connect6/                           the engine package
├── game_engine.py                  game loop, command protocol, move handling, book lookup
├── search.py                       one unified `_alpha_beta` loop: min-max, alpha-beta, quiescence variant
├── vcf.py                          the v5 root: forced-win offense, budgeted deepening, defence screen
├── parallel.py                     the deployed root: parallel split root with PVS subtrees, node/wall budgets
├── evaluation.py                   static evaluation: terminal scores, living-stone-set counting
├── candidates.py                   candidate generation: cell scoring, pairing, threat re-rank, completion pairs
├── threats.py                      threat counting: sliding-window defender-forcing lines
├── board.py                        board rules: win/draw detection, the `measure_line` walk
├── protocol.py                     text protocol I/O: `move2msg`/`msg2move`, board printing, logging
├── opening_book.py                 opening book: position keys, JSON load, recorded-move lookup
└── defines.py                      constants, enums, frozen `Position`/`Move`, tuple aliases
tests/                              the test suite
└── test.py                         98 deterministic tests, run with `uv run tests/test.py`
benchmarks/                         the benchmarks
├── battery.json                     the battery's 19 curated positions
├── trajectory.json                  a recorded 48-ply self-play game
├── perf_depth_v1_min_max.ipynb      min-max at depths 1-4 vs the O(B^depth) model
├── perf_depth_v2_alpha_beta.ipynb   min-max vs alpha-beta at depths 1-5
├── perf_search_v4_enhanced.ipynb    the enhanced search vs plain alpha-beta
├── perf_search_v5_layer.ipynb       the v5 layer vs the v4 engine over the trajectory
├── perf_search_v6_split.ipynb       the split root vs the v5 engine over the trajectory
├── perf_versions_v1_v6.ipynb        the version timeline v1-v6: games and cost tables
├── perf_position_battery.ipynb      the battery, assert-validated
├── measure_factor_budget.py         split-vs-shared cost and the budget comparison
├── probe_pvs_full.py                PVS economics over the trajectory
├── probe_pvs_tt.py                  PVS/TT probe over selected prefixes
├── tournament_cloudict.py           games against the reference Cloudict binary
├── tournament_cloudict.json         the recorded Cloudict tournament
├── tournament_verdicts.py           v4/v5/v6 games from the divergence prefixes
├── tournament_verdicts.json         the recorded verdict tournament
├── analyze_tournament.py            analysis of the recorded tournament JSON
├── assignment_coverage/             coverage notebooks and their records
└── cloudict/                        the reference opponent binary and its pattern table
tuning/                             the experiments and their generated artifacts
├── genetic_tuner.py                genetic tuning of the evaluation weights
├── run_experiment.py               repeats the GA experiment and refreshes winner.json
├── generate_opening_book.py        opening-book generation by engine self-play
├── harness.py                      shared scaffolding plus the position-battery runner
├── opening_book.json               engine default opening book
├── winner.json                     engine default weights
├── best_weights_d3_seed42.json     GA champion, seed 42
├── best_weights_d3_seed7.json      GA champion, seed 7
└── default.json                    the hand-tuned configuration: fallback and tuner baseline
scripts/                            auxiliary tooling (build, verification, notebook runs)
├── build_executable.py             the mypyc build: extensions, staged weights/book, launcher
├── verify_compiled.py              compiled-vs-interpreted verification
├── _verify_collect.py              the collection helper the verifier spawns
└── nbclient_run.py                 the stored-output notebook runner
```

- **Search** (`search.py`) — one unified `_alpha_beta` loop: the `prune` flag selects min-max or alpha-beta; the `enhanced` flag adds forward pruning by `FORWARD_WIDTHS` and quiescence at the horizon (completion pairs survive every width cut). The deployed root is `search_v6` (`parallel.py`): the forced-win layer first, the enhanced search deepened under a budget, then a defence screen — with the root split across worker processes and principal variation search inside each candidate's subtree.
- **Evaluation** (`evaluation.py`) — living stone sets weighted by tier (`LIVE_WEIGHTS`) and by space (`WINDOW_FACTOR`).
- **Candidates** (`candidates.py`, `threats.py`) — top cells by line potential paired into moves, co-linear pairs corrected, top pairs re-ranked by the threats they create, and two-stone completion pairs injected first.
- **Opening book** (`opening_book.py`) — position keys are stone inventories; a hit is applied and the search skipped, a miss or invalid entry falls back to the search silently.

## Tests

```sh
uv run tests/test.py
```

98 deterministic tests covering, in order of the pipeline they pin:

- **Board** — board state, win/draw termination, color helpers.
- **Static evaluation** — tier weights and the window factor.
- **Candidate generation and threats** — co-linear pair correction, the sliding window, joint two-stone counts, fork ranking.
- **Min-max and alpha-beta search** — forced-block regression tests; alpha-beta's equivalence to the naive search: same scores and moves, strictly fewer nodes.
- **Enhanced search** — quiescence, forward pruning, forced blocks.
- **V5 and v6 roots** — forced-win proofs, budgeted deepening, the defence screen; split-root agreement, worker-count invariance, the wall-time budget.
- **Opening book** — stone-inventory keys, entry rejection, clean search fallback through the game engine.
- **Protocol** — the engine binary over pipes: handshake, depth configuration, malformed commands.

The suite pins `WEIGHTS_FILE` and `BOOK_FILE` before importing the engine, so every assertion measures the same configuration regardless of the environment.

## Type checking

```sh
uv run mypy
```

The codebase is fully typed: `Color`/`GameResult` enums, frozen `Position`/`Move` dataclasses, annotated functions and attributes (`pyproject.toml` pins the checked files).

## Benchmarks

The benchmark notebooks plus the external tournament are runnable from the command line or in Jupyter. From the command line: `uv run python scripts/nbclient_run.py benchmarks/<notebook>.ipynb` re-executes a notebook headlessly and writes its stored outputs back. The notebooks pin their weights before importing the engine and assert the deterministic numbers they measure, so a clean run is itself a validation — only wall times vary between runs.

- `benchmarks/perf_depth_v1_min_max.ipynb` — the min-max baseline: `search_min_max` at depths 1–4, against the theoretical `O(B^depth)` model.
- `benchmarks/perf_depth_v2_alpha_beta.ipynb` — min-max at depths 1–4 and alpha-beta at depths 1–5, asserting identical scores, with the ideal `O(B^(depth/2))` guide.
- `benchmarks/perf_search_v4_enhanced.ipynb` — the enhanced search vs plain alpha-beta: position tables, the depth and pruning ladders, the head-to-head, a common trajectory and the transposition-table measurement.
- `benchmarks/perf_search_v5_layer.ipynb` — the v5 layer vs the v4 engine over the common trajectory: node costs, verdict flips, the floor-invariance ladder and the deepening probe.
- `benchmarks/perf_search_v6_split.ipynb` — the split root vs the v5 engine on the same trajectory: the regression fixture, the window/PVS depth ladder and the cost exhibit.
- `benchmarks/perf_versions_v1_v6.ipynb` — the version timeline v1-v6: twelve games in both colors, cost tables and the six-version genealogy.
- The battery runner (`uv run tuning/harness.py battery [--jobs N]`) — the nineteen curated positions played out in the deployment configuration; parallel and sequential results are identical.
- `benchmarks/perf_position_battery.ipynb` — the assert-validated record of the battery: all verdicts, transcript identities, ply counts and node budgets.
- `benchmarks/tournament_cloudict.py` — the compiled engine against the reference Cloudict binary, six openings, both colors, depth floors 3-5, every move recorded to `benchmarks/tournament_cloudict.json` (needs `benchmarks/cloudict/patterns.in` beside the binary).
- `benchmarks/tournament_verdicts.py` — v4/v5/v6 games started from the trajectory's verdict-flip and budget-boundary prefixes, with the floor-3/4/5 verdict probe first; recorded to `benchmarks/tournament_verdicts.json`.
- `benchmarks/analyze_tournament.py` — the analysis re-derived from the recorded tournament JSON: the matrix, the loss declarations, the economics and the win-claim dissection.
- `benchmarks/assignment_coverage/` — three coverage notebooks: the position battery against Cloudict, per-version search metrics on two boards, and reply timing with a protocol smoke test (records under `benchmarks/assignment_coverage/records/`).

## Tuning

```sh
uv run tuning/run_experiment.py                          # repeat the experiment (~15 min) and refresh tuning/winner.json
uv run tuning/genetic_tuner.py --depth 3 --generations 6 --population 6 --out tuning/best.json
uv run tuning/generate_opening_book.py                   # regenerate tuning/opening_book.json by self-play (~1 min)
```

`tuning/genetic_tuner.py` optimizes the evaluation weights with a genetic algorithm. A population of weight configurations plays a round-robin of full self-play games through the real alpha-beta search — fitness is wins (1) and draws (0.5) — and each generation keeps the elite, recombines the top half by uniform crossover, and mutates around the survivors. A seeded RNG plus the deterministic search make every run reproducible. The run ends with the best individual playing the hand-tuned baseline head-to-head and optionally writes the winner to JSON (`--out`); `--verbose` prints every game.

`tuning/run_experiment.py` repeats the full experiment: both documented seeds in parallel, the champion comparison, the control games — and copies the strongest champion into `tuning/winner.json`.

`tuning/generate_opening_book.py` plays the engine against itself over the opening lines and records every position's reply (`--depth`, `--plies`, `--openings`, `--weights`, `--out`).

`tuning/harness.py` is the scaffolding the tuning scripts and the benchmark notebooks share: it pins `WEIGHTS_FILE` before the engine is imported, builds boards and the documented positions, adapts search functions and weight vectors to a common player interface, and referees games.

**Weights configuration.** The weights are selected once, at import: `connect6/defines.py` reads the JSON named by the `WEIGHTS_FILE` environment variable, falling back as described in [Quick start](#quick-start). The test suite sets `WEIGHTS_FILE` to `tuning/default.json` before importing the engine. The genetic machinery itself (the `Weights` vector, gene ranges, constraint repair) lives in `tuning/genetic_tuner.py`, outside the engine package, and starts from `tuning/default.json` as its baseline.