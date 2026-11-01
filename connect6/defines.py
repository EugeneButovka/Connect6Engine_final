import json
import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Final

GRID_NUM: Final = 21  # Size of the board array: 19x19 cells plus a border ring.
MSG_LENGTH: Final = 512  # Max length of one protocol message.
LOG_FILE: Final = "tia-engine.log"
ENGINE_NAME: Final = "Connect6_Butovka_Ahmed"

# Static evaluation bounds; also used as terminal win/draw scores.
MAXINT: Final = 20000
MININT: Final = -20000

# Candidate generation limits.
MAX_CANDIDATE_MOVES: Final = 30  # Two-stone moves offered to the search per node.
MAX_CANDIDATE_CELLS: Final = 16  # Single cells the pair generation draws from.
SEARCH_DEPTH: Final = 3  # Default depth of the min-max exploration tree.

# Raw board cell values outside the two player colors.
NOSTONE: Final = 0
BORDER: Final = 3

# Line axes on the board; each vector is walked both ways.
Direction = tuple[int, int]
DIRECTIONS: Final[tuple[Direction, ...]] = ((1, 0), (0, 1), (1, 1), (1, -1))

# Evaluation weights, selected once at import from the WEIGHTS_FILE
# environment variable (missing explicit path is an error), else
# tuning/winner.json, else the hand-tuned tuning/default.json.
WEIGHTS_FILE_ENV: Final = "WEIGHTS_FILE"
_TUNING_DIR: Final = Path(__file__).resolve().parent.parent / "tuning"
DEFAULT_WEIGHTS_FILE: Final = _TUNING_DIR / "default.json"
WINNER_WEIGHTS_FILE: Final = _TUNING_DIR / "winner.json"
if not DEFAULT_WEIGHTS_FILE.is_file():
    raise RuntimeError(f"missing weights file: {DEFAULT_WEIGHTS_FILE}")
_configured = os.environ.get(WEIGHTS_FILE_ENV)
_source = Path(_configured) if _configured else WINNER_WEIGHTS_FILE
if not _source.is_file():
    if _configured:
        raise RuntimeError(f"missing weights file: {_source}")
    _source = DEFAULT_WEIGHTS_FILE
_WEIGHTS: Final[dict[str, int]] = json.loads(_source.read_text())

# Opening book: optional JSON of position keys -> move messages, loaded by
# tuning/generate_opening_book.py; a missing or unreadable file disables the
# book. BOOK_FILE pins another path.
BOOK_FILE_ENV: Final = "BOOK_FILE"
_configured_book = os.environ.get(BOOK_FILE_ENV)
OPENING_BOOK_FILE: Final = Path(_configured_book) if _configured_book else _TUNING_DIR / "opening_book.json"

# Value of a living set of stones, named by tier; the four tunable tiers read
# from the weights file, LIVE_FIVE pinned at 10000. LIVE_WEIGHTS is what the
# search hot loops index by run length.
class SetWeight(IntEnum):
    SINGLE = int(_WEIGHTS["single"])
    PAIR = int(_WEIGHTS["pair"])
    TRIPLE = int(_WEIGHTS["triple"])
    LIVE_FOUR = int(_WEIGHTS["four"])
    LIVE_FIVE = 10000


def weight_tables(data: dict[str, int]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    live = (
        0,
        int(data["single"]),
        int(data["pair"]),
        int(data["triple"]),
        int(data["four"]),
        int(SetWeight.LIVE_FIVE),
    )
    factor = (1, int(data["factor_two"]), int(data["factor_many"]))
    return live, factor, int(data["threat"])


_LIVE_TABLE, _FACTOR_TABLE, _THREAT_TABLE = weight_tables(_WEIGHTS)

LIVE_WEIGHTS: Final[tuple[int, ...]] = _LIVE_TABLE

# Space multiplier for living sets, indexed by min(free - (6 - length), 2):
# the number of extra completable 6-windows beyond the only one.
WINDOW_FACTOR: Final[tuple[int, ...]] = _FACTOR_TABLE

# Threat vocabulary: the defender places two stones per turn, so three or
# more simultaneous threats win. THREAT_WIN is the win sentinel returned from
# the sliding window; THREAT_WEIGHT prices one threat into the pair ranking.
THREAT_WIN: Final = 7
THREAT_WEIGHT: Final = _THREAT_TABLE
THREAT_RANK_WIDTH: Final = 10  # Pairs per node re-ranked by threat count.

# Forward pruning for the quiescence search: candidate-pair widths by ply
# from the root, full width at the root and the opponent's first reply ply,
# narrower deeper. Completion pairs are always added before the width cut.
FORWARD_WIDTHS: Final[tuple[int, ...]] = (30, 30, 16, 10)

# Quiescence: loud positions (threats just made, or a standing live five)
# expand forcing moves; quiet ones expand only pairs creating at least
# QUIESCENCE_MIN_THREATS threats; the extension stops after QUIESCENCE_DEPTH
# plies.
QUIESCENCE_MIN_THREATS: Final = 3
QUIESCENCE_WIDTH: Final = 8
QUIESCENCE_DEPTH: Final = 6

# VCF — the forced-win search of the deployed engine: a narrow recursion over
# forcing moves only. VCF_PLY_CAP bounds the chain length, VCF_NODE_CAP makes
# a pathological probe return inconclusive instead of hanging the move.
VCF_PLY_CAP: Final = 20
VCF_NODE_CAP: Final = 1500

# The v5 root: forced-win offense first, then the enhanced search deepened
# while the cumulative node count stays under V5_NODE_BUDGET, then a defence
# screen. V5_DEEPEN_STEPS is the number of plies added to the floor depth,
# V5_SCREEN_WIDTH bounds the probed candidates.
V5_NODE_BUDGET: Final = 12000
V5_DEEPEN_STEPS: Final = 2
V5_SCREEN_WIDTH: Final = 6

# The v6 root: the v5 pipeline with the root split across worker processes,
# principal variation search inside each candidate's subtree. Fixed mode
# deepens under V6_NODE_BUDGET; budget mode deepens by wall time under
# V6_TIME_BUDGET. The depth command of the protocol is the floor of both
# modes; V6_JOBS sets the number of workers (jobs=1 runs the split inline).
V6_NODE_BUDGET: Final = 36000
V6_TIME_BUDGET: Final = 28.0
V6_GROWTH_FACTOR: Final = 12.0
V6_SCREEN_RESERVE: Final = 0.3
V6_JOBS_ENV: Final = "V6_JOBS"
V6_TIME_BUDGET_ENV: Final = "V6_TIME_BUDGET"

# Lightweight hot-path companions of Position/Move: plain tuples.
Cell = tuple[int, int]  # (x, y)
ScoredCell = tuple[int, int, int]  # (score, x, y)
CellPair = tuple[Cell, Cell]  # order-normalized pair of cells
CompletionPair = tuple[int, int, int, int]  # (x1, y1, x2, y2)
RankedPair = tuple[int, int, int]  # (score_sum, index_a, index_b) into the scored-cell list

# Result of one line walk: (length, free, end_a, end_b).
Line = tuple[int, int, Cell, Cell]


class Color(IntEnum):
    BLACK = 1
    WHITE = 2


# A raw board: GRID_NUM x GRID_NUM cell values (NOSTONE, Color, BORDER) with a border ring.
Board = list[list[int]]


class GameResult(Enum):
    WIN = 1
    DRAW = 2


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Move:
    positions: tuple[Position, Position]


@dataclass(frozen=True, slots=True)
class SearchResult:
    move: Move
    score: int


@dataclass(slots=True)
class SearchStats:
    node_count: int = 0
    depth_reached: int = 0