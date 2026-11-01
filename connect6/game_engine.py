import sys
import time
from collections.abc import Callable

from .board import check_game_end, init_board, make_move, opponent
from .defines import (
    ENGINE_NAME,
    GRID_NUM,
    MSG_LENGTH,
    SEARCH_DEPTH,
    Board,
    Color,
    GameResult,
    Move,
    Position,
    SearchStats,
)
from .opening_book import book_move, load_opening_book
from .parallel import configured_budget, search_v6
from .protocol import color_to_name, log_command, log_error, move2msg, msg2move, print_board


def _flush_output() -> None:
    sys.stdout.flush()


class GameEngine:
    def __init__(self, name: str = ENGINE_NAME) -> None:
        self.engine_name = name
        if len(name) >= MSG_LENGTH:
            print(f"Too long Engine Name: {name}, should be less than: {MSG_LENGTH}", file=sys.stderr)
        self.search_depth = SEARCH_DEPTH
        self.use_vcf = True
        self.color: Color | None = None
        self.board: Board = [[0] * GRID_NUM for _ in range(GRID_NUM)]
        self.last_move = Move((Position(0, 0), Position(0, 0)))
        self._book: dict[str, str] = load_opening_book()
        self._init_game()

    def _init_game(self) -> None:
        init_board(self.board)

    def run(self) -> int:
        self._print_help()
        handlers: dict[str, Callable[[str], bool]] = {
            "name": self._cmd_name,
            "print": self._cmd_print,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "vcf": self._cmd_vcf,
            "unvcf": self._cmd_unvcf,
            "black": self._cmd_black,
            "white": self._cmd_white,
            "next": self._cmd_next,
            "new": self._cmd_new,
            "move": self._cmd_move,
            "depth": self._cmd_depth,
            "help": self._cmd_help,
        }
        while True:
            try:
                msg = input().strip()
            except EOFError:
                break
            log_command(msg)
            command, _, _ = msg.partition(" ")
            handler = handlers.get(command)
            if handler is None:
                continue
            try:
                if not handler(msg):
                    break
            except Exception as error:
                message = f"Error processing '{msg}': {error}"
                print(message, file=sys.stderr)
                log_error(message)
        return 0

    def _apply_move(self, move: Move, color: Color) -> GameResult | None:
        make_move(self.board, move, color)
        return check_game_end(self.board, move)

    def _announce_end(self, result: GameResult | None, color: Color, player: str) -> bool:
        if result == GameResult.WIN:
            print(f"{player} wins with {color_to_name(color)}!")
            return False
        if result == GameResult.DRAW:
            print("Draw!")
            return False
        return True

    def _require_color(self) -> Color:
        if self.color is None:
            raise RuntimeError("no game in progress: start one with 'new black' or 'new white'")
        return self.color

    def _search_and_play(self, pre_move: Move) -> bool:
        color = self._require_color()
        booked = book_move(self.board, self._book)
        if booked is not None:
            self.last_move = booked
            end_result = self._apply_move(booked, color)
            print(f"Book:\t{move2msg(booked)}\n")
            print(f"move {move2msg(booked)}")
            _flush_output()
            return self._announce_end(end_result, color, "AI")
        stats = SearchStats()
        start = time.perf_counter()
        result = search_v6(self.board, color, self.search_depth, pre_move, stats, self.use_vcf, configured_budget())
        end = time.perf_counter()

        print(f"Search Time:\t{end - start:.3f}")
        print(f"Node:\t{stats.node_count}\n")
        print(f"Score:\t{result.score:.3f}")

        self.last_move = result.move
        end_result = self._apply_move(self.last_move, color)
        print(f"move {move2msg(self.last_move)}")
        _flush_output()
        return self._announce_end(end_result, color, "AI")

    def _print_help(self) -> None:
        print(
            f"On help for GameEngine {self.engine_name}\n"
            " name        - print the name of the Game Engine.\n"
            " print       - print the board.\n"
            " exit/quit   - quit the game.\n"
            " black XXXX  - place the black stone on the position XXXX on the board.\n"
            " white XXXX  - place the white stone on the position XXXX on the board, X is from A to S.\n"
            " next        - the engine will search the move for the next step.\n"
            " move XXXX   - tell the engine that the opponent made the move XXXX,\n"
            "              and the engine will search the move for the next step.\n"
            " new black   - start a new game and set the engine to black player.\n"
            " new white   - start a new game and set it to white.\n"
            " depth d     - set the min-max search depth, default is 3.\n"
            " vcf         - set vcf search.\n"
            " unvcf       - set none vcf search.\n"
            " help        - print this help.\n")

    def _cmd_name(self, msg: str) -> bool:
        print(f"name {self.engine_name}")
        return True

    def _cmd_print(self, msg: str) -> bool:
        print_board(self.board)
        return True

    def _cmd_exit(self, msg: str) -> bool:
        return False

    def _cmd_vcf(self, msg: str) -> bool:
        self.use_vcf = True
        return True

    def _cmd_unvcf(self, msg: str) -> bool:
        self.use_vcf = False
        return True

    def _cmd_black(self, msg: str) -> bool:
        self.last_move = msg2move(msg[6:])
        make_move(self.board, self.last_move, Color.BLACK)
        self.color = Color.BLACK
        return True

    def _cmd_white(self, msg: str) -> bool:
        self.last_move = msg2move(msg[6:])
        make_move(self.board, self.last_move, Color.WHITE)
        self.color = Color.WHITE
        return True

    def _cmd_next(self, msg: str) -> bool:
        self.color = opponent(self._require_color())
        return self._search_and_play(self.last_move)

    def _cmd_new(self, msg: str) -> bool:
        self._init_game()
        if msg[4:] == "black":
            self.last_move = msg2move("JJ")
            make_move(self.board, self.last_move, Color.BLACK)
            self.color = Color.BLACK
            print("move JJ")
            _flush_output()
        else:
            self.color = Color.WHITE
        return True

    def _cmd_move(self, msg: str) -> bool:
        color = self._require_color()
        self.last_move = msg2move(msg[5:])
        result = self._apply_move(self.last_move, opponent(color))
        if not self._announce_end(result, opponent(color), "Human"):
            return False
        return self._search_and_play(self.last_move)

    def _cmd_depth(self, msg: str) -> bool:
        d = int(msg[6:])
        if 0 < d < 10:
            self.search_depth = d
        print(f"Set the search depth to {self.search_depth}.\n")
        return True

    def _cmd_help(self, msg: str) -> bool:
        self._print_help()
        return True