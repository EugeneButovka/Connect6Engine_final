import argparse
import json
from pathlib import Path

import harness

DEFAULT_OPENINGS = ["IIKK", "JIKJ", "JKJI", "IKKJ", "LLHH"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate the opening book by engine self-play.")
    parser.add_argument("--depth", type=int, default=4, help="search depth for self-play moves")
    parser.add_argument("--plies", type=int, default=10, help="plies to record per opening line")
    parser.add_argument("--openings", nargs="+", default=DEFAULT_OPENINGS, help="white's first replies to JJ")
    parser.add_argument("--weights", default=str(root / "tuning" / "winner.json"), help="weights file to pin")
    parser.add_argument("--out", default=str(root / "tuning" / "opening_book.json"), help="output book path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harness.pin_weights(args.weights)

    from connect6.board import check_game_end, make_move
    from connect6.defines import Color, SearchStats
    from connect6.opening_book import position_key
    from connect6.protocol import move2msg, msg2move
    from connect6.search import search_alpha_beta_with_quiescence

    def play_line(seed: str, record_first_reply: bool) -> dict[str, str]:
        board = harness.empty_board()
        first = harness.center()
        make_move(board, first, Color.BLACK)
        book: dict[str, str] = {"": "JJ"}
        pre_move = first
        for ply in range(1, args.plies + 1):
            mover = Color.WHITE if ply % 2 == 1 else Color.BLACK
            if ply == 1:
                move = msg2move(seed)
            else:
                result = search_alpha_beta_with_quiescence(board, mover, args.depth, pre_move, SearchStats())
                move = result.move
            if ply > 1 or record_first_reply:
                book[position_key(board)] = move2msg(move)
            make_move(board, move, mover)
            if check_game_end(board, move) is not None:
                break
            pre_move = move
        return book

    merged: dict[str, str] = {}
    for index, seed in enumerate(args.openings):
        line = play_line(seed, record_first_reply=(index == 0))
        merged.update(line)
        print(f"line {seed}: {len(line)} entries")
    out_path = Path(args.out)
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(merged)} entries to {out_path}")


if __name__ == "__main__":
    main()