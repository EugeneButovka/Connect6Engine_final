import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tuning"))

import genetic_tuner as gt
from connect6.defines import Color

SEEDS = (42, 7)
SETTINGS = {
    "population": 6,
    "generations": 6,
    "depth": 3,
    "max_plies": 60,
    "elite": 2,
    "mutation_rate": 0.35,
}
WINNER_FILE = ROOT / "tuning" / "winner.json"


def run_tuner(seed: int) -> tuple[subprocess.Popen, Path]:
    json_path = ROOT / "tuning" / f"best_weights_d3_seed{seed}.json"
    cmd = [
        sys.executable,
        str(ROOT / "tuning" / "genetic_tuner.py"),
        "--population", str(SETTINGS["population"]),
        "--generations", str(SETTINGS["generations"]),
        "--depth", str(SETTINGS["depth"]),
        "--max-plies", str(SETTINGS["max_plies"]),
        "--elite", str(SETTINGS["elite"]),
        "--mutation-rate", str(SETTINGS["mutation_rate"]),
        "--seed", str(seed),
        "--out", str(json_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )
    return process, json_path


def champion_of(json_path: Path) -> tuple[gt.Weights, float]:
    payload = json.loads(json_path.read_text())
    return gt.Weights(**payload["best"]), float(payload["best_fitness"])


def head_to_head_points(a: gt.Weights, b: gt.Weights) -> float:
    points = 0.0
    for black, white in ((a, b), (b, a)):
        winner, _ = gt.play_game(black, white, SETTINGS["depth"], SETTINGS["max_plies"])
        if winner is None:
            points += 0.5
        elif (black is a) == (winner is Color.BLACK):
            points += 1.0
    return points


def logged(black: gt.Weights, white: gt.Weights) -> tuple[list[str], Color | None, int]:
    moves: list[str] = []
    winner, plies = gt.play_game(black, white, SETTINGS["depth"], SETTINGS["max_plies"], moves)
    return moves, winner, plies


def main() -> None:
    print(
        f"genetic experiment | population {SETTINGS['population']} "
        f"| generations {SETTINGS['generations']} | depth {SETTINGS['depth']} "
        f"| max plies {SETTINGS['max_plies']} | seeds {SEEDS}"
    )
    processes = {}
    for seed in SEEDS:
        processes[seed] = run_tuner(seed)
    outputs = {}
    for seed, (process, json_path) in processes.items():
        output, _ = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"tuner run for seed {seed} failed:\n{output}")
        outputs[seed] = output
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        print(outputs[seed].rstrip())

    champions = {seed: champion_of(processes[seed][1]) for seed in SEEDS}
    print("\n=== champion comparison ===")
    for seed, (champion, fitness) in champions.items():
        print(f"seed {seed}: fitness {fitness:.1f} | {gt.fmt(champion)}")

    ranked = sorted(champions.items(), key=lambda item: item[1][1], reverse=True)
    best_seed, (best_champion, _) = ranked[0]
    if len(ranked) > 1 and ranked[1][1][1] == ranked[0][1][1]:
        rival_seed, (rival_champion, _) = ranked[1]
        if head_to_head_points(rival_champion, best_champion) > 1.0:
            best_seed, best_champion = rival_seed, rival_champion
    print(
        f"strongest: seed {best_seed} (fitness {champions[best_seed][1]:.1f}) -> {WINNER_FILE}"
    )
    WINNER_FILE.parent.mkdir(parents=True, exist_ok=True)
    WINNER_FILE.write_text(json.dumps(asdict(best_champion), indent=2) + "\n")

    print("\n=== controls (deterministic games, one per pairing) ===")
    base_moves, base_winner, base_plies = logged(gt.BASE, gt.BASE)
    base_verdict = "draw" if base_winner is None else ("black wins" if base_winner is Color.BLACK else "white wins")
    print(f"baseline mirror: {base_verdict}, ply {base_plies}")

    champion_moves = {}
    for seed in SEEDS:
        moves, winner, plies = logged(gt.BASE, champions[seed][0])
        verdict = "draw" if winner is None else ("black wins" if winner is Color.BLACK else "white wins")
        print(f"seed-{seed} champion as white vs baseline: {verdict}, ply {plies}")
        champion_moves[seed] = moves

    first, second = SEEDS
    identical = champion_moves[first] == champion_moves[second]
    print(f"champion refutation lines identical: {identical}")
    divergence = 0
    for a, b in zip(champion_moves[first], champion_moves[second]):
        if a != b:
            break
        divergence += 1
    common = 0
    for a, b in zip(champion_moves[first], base_moves):
        if a != b:
            break
        common += 1
    print(
        f"champions share {divergence}/{min(len(champion_moves[first]), len(champion_moves[second]))} moves; "
        f"both diverge from the baseline mirror at ply {common + 1}"
    )

    for black_seed in SEEDS:
        for white_seed in SEEDS:
            if black_seed == white_seed:
                continue
            winner, plies = gt.play_game(
                champions[black_seed][0],
                champions[white_seed][0],
                SETTINGS["depth"],
                SETTINGS["max_plies"],
            )
            verdict = "draw" if winner is None else ("black wins" if winner is Color.BLACK else "white wins")
            print(f"champion seed-{black_seed}(B) vs seed-{white_seed}(W): {verdict}, ply {plies}")

    gt.apply_weights(gt.BASE)
    print("\nexperiment complete; engine globals restored to the hand-tuned baseline")


if __name__ == "__main__":
    main()