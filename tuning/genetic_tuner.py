import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import harness
from connect6 import candidates as candidates_module
from connect6 import evaluation as evaluation_module
from connect6.defines import (
    Color,
    SetWeight,
    weight_tables,
)

LIVE_FIVE: int = int(SetWeight.LIVE_FIVE)


@dataclass(frozen=True, slots=True)
class Weights:
    single: int
    pair: int
    triple: int
    four: int
    factor_two: int
    factor_many: int
    threat: int


_HAND_TUNED = json.loads((Path(__file__).resolve().parent / "default.json").read_text())
BASE = Weights(**{name: int(value) for name, value in _HAND_TUNED.items()})

GENE_RANGES: dict[str, tuple[int, int]] = {
    "single": (1, 5),
    "pair": (4, 40),
    "triple": (30, 300),
    "four": (300, 2500),
    "factor_two": (1, 4),
    "factor_many": (2, 6),
    "threat": (250, 4000),
}


def repair(weights: Weights) -> Weights:
    values: dict[str, int] = {}
    for name, (lo, hi) in GENE_RANGES.items():
        values[name] = min(max(getattr(weights, name), lo), hi)
    values["pair"] = max(values["pair"], 3 * values["single"])
    values["triple"] = max(values["triple"], 3 * values["pair"])
    values["four"] = min(max(values["four"], 3 * values["triple"]), (LIVE_FIVE - 1) // 3)
    values["factor_many"] = min(values["factor_many"], (LIVE_FIVE - 1) // values["four"])
    values["factor_two"] = min(values["factor_two"], values["factor_many"] - 1)
    return Weights(**values)


def apply_weights(weights: Weights) -> None:
    live, factor, threat = weight_tables(asdict(weights))
    setattr(evaluation_module, "LIVE_WEIGHTS", live)
    setattr(evaluation_module, "WINDOW_FACTOR", factor)
    setattr(candidates_module, "LIVE_WEIGHTS", live)
    setattr(candidates_module, "WINDOW_FACTOR", factor)
    setattr(candidates_module, "THREAT_WEIGHT", threat)


def fmt(weights: Weights) -> str:
    return (
        f"tiers=({weights.single}, {weights.pair}, {weights.triple}, {weights.four}, {LIVE_FIVE}) "
        f"factor=({weights.factor_two}, {weights.factor_many}) threat={weights.threat}"
    )


def random_individual(rng: random.Random) -> Weights:
    values: dict[str, int] = {}
    for name, (lo, hi) in GENE_RANGES.items():
        values[name] = min(max(round(getattr(BASE, name) * rng.uniform(0.5, 2.0)), lo), hi)
    return repair(Weights(**values))


def crossover(a: Weights, b: Weights, rng: random.Random) -> Weights:
    values = {name: getattr(a if rng.random() < 0.5 else b, name) for name in GENE_RANGES}
    return Weights(**values)


def mutate(weights: Weights, rate: float, rng: random.Random) -> Weights:
    values: dict[str, int] = {}
    for name in GENE_RANGES:
        gene = getattr(weights, name)
        if rng.random() < rate:
            gene = round(gene * rng.uniform(0.7, 1.5))
        values[name] = gene
    return repair(Weights(**values))


def play_game(
    black: Weights,
    white: Weights,
    depth: int,
    max_plies: int,
    moves: list[str] | None = None,
) -> tuple[Color | None, int]:
    record = harness.play_game(
        harness.weights_player(black),
        harness.weights_player(white),
        depth,
        max_plies,
    )
    if moves is not None:
        moves.extend(record.moves)
    return record.winner, record.plies


def round_robin(
    population: list[Weights], depth: int, max_plies: int, verbose: bool
) -> tuple[list[float], int, int]:
    size = len(population)
    scores = [0.0] * size
    wins = 0
    games = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            winner, plies = play_game(population[i], population[j], depth, max_plies)
            games += 1
            if winner is Color.BLACK:
                scores[i] += 1.0
                wins += 1
            elif winner is Color.WHITE:
                scores[j] += 1.0
                wins += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5
            if verbose:
                if winner is None:
                    verdict = "draw"
                else:
                    verdict = "black wins" if winner is Color.BLACK else "white wins"
                print(f"    game {i:2d}(B) vs {j:2d}(W): {verdict} at ply {plies}")
    return scores, wins, games


def next_generation(
    population: list[Weights],
    scores: list[float],
    elite: int,
    mutation_rate: float,
    rng: random.Random,
) -> list[Weights]:
    order = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
    parents = [population[i] for i in order[: max(2, len(population) // 2)]]
    children = [population[i] for i in order[:elite]]
    while len(children) < len(population):
        child = crossover(rng.choice(parents), rng.choice(parents), rng)
        children.append(mutate(child, mutation_rate, rng))
    return children


def outcome(winner: Color | None, own: Color) -> str:
    if winner is None:
        return "draw"
    return "win" if winner is own else "loss"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="genetic tuning of the evaluation weights via round-robin self-play"
    )
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.population < 4:
        parser.error("--population must be at least 4")
    if args.generations < 1:
        parser.error("--generations must be at least 1")
    if not 0 < args.elite < args.population:
        parser.error("--elite must be between 1 and population - 1")
    if not 0 < args.depth < 10:
        parser.error("--depth must be between 1 and 9")
    if args.max_plies < 2:
        parser.error("--max-plies must be at least 2")

    rng = random.Random(args.seed)
    population: list[Weights] = [BASE] + [random_individual(rng) for _ in range(args.population - 1)]
    best_weights = BASE
    best_score = -1.0
    best_generation = 0

    print(f"population {args.population} | generations {args.generations} | depth {args.depth} "
          f"| max plies {args.max_plies} | seed {args.seed}")
    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        scores, wins, games = round_robin(population, args.depth, args.max_plies, args.verbose)
        elapsed = time.perf_counter() - started
        order = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        top = order[0]
        ranked = sorted(scores, reverse=True)
        base_note = ""
        if BASE in population:
            base_note = f" | base {scores[population.index(BASE)]:4.1f}"
        print(
            f"gen {generation:2d} | best {scores[top]:5.1f} | median {ranked[len(ranked) // 2]:5.1f}"
            f"{base_note}"
            f" | decisive {wins}/{games} | {elapsed:6.1f}s | {fmt(population[top])}"
        )
        if scores[top] > best_score:
            best_score = scores[top]
            best_weights = population[top]
            best_generation = generation
        population = next_generation(population, scores, args.elite, args.mutation_rate, rng)

    apply_weights(BASE)
    print()
    print(f"best individual (generation {best_generation}, fitness {best_score:.1f}):")
    print(f"  {fmt(best_weights)}")
    print("hand-tuned baseline:")
    print(f"  {fmt(BASE)}")
    as_black, black_plies = play_game(best_weights, BASE, args.depth, args.max_plies)
    as_white, white_plies = play_game(BASE, best_weights, args.depth, args.max_plies)
    print(
        "champion vs baseline: "
        f"as black {outcome(as_black, Color.BLACK)} (ply {black_plies}), "
        f"as white {outcome(as_white, Color.WHITE)} (ply {white_plies})"
    )

    if args.out:
        payload = {
            "best": asdict(best_weights),
            "base": asdict(BASE),
            "best_fitness": best_score,
            "best_generation": best_generation,
            "vs_base": {
                "as_black": outcome(as_black, Color.BLACK),
                "as_white": outcome(as_white, Color.WHITE),
            },
            "settings": {
                "population": args.population,
                "generations": args.generations,
                "depth": args.depth,
                "max_plies": args.max_plies,
                "seed": args.seed,
                "elite": args.elite,
                "mutation_rate": args.mutation_rate,
            },
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"weights written to {args.out}")


if __name__ == "__main__":
    main()