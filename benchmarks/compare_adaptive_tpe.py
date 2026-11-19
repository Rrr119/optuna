"""Comparison experiment: AdaptiveTPESampler vs TPESampler vs RandomSampler.

Benchmarks convergence speed on high-dimensional and conditional search spaces.
Results (CSV files, a convergence plot, and a JSON summary) are written to
``artifacts/adaptive_tpe_benchmark/`` by default.

Usage:

    python benchmarks/compare_adaptive_tpe.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import optuna


N_TRIALS = 200
SEEDS = [0, 1, 2, 3, 4]


def sphere(dim: int) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        return sum(trial.suggest_float(f"x{i}", -5.0, 5.0) ** 2 for i in range(dim))

    return objective


def ackley(dim: int) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        xs = [trial.suggest_float(f"x{i}", -32.768, 32.768) for i in range(dim)]
        s1 = sum(x * x for x in xs)
        s2 = sum(np.cos(2.0 * np.pi * x) for x in xs)
        return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / dim)) - np.exp(s2 / dim) + 20.0 + np.e)

    return objective


def conditional_objective(trial: optuna.Trial) -> float:
    """Conditional (tree-structured) search space with ~30 effective dimensions."""
    model = trial.suggest_categorical("model", ["ridge", "mlp", "rf"])
    if model == "ridge":
        alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
        noise = sum(trial.suggest_float(f"ridge_w{i}", -3.0, 3.0) ** 2 for i in range(10))
        return float((np.log10(alpha) - 1.0) ** 2 + 0.01 * noise)
    elif model == "mlp":
        loss = 0.0
        depth = trial.suggest_int("depth", 1, 6)
        loss += (depth - 3) ** 2
        for i in range(depth):
            width = trial.suggest_int(f"width_l{i}", 8, 512, log=True)
            loss += (np.log2(width) - 6.0) ** 2
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        loss += (np.log10(lr) + 3.0) ** 2
        return float(loss)
    else:
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        noise = sum(trial.suggest_float(f"rf_w{i}", -3.0, 3.0) ** 2 for i in range(10))
        return float(
            ((n_estimators - 200) / 100.0) ** 2 + ((max_depth - 8) / 4.0) ** 2 + 0.01 * noise
        )


def make_samplers(seed: int) -> dict[str, optuna.samplers.BaseSampler]:
    return {
        "Random": optuna.samplers.RandomSampler(seed=seed),
        "TPE": optuna.samplers.TPESampler(seed=seed),
        "AdaptiveTPE": optuna.samplers.AdaptiveTPESampler(seed=seed),
    }


def run_benchmark(
    objective: Callable[[optuna.Trial], float], n_trials: int, seeds: list[int]
) -> dict[str, np.ndarray]:
    curves: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        for sampler_name, sampler in make_samplers(seed).items():
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            values = np.array([t.value for t in study.trials], dtype=float)
            curves.setdefault(sampler_name, []).append(np.minimum.accumulate(values))
    return {k: np.stack(v) for k, v in curves.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--out-dir", default="artifacts/adaptive_tpe_benchmark")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(args.out_dir, exist_ok=True)

    problems = {
        "sphere_40d": sphere(40),
        "ackley_40d": ackley(40),
        "conditional": conditional_objective,
    }

    summary: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 5))
    for ax, (problem_name, objective) in zip(axes, problems.items()):
        results = run_benchmark(objective, args.n_trials, SEEDS)
        summary[problem_name] = {}
        for sampler_name, runs in results.items():
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            x = np.arange(1, args.n_trials + 1)
            ax.plot(x, mean, label=sampler_name)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            summary[problem_name][sampler_name] = float(mean[-1])
            np.savetxt(
                os.path.join(args.out_dir, f"{problem_name}_{sampler_name}.csv"),
                runs,
                delimiter=",",
                header=f"best-so-far per trial, rows=seeds {SEEDS}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best objective value (log)")
        ax.set_title(problem_name)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "convergence.png"), dpi=150)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Final best objective (mean over {len(SEEDS)} seeds):")
    for problem_name, sampler_scores in summary.items():
        print(f"  {problem_name}:")
        for sampler_name, score in sorted(sampler_scores.items(), key=lambda kv: kv[1]):
            print(f"    {sampler_name:<12} {score:.6g}")
    print(f"Artifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
