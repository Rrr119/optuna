from __future__ import annotations

import numpy as np
import pytest

import optuna
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import AdaptiveTPESampler
from optuna.samplers._tpe.adaptive_sampler import adaptive_gamma
from optuna.samplers._tpe.adaptive_sampler import linear_forgetting_weights
from optuna.samplers._tpe.sampler import default_gamma


def test_adaptive_gamma_is_sublinear() -> None:
    assert adaptive_gamma(1) == 1
    # The elite fraction must shrink as the number of trials grows.
    assert adaptive_gamma(100) / 100 < adaptive_gamma(10) / 10
    # It never exceeds the default schedule.
    for n in (1, 5, 10, 50, 100, 1000):
        assert 1 <= adaptive_gamma(n) <= default_gamma(n)


def test_linear_forgetting_weights() -> None:
    assert linear_forgetting_weights(0).size == 0
    weights = linear_forgetting_weights(10)
    assert weights.shape == (10,)
    # Older trials (smaller indices) get smaller weights.
    assert np.all(np.diff(weights) > 0)
    assert weights[-1] == pytest.approx(1.0)


def test_invalid_local_search_arguments() -> None:
    with pytest.raises(ValueError):
        AdaptiveTPESampler(local_search_prob=-0.1)
    with pytest.raises(ValueError):
        AdaptiveTPESampler(local_search_prob=1.1)
    with pytest.raises(ValueError):
        AdaptiveTPESampler(local_search_sigma=0.0)
    with pytest.raises(ValueError):
        AdaptiveTPESampler(group=True, multivariate=False)


def _sphere_objective(trial: optuna.Trial, dim: int = 10) -> float:
    return sum(trial.suggest_float(f"x{i}", -5.0, 5.0) ** 2 for i in range(dim))


def test_single_objective_optimization_converges() -> None:
    sampler = AdaptiveTPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere_objective, n_trials=100)
    # Loose sanity bound: TPE-style samplers easily reach below 10 on a 10-dim sphere.
    assert study.best_value < 10.0


def test_conditional_search_space() -> None:
    def objective(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical("model", ["linear", "mlp"])
        if model == "linear":
            c = trial.suggest_float("c", 1e-3, 1e3, log=True)
            return (c - 1.0) ** 2
        width = trial.suggest_int("width", 8, 256)
        depth = trial.suggest_int("depth", 1, 8)
        return float((width - 64) ** 2 + (depth - 3) ** 2)

    sampler = AdaptiveTPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=50)
    assert len(study.trials) == 50
    # Both branches of the conditional space must have been explored.
    models = {t.params["model"] for t in study.trials}
    assert models == {"linear", "mlp"}


def test_rdb_storage_compatibility() -> None:
    storage = optuna.storages.RDBStorage("sqlite:///:memory:")
    sampler = AdaptiveTPESampler(seed=0)
    study = optuna.create_study(storage=storage, sampler=sampler)
    study.optimize(_sphere_objective, n_trials=30)
    assert len(study.trials) == 30
    # The constant-liar relative params must be persisted in the storage.
    assert any(
        key.startswith("tpe:relative_params")
        for trial in study.trials
        for key in trial.system_attrs
    )


def test_reproducibility_with_seed() -> None:
    def run() -> list[dict[str, float]]:
        study = optuna.create_study(sampler=AdaptiveTPESampler(seed=42))
        study.optimize(_sphere_objective, n_trials=30)
        return [t.params for t in study.trials]

    assert run() == run()


def test_multi_objective_runs_without_local_search() -> None:
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", -5.0, 5.0)
        return x**2, (x - 2.0) ** 2

    sampler = AdaptiveTPESampler(seed=0, local_search_prob=1.0)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=30)
    assert len(study.trials) == 30


def test_local_search_disabled_with_zero_prob() -> None:
    sampler = AdaptiveTPESampler(seed=0, local_search_prob=0.0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere_objective, n_trials=30)
    assert len(study.trials) == 30


@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(0.0, 10.0),
        FloatDistribution(1e-3, 1e3, log=True),
        FloatDistribution(0.0, 1.0, step=0.1),
        IntDistribution(0, 100),
        IntDistribution(1, 1000, log=True),
        IntDistribution(0, 100, step=5),
    ],
)
def test_sample_near_value_respects_domain(
    distribution: FloatDistribution | IntDistribution,
) -> None:
    sampler = AdaptiveTPESampler(seed=0, local_search_sigma=0.5)
    center = (float(distribution.low) + float(distribution.high)) / 2.0
    if distribution.log:
        center = float(np.sqrt(distribution.low * distribution.high))
    for _ in range(100):
        value = sampler._sample_near_value(distribution, center)
        assert distribution.low <= value <= distribution.high
        step = distribution.step
        if step is not None and step != 1:
            k = (value - distribution.low) / step
            assert abs(k - round(k)) < 1e-8
        if isinstance(distribution, IntDistribution):
            assert isinstance(value, int)
