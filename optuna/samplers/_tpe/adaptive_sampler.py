from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.sampler import TPESampler
from optuna.search_space.group_decomposed import _GroupDecomposedSearchSpace


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def adaptive_gamma(x: int) -> int:
    """Elite-set size that grows sub-linearly in the number of trials.

    Unlike :func:`~optuna.samplers._tpe.sampler.default_gamma`, which keeps the elite
    fraction at 10% (capped at 25), this schedule shrinks the elite fraction as
    optimization progresses (roughly ``1 / sqrt(x)``), gradually shifting TPE from
    exploration to exploitation. This speeds up convergence in high-dimensional
    search spaces where the global density ratio is slow to concentrate.
    """

    return min(math.ceil(0.1 * x), max(1, math.ceil(0.5 * math.sqrt(x))))


def linear_forgetting_weights(x: int) -> np.ndarray:
    """Linearly increasing weights over trials ordered by recency.

    Older trials are down-weighted so that the Parzen estimators focus on the
    recent (typically better) region of the search space, which reduces the
    influence of early, scattered exploration samples in high-dimensional spaces.
    """

    if x == 0:
        return np.asarray([])
    return np.arange(1, x + 1, dtype=float) / x


class AdaptiveTPESampler(TPESampler):
    """An improved TPE sampler designed for faster convergence in high-dimensional spaces.

    This sampler inherits the full :class:`~optuna.samplers.TPESampler` machinery
    (multivariate TPE, group-decomposed search spaces for conditional parameters,
    constant liar, and compatibility with every Optuna storage backend) and adds
    three modifications on top of it:

    1. **Adaptive gamma**: the elite-set size grows sub-linearly
       (``O(sqrt(n_trials))``) instead of linearly, so the density ratio
       ``l(x) / g(x)`` becomes increasingly exploitative as trials accumulate.
    2. **Linear forgetting**: older trials are down-weighted linearly, reducing the
       pull of early exploration samples that are far from the promising region.
    3. **Incumbent-centered local search**: with probability ``local_search_prob``,
       numerical parameters are sampled from a Gaussian centered at the incumbent
       (best-so-far) configuration instead of the global TPE candidates, which
       accelerates refinement once a good basin is found. This is only applied to
       single-objective optimization.

    Conditional search spaces are supported out of the box because ``group``
    defaults to :obj:`True`, i.e. the multivariate TPE is applied to each
    subspace of the group-decomposed search space.

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import AdaptiveTPESampler


            def objective(trial):
                model = trial.suggest_categorical("model", ["linear", "mlp"])
                if model == "linear":
                    c = trial.suggest_float("c", 1e-3, 1e3, log=True)
                    return (c - 1.0) ** 2
                width = trial.suggest_int("width", 8, 256)
                depth = trial.suggest_int("depth", 1, 8)
                return (width - 64) ** 2 + (depth - 3) ** 2


            study = optuna.create_study(sampler=AdaptiveTPESampler(seed=0))
            study.optimize(objective, n_trials=50)

    Args:
        n_startup_trials:
            Random sampling is used until this many trials finish.
        n_ei_candidates:
            Number of candidate samples used to calculate the expected improvement.
        seed:
            Seed for random number generator.
        local_search_prob:
            Probability of sampling numerical parameters around the incumbent
            configuration instead of using the global TPE candidates. Must be in
            ``[0, 1]``. Set to ``0`` to disable the local search component.
        local_search_sigma:
            Standard deviation of the incumbent-centered Gaussian, expressed as a
            fraction of each parameter's range (in log-space for log-scale
            distributions). Must be positive.
        multivariate:
            If :obj:`True`, the multivariate TPE is used. Defaults to :obj:`True`.
        group:
            If :obj:`True`, the search space is decomposed into groups so that
            conditional (dynamic) search spaces are handled jointly. Defaults to
            :obj:`True`.
        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting nearby
            configurations.
    """

    def __init__(
        self,
        *,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        seed: int | None = None,
        local_search_prob: float = 0.3,
        local_search_sigma: float = 0.05,
        multivariate: bool | None = True,
        group: bool = True,
        constant_liar: bool = True,
    ) -> None:
        if not 0.0 <= local_search_prob <= 1.0:
            raise ValueError(f"local_search_prob must be in [0, 1], but got {local_search_prob}.")
        if local_search_sigma <= 0.0:
            raise ValueError(f"local_search_sigma must be positive, but got {local_search_sigma}.")

        if group and multivariate is False:
            raise ValueError(
                "``group`` option can only be enabled when ``multivariate`` is enabled."
            )

        # Initialize with ``group=False`` and set up the group-decomposed search space
        # manually to avoid emitting the experimental-warning for an argument that is
        # enabled by default in this sampler.
        super().__init__(
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            seed=seed,
            multivariate=multivariate,
            group=False,
            constant_liar=constant_liar,
        )
        self._group = group
        if group:
            self._group_decomposed_search_space = _GroupDecomposedSearchSpace(True)

        # Override the split/weight schedules without touching the deprecated
        # ``gamma``/``weights`` constructor arguments of TPESampler.
        self._gamma = adaptive_gamma
        self._parzen_estimator_parameters = self._parzen_estimator_parameters._replace(
            weights=linear_forgetting_weights
        )

        self._local_search_prob = local_search_prob
        self._local_search_sigma = local_search_sigma

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        params = super()._sample(study, trial, search_space)
        if self._local_search_prob > 0.0 and not study._is_multi_objective():
            params = self._perturb_around_incumbent(study, search_space, params)
        return params

    def _perturb_around_incumbent(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if self._rng.rng.uniform() >= self._local_search_prob:
            return params

        try:
            incumbent = study.best_trial
        except (RuntimeError, ValueError):
            # No completed (feasible) trial yet.
            return params

        for name, distribution in search_space.items():
            if name not in incumbent.params or name not in params:
                continue
            if not isinstance(distribution, (FloatDistribution, IntDistribution)):
                continue
            params[name] = self._sample_near_value(distribution, incumbent.params[name])
        return params

    def _sample_near_value(
        self, distribution: FloatDistribution | IntDistribution, center: float
    ) -> float | int:
        log_scale = distribution.log

        low = float(distribution.low)
        high = float(distribution.high)
        center_value = float(center)
        if log_scale:
            low, high, center_value = math.log(low), math.log(high), math.log(center_value)

        sigma = self._local_search_sigma * (high - low)
        value = self._rng.rng.normal(center_value, sigma)
        value = min(max(value, low), high)
        if log_scale:
            value = math.exp(value)

        if isinstance(distribution, IntDistribution):
            value = float(round(value))
        step = distribution.step
        if step is not None and step != 1:
            value = distribution.low + step * round((value - distribution.low) / step)
        value = min(max(value, float(distribution.low)), float(distribution.high))

        if isinstance(distribution, IntDistribution):
            return int(value)
        return float(value)
