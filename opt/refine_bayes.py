"""Bayesian searches for the refinement steps (v8 T10), with CMA-ES's ask / tell interface.

    s = make_search("tpe" | "gp", x0, lo, hi, budget, seed)
    while not s.stop():
        xs = s.ask()
        s.tell(xs, [f(x) for x in xs])

- **TPE** (Optuna's Tree-structured Parzen Estimator): models where good and bad results
  lie and samples where good ones are likelier. The first trial is ``x0``.
- **GP** (our own, numpy / scipy; no PyTorch): a Gaussian process with an RBF kernel (length
  scale and signal chosen by marginal likelihood from a small grid) over the inputs scaled to
  [0, 1]; the next point maximizes Expected Improvement over random candidates plus a local
  cloud around the best point. Starts with ``x0`` and a Latin hypercube.

Costs from the planners can be huge for rejected candidates (1e3, 1e9): the GP caps them at
the 80th percentile of the values seen before standardizing, so they mark a region as bad
without flattening the rest of the model.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def make_search(method: str, x0, lo, hi, budget: int, seed: int = 1, batch: int = 4):
    if method == "tpe":
        return TpeSearch(x0, lo, hi, budget, seed, batch)
    if method == "gp":
        return GpSearch(x0, lo, hi, budget, seed)
    raise ValueError(f"unknown search method {method!r}")


class TpeSearch:
    def __init__(self, x0, lo, hi, budget: int, seed: int, batch: int):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.budget, self.batch, self.told = int(budget), int(batch), 0
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(3, min(8, budget // 3)))
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        x0 = np.clip(np.asarray(x0, float), self.lo, self.hi)
        self.study.enqueue_trial({f"x{i}": float(v) for i, v in enumerate(x0)})
        self._open: list = []

    def stop(self) -> bool:
        return self.told >= self.budget

    def ask(self) -> list[np.ndarray]:
        n = min(self.batch, self.budget - self.told)
        self._open = [self.study.ask() for _ in range(n)]
        return [
            np.array(
                [
                    t.suggest_float(f"x{i}", float(a), float(b))
                    for i, (a, b) in enumerate(zip(self.lo, self.hi, strict=True))
                ]
            )
            for t in self._open
        ]

    def tell(self, xs, fs) -> None:
        for t, f in zip(self._open, fs, strict=True):
            self.study.tell(t, float(f) if np.isfinite(f) else 1e12)
        self.told += len(self._open)
        self._open = []


class GpSearch:
    LENGTHS = (0.05, 0.1, 0.2, 0.35, 0.6, 1.0)
    NOISE = 1e-6

    def __init__(self, x0, lo, hi, budget: int, seed: int):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.d = len(self.lo)
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        n0 = min(self.budget, max(3, 2 * self.d + 1))
        z0 = np.clip((np.asarray(x0, float) - self.lo) / self.span, 0, 1)
        self._init = [z0, *self._lhs(n0 - 1)]
        self.Z: list[np.ndarray] = []
        self.y: list[float] = []

    def _lhs(self, n: int) -> list[np.ndarray]:
        if n <= 0:
            return []
        cut = (np.arange(n)[:, None] + self.rng.random((n, self.d))) / n
        for j in range(self.d):
            cut[:, j] = self.rng.permutation(cut[:, j])
        return list(cut)

    def stop(self) -> bool:
        return len(self.y) >= self.budget

    def _x(self, z: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(z, 0, 1) * self.span

    def ask(self) -> list[np.ndarray]:
        if self._init:
            n = min(len(self._init), self.budget - len(self.y))
            batch, self._init = self._init[:n], self._init[n:]
            self._asked = batch
            return [self._x(z) for z in batch]
        z = self._next()
        self._asked = [z]
        return [self._x(z)]

    def tell(self, xs, fs) -> None:
        for z, f in zip(self._asked, fs, strict=True):
            self.Z.append(np.asarray(z, float))
            self.y.append(float(f) if np.isfinite(f) else 1e12)

    # --- the model ---------------------------------------------------------------------------

    def _fit(self):
        Z, y = np.array(self.Z), np.array(self.y)
        y = np.minimum(y, np.quantile(y, 0.8)) if len(y) >= 5 else y
        mu, sd = y.mean(), y.std() or 1.0
        ys = (y - mu) / sd
        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        best = None
        for ell in self.LENGTHS:
            K = np.exp(-0.5 * D2 / ell**2) + self.NOISE * np.eye(len(Z))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            a = np.linalg.solve(L.T, np.linalg.solve(L, ys))
            ll = -0.5 * ys @ a - np.log(np.diag(L)).sum()
            if best is None or ll > best[0]:
                best = (ll, ell, L, a)
        return Z, ys, best

    def _next(self) -> np.ndarray:
        Z, ys, best = self._fit()
        if best is None:  # degenerate (duplicate points): explore
            return self.rng.random(self.d)
        _, ell, L, a = best
        z_best = Z[int(np.argmin(ys))]
        cand = np.vstack(
            [
                self.rng.random((1500, self.d)),
                np.clip(z_best + self.rng.normal(0, 0.05, (500, self.d)), 0, 1),
            ]
        )
        Ks = np.exp(-0.5 * ((cand[:, None, :] - Z[None, :, :]) ** 2).sum(-1) / ell**2)
        m = Ks @ a
        v = np.linalg.solve(L, Ks.T)
        s = np.sqrt(np.maximum(1.0 - (v**2).sum(0), 1e-12))
        imp = ys.min() - m - 0.01
        zsc = imp / s
        ei = imp * norm.cdf(zsc) + s * norm.pdf(zsc)
        return cand[int(np.argmax(ei))]
