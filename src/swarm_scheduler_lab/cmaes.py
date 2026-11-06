"""A small (mu/mu_w, lambda)-CMA-ES used as a model-free baseline.

The searched policies are five bounded weights, a fixed low-dimensional numeric
space in which classical optimizers such as CMA-ES are strong baselines for LLM
proposers (e.g. arXiv:2603.24647). Parameters follow the defaults of Hansen's
tutorial (arXiv:1604.00772). The search runs in the unit box; samples are
clipped and the clipped points are used in the update. Objectives are minimized
and ``None`` (an invalid candidate) ranks last.
"""

from __future__ import annotations

import math

import numpy as np


class CMAES:
    def __init__(self, mean: np.ndarray, sigma: float, rng: np.random.Generator,
                 population: int | None = None):
        self.mean = np.clip(np.asarray(mean, dtype=float), 0.0, 1.0)
        self.n = n = len(self.mean)
        if n < 1 or not sigma > 0:
            raise ValueError("CMA-ES needs a nonempty mean and a positive sigma")
        self.sigma = float(sigma)
        self.rng = rng
        self.population = population or 4 + int(math.floor(3 * math.log(n)))
        if self.population < 2:
            raise ValueError("CMA-ES population must be at least 2")
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.inv_sqrt_C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.generation = 0

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.population, self.n))
        return np.clip(self.mean + self.sigma * (z * self.D) @ self.B.T, 0.0, 1.0)

    def tell(self, xs: np.ndarray, values: list[float | None]) -> None:
        n = self.n
        mu = self.population // 2
        order = sorted((i for i, v in enumerate(values) if v is not None), key=lambda i: values[i])
        selected = order[:mu]
        self.generation += 1
        if not selected:
            return
        weights = math.log(mu + 0.5) - np.log(np.arange(1, len(selected) + 1))
        w = weights / weights.sum()
        mueff = 1.0 / float(np.sum(w**2))
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        old = self.mean
        chosen = np.asarray(xs, dtype=float)[selected]
        self.mean = w @ chosen
        y_w = (self.mean - old) / self.sigma
        self.ps = (1 - cs) * self.ps + math.sqrt(cs * (2 - cs) * mueff) * (self.inv_sqrt_C @ y_w)
        ps_norm = float(np.linalg.norm(self.ps))
        h_sig = ps_norm / math.sqrt(1 - (1 - cs) ** (2 * self.generation)) / chi_n < 1.4 + 2 / (n + 1)
        self.pc = (1 - cc) * self.pc + h_sig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        steps = (chosen - old) / self.sigma
        self.C = ((1 - c1 - cmu) * self.C
                  + c1 * (np.outer(self.pc, self.pc) + (1 - h_sig) * cc * (2 - cc) * self.C)
                  + cmu * (steps.T * w) @ steps)
        self.sigma = min(self.sigma * math.exp((cs / damps) * (ps_norm / chi_n - 1)), 1.0)
        self.C = (self.C + self.C.T) / 2
        eigenvalues, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigenvalues, 1e-20))
        self.inv_sqrt_C = (self.B / self.D) @ self.B.T
