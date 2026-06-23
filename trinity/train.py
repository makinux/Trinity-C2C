"""
Learned coordinator skeleton (2/2): optimize the linear head theta with sep-CMA-ES
=============================================================
J(theta) = E_tau[R(tau)], R in {0,1} (task success). Derivative-free. Caps the rollout count via a strict budget.
- production objective: score real rollouts with the trinity_eval scorer (needs: local models)
- selftest (no model): verify (a) the optimizer converges (b) the coordinator wiring (c) the head is learnable

Run:
  python -m trinity.train --selftest     # no model
  python -m trinity.train                # real training (needs Qwen3-0.6B + 3 local models)
"""
from __future__ import annotations

import sys
import numpy as np

from trinity.p0 import POOL, Config, run, State, Role
from trinity.coordinator import (
    MockFeaturizer, Qwen3HiddenStateFeaturizer, LinearHead, LearnedCoordinator,
    ROLE_ACTIONS,
)
from trinity.eval import TASKS, make_scorer


# ============================================================
# 1. separable CMA-ES (diagonal covariance)
#    Equivalent to the paper's sep-CMA-ES. In production you can also use
#    cma.CMAEvolutionStrategy(x0, sigma, {'CMA_diagonal': True}) from `pip install cma`.
# ============================================================
class SepCMAES:
    def __init__(self, n: int, x0=None, sigma0: float = 0.3, popsize: int | None = None, seed: int = 0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros(n) if x0 is None else np.array(x0, float)
        self.sigma = float(sigma0)
        self.lam = popsize or (4 + int(3 * np.log(n)))      # lambda = 4 + 3 ln n
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.C = np.ones(n)                                 # diagonal variance
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.gen = 0

    def ask(self) -> np.ndarray:
        self._z = self.rng.standard_normal((self.lam, self.n))
        self._y = self._z * np.sqrt(self.C)
        return self.m + self.sigma * self._y

    def tell(self, fitnesses) -> None:
        idx = np.argsort(-np.asarray(fitnesses))[:self.mu]   # higher reward is better
        z, y = self._z[idx], self._y[idx]
        zw, yw = self.w @ z, self.w @ y
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        self.sigma *= float(np.exp(np.clip((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1), -1.0, 1.0)))
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
                < (1.4 + 2 / (self.n + 1)) * self.chiN)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.w @ (y ** 2)))
        self.C = np.clip(self.C, 1e-12, 1e6)        # numerical stabilization (prevent drift/explode)
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))
        self.gen += 1


# ============================================================
# 2. objective: score theta with real rollouts (J(theta) = mean task success)
# ============================================================
def make_rollout_objective(tasks, featurizer, head, pool, max_turns: int = 5):
    scorers = [make_scorer(t) for t in tasks]

    def objective(theta: np.ndarray) -> float:
        coord = LearnedCoordinator(featurizer, head, theta)
        ok = 0
        for t, sc in zip(tasks, scorers):
            res = run(t.query, coord, pool, Config(max_turns=max_turns, verbose=False))
            ok += int(bool(res["final"]) and sc(res["final"]))
        return ok / len(tasks)

    return objective


# ============================================================
# 3. Training loop (cap atomic rollouts via a strict budget)
# ============================================================
def train(objective, n_params: int, n_rollouts_per_eval: int,
          budget: int = 600, sigma0: float = 0.3, m_reps: int = 1,
          seed: int = 0, x0=None, log: bool = True):
    # Room for improvement (Codex): under noise, re-evaluating elites with higher m_reps before tell() is more stable (adaptive reps).
    # The budget is checked per generation, so the last generation may overshoot a little.
    es = SepCMAES(n_params, x0=x0, sigma0=sigma0, seed=seed)
    spent = 0
    best = (-1.0, None)
    while spent < budget:
        xs = es.ask()
        fits = []
        for x in xs:
            f = float(np.mean([objective(x) for _ in range(m_reps)]))   # m_reps replications (noise averaging)
            spent += m_reps * n_rollouts_per_eval
            fits.append(f)
            if f > best[0]:
                best = (f, x.copy())
        es.tell(fits)
        if log:
            print(f"gen {es.gen:3d} | spent {spent:5d}/{budget} | best {best[0]:.3f} | sigma {es.sigma:.3f}")
    return best


# ============================================================
# 4. Real training (needs: Qwen3-0.6B + 3 local models)
# ============================================================
def run_real_training():
    from trinity.config import get
    feat = Qwen3HiddenStateFeaturizer(get("coordinator", "slm_model", "Qwen/Qwen3-0.6B"))   # GPU + transformers
    head = LinearHead(dim=feat.dim)
    print(f"head params = {head.n_params}  (dim={feat.dim} x {head.n_actions} roles + bias)")
    obj = make_rollout_objective(TASKS, feat, head, POOL)
    # sep-CMA-ES hyperparameters come from config.training (budget/sigma0/m_reps/seed).
    best_f, best_theta = train(obj, head.n_params, n_rollouts_per_eval=len(TASKS),
                               budget=get("training", "budget", 8000),
                               sigma0=get("training", "sigma0", 0.3),
                               m_reps=get("training", "m_reps", 8),
                               seed=get("training", "seed", 0))
    np.save("coordinator_theta.npy", best_theta)
    print(f"best J(theta)={best_f:.3f}  -> saved coordinator_theta.npy")


# ============================================================
# 5. Self-test (no model needed)
# ============================================================
def selftest():
    # (a) does sep-CMA-ES converge to a known optimum (maximize -||x-3||^2)
    target = 3.0 * np.ones(8)
    es = SepCMAES(8, sigma0=1.0, seed=1)
    for _ in range(80):
        xs = es.ask()
        es.tell([-float(np.sum((x - target) ** 2)) for x in xs])
    err = float(np.linalg.norm(es.m - target))
    assert err < 0.3, f"not converged: {err}"
    print(f"[selftest] sep-CMA-ES converges : ||m-target||={err:.3f}")

    # (b) wiring: a trajectory runs with LearnedCoordinator + MockFeaturizer + mock models
    from trinity.eval import MockModel
    feat = MockFeaturizer(16)
    head = LinearHead(dim=16)
    rng = np.random.default_rng(0)
    coord = LearnedCoordinator(feat, head, rng.standard_normal(head.n_params))

    def worker_fn(s, u):
        return ("```python\nimport heapq\ndef merge(a,b):\n    return list(heapq.merge(a,b))\n```"
                if "merge" in u else "```python\ndef solve():\n    return None\n```")
    pool = {
        "thinker":  MockModel("t", lambda s, u: "plan"),
        "worker":   MockModel("w", worker_fn),
        "verifier": MockModel("v", lambda s, u: "VERDICT: ACCEPT"),
    }
    res = run(TASKS[0].query, coord, pool, Config(max_turns=5, verbose=False))
    assert res["final"] is not None, "no trajectory produced"
    print("[selftest] coordinator wiring OK : orchestrate->decide->rollout")

    # (c) learning happens: the head can be shaped by sep-CMA-ES toward the target policy (pick WORKER in all states)
    states = [State(query=f"q{i}") for i in range(12)]
    w_idx = ROLE_ACTIONS.index(Role.WORKER)

    def pick_worker_rate(theta):
        return float(np.mean([int(np.argmax(head.logits(theta, feat.encode(s))) == w_idx) for s in states]))

    init = pick_worker_rate(np.zeros(head.n_params))
    best_f, _ = train(lambda th: pick_worker_rate(th), head.n_params,
                      n_rollouts_per_eval=len(states), budget=3000, sigma0=0.5, seed=2, log=False)
    print(f"[selftest] head is trainable : pick-WORKER {init:.2f} -> {best_f:.2f}")
    assert best_f >= 0.95, best_f
    print("[selftest] ALL PASSED")


# ============================================================
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_real_training()
