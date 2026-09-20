"""Noisy-OR failure propagation: ground truth simulator and mean-field inference.

Generative model
-----------------
Fix a root cause node ``r`` that fails with probability 1 at time 0. Every
other node ``v`` has an independent background ("leak") failure indicator
``L_v ~ Bernoulli(leak_prob[v])``, and every edge ``(u, v)`` has an
independent transmission indicator ``T_uv ~ Bernoulli(edge_prob[u, v])``.
All ``L_v`` and ``T_uv`` are mutually independent random bits. A non-root
node ``v`` fails if and only if

    Fails(v) = L_v OR (OR over parents u of v of (T_uv AND Fails(u)))

This is the standard noisy-OR causal independence model (Pearl, 1988):
each parent is an independent sufficient cause of failure, and the leak
term captures causes external to the graph.

Two ways to compute P(Fails(v) | root = r) are implemented:

1. ``monte_carlo_marginals``: draws the independent random bits explicitly
   and simulates forward in topological order. This is unbiased and is
   treated as ground truth in this project (with a large enough sample
   size the Monte Carlo standard error is smaller than the effects we
   measure; see ``tests/test_propagation.py``).

2. ``mean_field_marginals``: a closed-form recursion,

       P(v fails) = 1 - (1 - leak_prob[v]) * prod_{u in parents(v)} (1 - edge_prob[u, v] * P(u fails))

   evaluated once in topological order. This formula is exact when the
   parents of every node are conditionally independent given the root,
   which holds on a polytree. On a general DAG with convergence points
   (two parents of the same node sharing a common ancestor), the parents'
   failure events are positively correlated through the shared ancestor,
   the independence assumption is violated, and the formula is a biased
   approximation. Quantifying that bias against the Monte Carlo ground
   truth is the calibration study in this project (see
   ``scripts/run_demo.py`` and ``results/calibration_report.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from .dag import PipelineDAG


def mean_field_marginals(dag: PipelineDAG, root: str) -> dict[str, float]:
    """Closed-form mean-field marginal failure probabilities given a root.

    Returns a dict mapping every node to P(node fails | root failed with
    probability 1), computed with the independence-of-causal-influence
    approximation described in the module docstring.
    """
    p: dict[str, float] = {}
    for node in dag.nodes:
        if node == root:
            p[node] = 1.0
            continue
        survive = 1.0 - dag.leak_prob[node]
        for parent in dag.parents[node]:
            p_parent_fail = p[parent]
            trans = dag.edge_prob[(parent, node)]
            survive *= 1.0 - trans * p_parent_fail
        p[node] = 1.0 - survive
    return p


@dataclass(frozen=True)
class SimulatedIncident:
    root: str
    failed: frozenset[str]


def _simulate_once(dag: PipelineDAG, root: str, rng: random.Random) -> frozenset[str]:
    failed: dict[str, bool] = {}
    for node in dag.nodes:
        if node == root:
            failed[node] = True
            continue
        leaked = rng.random() < dag.leak_prob[node]
        triggered = False
        if not leaked:
            for parent in dag.parents[node]:
                if failed[parent] and rng.random() < dag.edge_prob[(parent, node)]:
                    triggered = True
                    break
        failed[node] = leaked or triggered
    return frozenset(n for n, f in failed.items() if f)


def simulate_incident(dag: PipelineDAG, root: str, rng: random.Random) -> SimulatedIncident:
    """Draws one incident (a failed-node set) from the generative model."""
    return SimulatedIncident(root=root, failed=_simulate_once(dag, root, rng))


def monte_carlo_marginals(dag: PipelineDAG, root: str, trials: int, seed: int) -> dict[str, float]:
    """Ground-truth marginal failure probabilities via Monte Carlo simulation."""
    rng = random.Random(seed)
    counts = {n: 0 for n in dag.nodes}
    for _ in range(trials):
        failed = _simulate_once(dag, root, rng)
        for n in failed:
            counts[n] += 1
    return {n: counts[n] / trials for n in dag.nodes}
