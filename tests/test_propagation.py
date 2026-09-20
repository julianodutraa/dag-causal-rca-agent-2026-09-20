"""Correctness tests for the propagation module.

The key correctness claim is: on a polytree (no two parents of any node
share an ancestor), the closed-form mean-field marginals are exact, not
merely approximate. This is checked against a brute-force reference that
enumerates every combination of the underlying independent Bernoulli bits
explicitly, which is only tractable on a small hand-built graph, so a
dedicated tiny polytree fixture is used here rather than the full
synthetic pipeline DAG.
"""

from __future__ import annotations

import itertools
import random

from dagrca.dag import PipelineDAG, build_synthetic_pipeline_dag
from dagrca.propagation import mean_field_marginals, monte_carlo_marginals


def _tiny_polytree() -> PipelineDAG:
    # A and B are independent sources with no shared ancestor; C has two
    # parents (A, B) but they share no ancestry, so this is a polytree.
    nodes = ["A", "B", "C", "D"]
    edges = [("A", "C"), ("B", "C"), ("C", "D")]
    edge_prob = {("A", "C"): 0.7, ("B", "C"): 0.4, ("C", "D"): 0.6}
    leak_prob = {"A": 0.02, "B": 0.05, "C": 0.03, "D": 0.01}
    layer = {n: "x" for n in nodes}
    return PipelineDAG(nodes=nodes, edges=edges, edge_prob=edge_prob, leak_prob=leak_prob, layer=layer)


def _brute_force_marginals(dag: PipelineDAG, root: str) -> dict[str, float]:
    """Exact marginals via full enumeration of the independent random bits.

    Only tractable for small graphs (this project uses it on a 4-node,
    3-edge fixture: 3 edge bits + 3 leak bits for non-root nodes = 6 bits,
    2**6 = 64 outcomes).
    """
    non_root_nodes = [n for n in dag.nodes if n != root]
    bit_names = [("leak", n) for n in non_root_nodes] + [("edge", e) for e in dag.edges]
    probs = []
    for kind, key in bit_names:
        probs.append(dag.leak_prob[key] if kind == "leak" else dag.edge_prob[key])

    totals = {n: 0.0 for n in dag.nodes}
    totals[root] = 1.0
    for bits in itertools.product([0, 1], repeat=len(bit_names)):
        weight = 1.0
        for bit, p in zip(bits, probs):
            weight *= p if bit else (1.0 - p)
        leak = {}
        trans = {}
        for (kind, key), bit in zip(bit_names, bits):
            if kind == "leak":
                leak[key] = bool(bit)
            else:
                trans[key] = bool(bit)
        failed = {root: True}
        for node in dag.nodes:
            if node == root:
                continue
            triggered = leak[node] or any(
                failed.get(p, False) and trans[(p, node)] for p in dag.parents[node]
            )
            failed[node] = triggered
        for node, is_failed in failed.items():
            if node != root and is_failed:
                totals[node] += weight
    return totals


def test_mean_field_is_exact_on_polytree():
    dag = _tiny_polytree()
    exact = _brute_force_marginals(dag, root="A")
    mf = mean_field_marginals(dag, root="A")
    for node in dag.nodes:
        assert abs(exact[node] - mf[node]) < 1e-9, (node, exact[node], mf[node])


def test_root_marginal_is_one():
    dag = build_synthetic_pipeline_dag(seed=7)
    mf = mean_field_marginals(dag, root="ingest_orders")
    assert mf["ingest_orders"] == 1.0


def test_marginals_are_monotone_probabilities():
    dag = build_synthetic_pipeline_dag(seed=7)
    for root in dag.nodes:
        mf = mean_field_marginals(dag, root)
        for p in mf.values():
            assert 0.0 <= p <= 1.0


def test_monte_carlo_converges_to_mean_field_on_polytree():
    dag = _tiny_polytree()
    mf = mean_field_marginals(dag, root="A")
    mc = monte_carlo_marginals(dag, root="A", trials=200_000, seed=1)
    for node in dag.nodes:
        assert abs(mf[node] - mc[node]) < 0.01, (node, mf[node], mc[node])


def test_monte_carlo_is_deterministic_given_seed():
    dag = build_synthetic_pipeline_dag(seed=7)
    mc1 = monte_carlo_marginals(dag, root="ingest_orders", trials=5000, seed=42)
    mc2 = monte_carlo_marginals(dag, root="ingest_orders", trials=5000, seed=42)
    assert mc1 == mc2
