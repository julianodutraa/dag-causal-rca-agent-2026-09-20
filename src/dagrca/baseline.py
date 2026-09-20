"""Naive baseline: rank failed nodes by topological earliness.

A common ad hoc heuristic used when no statistical model is available: the
first task to fail, in DAG topological order, is guessed to be the root
cause. This ignores the fact that two independent failures can occur close
together in topological order, and it ignores transmission probabilities
entirely, but it is cheap, explainable, and a fair floor to compare against.
"""

from __future__ import annotations

from .dag import PipelineDAG


def rank_by_topological_earliness(dag: PipelineDAG, observed_failed: frozenset[str]) -> list[str]:
    return sorted(observed_failed, key=dag.nodes.index)
