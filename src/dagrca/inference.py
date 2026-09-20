"""Bayesian root-cause ranking from an observed failure set.

Given an observed incident, the set ``F`` of nodes that failed, we want a
posterior distribution over which node was the true root cause. Exact
inference under the noisy-OR model requires reasoning jointly over all of
the independent leak and transmission bits, which is intractable to do in
closed form once the graph has convergence points. Instead we use the
standard product-of-marginals (mean-field) approximate likelihood: treat
each node's fail/not-fail status as conditionally independent given the
hypothesized root, using the marginals from ``propagation.mean_field_marginals``:

    P(F | root = r) ~= prod_{v in F} P(v fails | r) * prod_{v not in F} (1 - P(v fails | r))

Combined with a prior over candidate roots (uniform by default, but any
node can be weighted, e.g. by historical incident base rates), this gives
an (approximate) posterior

    P(root = r | F) ~= prior(r) * P(F | r) / normalizer

Because this is a probability, not a hard classification, the output is a
full ranked list with scores, which is what both the evaluation harness
(``evaluate.py``) and the LLM evidence fusion (``llm_agent.py``) consume.

A small numerical floor (``EPS``) is applied to every marginal to keep the
log-likelihood finite when a node's mean-field failure probability rounds
to exactly 0 or 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .dag import PipelineDAG
from .propagation import mean_field_marginals

EPS = 1e-6


@dataclass(frozen=True)
class RankedCandidate:
    node: str
    log_score: float
    posterior: float


def log_likelihood_observed_failures(
    dag: PipelineDAG, root: str, observed_failed: frozenset[str]
) -> float:
    """log P(observed_failed | root), under the product-of-marginals approximation."""
    marginals = mean_field_marginals(dag, root)
    log_lik = 0.0
    for node in dag.nodes:
        p_fail = min(max(marginals[node], EPS), 1.0 - EPS)
        if node in observed_failed:
            log_lik += math.log(p_fail)
        else:
            log_lik += math.log(1.0 - p_fail)
    return log_lik


def rank_root_causes(
    dag: PipelineDAG,
    observed_failed: frozenset[str],
    candidates: list[str] | None = None,
    prior: dict[str, float] | None = None,
) -> list[RankedCandidate]:
    """Ranks candidate root causes by approximate posterior probability, descending.

    Only nodes present in ``observed_failed`` are plausible roots under this
    generative model (a node that never failed cannot be the root, since the
    root fails with probability 1), so by default ``candidates`` is exactly
    ``observed_failed``.
    """
    if candidates is None:
        candidates = sorted(observed_failed, key=dag.nodes.index)
    if prior is None:
        prior = {c: 1.0 / len(candidates) for c in candidates}

    log_scores: dict[str, float] = {}
    for r in candidates:
        log_prior = math.log(max(prior.get(r, 1e-9), 1e-12))
        log_scores[r] = log_prior + log_likelihood_observed_failures(dag, r, observed_failed)

    max_log = max(log_scores.values())
    exp_scores = {r: math.exp(s - max_log) for r, s in log_scores.items()}
    total = sum(exp_scores.values())
    posteriors = {r: v / total for r, v in exp_scores.items()}

    ranked = sorted(candidates, key=lambda r: log_scores[r], reverse=True)
    return [RankedCandidate(node=r, log_score=log_scores[r], posterior=posteriors[r]) for r in ranked]
