"""Evaluation harness: root-cause localization metrics and calibration study.

Two independent measurements are produced:

1. Root-cause localization accuracy (``run_localization_evaluation``): over
   many simulated incidents with a known true root, rank candidate roots
   with three methods (topological baseline, Bayesian mean-field posterior,
   Bayesian posterior fused with the evidence agent) and report Precision@1,
   Precision@3, and Mean Reciprocal Rank (MRR) for each.

2. Mean-field calibration (``run_calibration_study``): compare the
   closed-form mean-field marginals against Monte Carlo ground truth,
   separately for convergence nodes and non-convergence nodes, to quantify
   the bias introduced by the independence-of-causal-influence assumption.

Both are deterministic given a seed, and both write their measured (not
cherry-picked) numbers to ``results/`` when invoked through
``scripts/run_demo.py``.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from .dag import PipelineDAG
from .propagation import mean_field_marginals, monte_carlo_marginals, simulate_incident
from .inference import rank_root_causes, log_likelihood_observed_failures
from .baseline import rank_by_topological_earliness
from .evidence import generate_evidence
from .llm_agent import LLMClient, StubLLMClient, fuse_scores


def _reciprocal_rank(ranking: list[str], true_root: str) -> float:
    if true_root not in ranking:
        return 0.0
    return 1.0 / (ranking.index(true_root) + 1)


def _precision_at_k(ranking: list[str], true_root: str, k: int) -> float:
    return 1.0 if true_root in ranking[:k] else 0.0


@dataclass(frozen=True)
class MethodMetrics:
    precision_at_1: float
    precision_at_3: float
    mrr: float
    n_incidents: int


def run_localization_evaluation(
    dag: PipelineDAG,
    n_incidents: int,
    seed: int,
    fusion_weight: float = 0.6,
    top_k_for_agent: int = 3,
    llm_client: LLMClient | None = None,
    min_failed_for_incident: int = 2,
) -> dict[str, MethodMetrics]:
    """Runs the full evaluation and returns metrics per method.

    Only incidents where at least ``min_failed_for_incident`` nodes failed
    are kept (an incident where only the root itself failed carries no
    localization signal at all and would trivially inflate every method's
    accuracy equally, which would be a misleading number to publish).
    """
    rng = random.Random(seed)
    agent = llm_client or StubLLMClient()

    results = {
        "baseline_topological": [],
        "bayesian_mean_field": [],
        "bayesian_plus_agent": [],
    }
    kept = 0
    attempts = 0
    while kept < n_incidents and attempts < n_incidents * 20:
        attempts += 1
        true_root = rng.choice(dag.nodes)
        sim_seed = rng.randint(0, 2**31 - 1)
        incident = simulate_incident(dag, true_root, random.Random(sim_seed))
        if len(incident.failed) < min_failed_for_incident:
            continue
        kept += 1

        baseline_ranking = rank_by_topological_earliness(dag, incident.failed)
        results["baseline_topological"].append(
            (
                _precision_at_k(baseline_ranking, true_root, 1),
                _precision_at_k(baseline_ranking, true_root, 3),
                _reciprocal_rank(baseline_ranking, true_root),
            )
        )

        bayes_ranked = rank_root_causes(dag, incident.failed)
        bayes_ranking = [c.node for c in bayes_ranked]
        results["bayesian_mean_field"].append(
            (
                _precision_at_k(bayes_ranking, true_root, 1),
                _precision_at_k(bayes_ranking, true_root, 3),
                _reciprocal_rank(bayes_ranking, true_root),
            )
        )

        top_candidates = bayes_ranking[:top_k_for_agent]
        evidence_seed = rng.randint(0, 2**31 - 1)
        evidence = generate_evidence(dag, true_root, incident.failed, seed=evidence_seed)
        verdicts = agent.score_candidates(top_candidates, evidence)
        bayes_log_scores = {c.node: c.log_score for c in bayes_ranked}
        fused_scores = fuse_scores(bayes_log_scores, verdicts, fusion_weight=fusion_weight)
        fused_ranking = sorted(fused_scores, key=lambda n: fused_scores[n], reverse=True)
        # candidates outside the agent's top-K keep their original relative
        # order, appended after the fused (re-ranked) top-K.
        remainder = [n for n in bayes_ranking if n not in top_candidates]
        fused_ranking = fused_ranking[: len(top_candidates)] + remainder
        results["bayesian_plus_agent"].append(
            (
                _precision_at_k(fused_ranking, true_root, 1),
                _precision_at_k(fused_ranking, true_root, 3),
                _reciprocal_rank(fused_ranking, true_root),
            )
        )

    metrics = {}
    for method, rows in results.items():
        p1 = statistics.fmean(r[0] for r in rows)
        p3 = statistics.fmean(r[1] for r in rows)
        mrr = statistics.fmean(r[2] for r in rows)
        metrics[method] = MethodMetrics(precision_at_1=p1, precision_at_3=p3, mrr=mrr, n_incidents=len(rows))
    return metrics


def run_stratified_evaluation(
    dag: PipelineDAG,
    n_incidents: int,
    seed: int,
    fusion_weight: float = 0.6,
    top_k_for_agent: int = 3,
    llm_client: LLMClient | None = None,
    min_failed_for_incident: int = 2,
) -> dict[str, dict]:
    """Splits incidents into "easy" and "hard" strata and reports P@1 per stratum.

    An incident is labeled "hard" when the topological baseline's top-1
    guess disagrees with the Bayesian mean-field posterior's top-1 guess,
    that is, exactly the incidents where the naive heuristic and the
    statistical model give conflicting advice. On agreement ("easy")
    incidents both methods are trivially correct or trivially wrong
    together, so the disagreement subset is where a method's real
    discriminative power shows up, and it is the fairest place to check
    whether the evidence agent is pulling its weight.
    """
    rng = random.Random(seed)
    agent = llm_client or StubLLMClient()

    strata = {"easy_agreement": [], "hard_disagreement": []}
    kept = 0
    attempts = 0
    while kept < n_incidents and attempts < n_incidents * 20:
        attempts += 1
        true_root = rng.choice(dag.nodes)
        sim_seed = rng.randint(0, 2**31 - 1)
        incident = simulate_incident(dag, true_root, random.Random(sim_seed))
        if len(incident.failed) < min_failed_for_incident:
            continue
        kept += 1

        baseline_ranking = rank_by_topological_earliness(dag, incident.failed)
        bayes_ranked = rank_root_causes(dag, incident.failed)
        bayes_ranking = [c.node for c in bayes_ranked]

        top_candidates = bayes_ranking[:top_k_for_agent]
        evidence_seed = rng.randint(0, 2**31 - 1)
        evidence = generate_evidence(dag, true_root, incident.failed, seed=evidence_seed)
        verdicts = agent.score_candidates(top_candidates, evidence)
        bayes_log_scores = {c.node: c.log_score for c in bayes_ranked}
        fused_scores = fuse_scores(bayes_log_scores, verdicts, fusion_weight=fusion_weight)
        fused_ranking = sorted(fused_scores, key=lambda n: fused_scores[n], reverse=True)
        remainder = [n for n in bayes_ranking if n not in top_candidates]
        fused_ranking = fused_ranking[: len(top_candidates)] + remainder

        stratum = "easy_agreement" if baseline_ranking[0] == bayes_ranking[0] else "hard_disagreement"
        strata[stratum].append(
            {
                "baseline_p1": _precision_at_k(baseline_ranking, true_root, 1),
                "bayesian_p1": _precision_at_k(bayes_ranking, true_root, 1),
                "fused_p1": _precision_at_k(fused_ranking, true_root, 1),
            }
        )

    report = {}
    for stratum, rows in strata.items():
        if not rows:
            report[stratum] = {"n_incidents": 0}
            continue
        report[stratum] = {
            "n_incidents": len(rows),
            "baseline_p1": statistics.fmean(r["baseline_p1"] for r in rows),
            "bayesian_p1": statistics.fmean(r["bayesian_p1"] for r in rows),
            "fused_p1": statistics.fmean(r["fused_p1"] for r in rows),
        }
    return report


def run_calibration_study(dag: PipelineDAG, trials_per_root: int, seed: int) -> dict:
    """Measures mean-field approximation error against Monte Carlo ground truth.

    Returns a dict with per-node mean absolute error (MAE) between the
    closed-form marginal and the Monte Carlo estimate, averaged over all
    choices of root, plus the same MAE aggregated separately over
    convergence nodes and non-convergence nodes.
    """
    convergence = set(dag.convergence_nodes())
    per_node_errors: dict[str, list[float]] = {n: [] for n in dag.nodes}

    for i, root in enumerate(dag.nodes):
        mf = mean_field_marginals(dag, root)
        mc = monte_carlo_marginals(dag, root, trials=trials_per_root, seed=seed + i)
        for node in dag.nodes:
            if node == root:
                continue
            per_node_errors[node].append(abs(mf[node] - mc[node]))

    per_node_mae = {n: statistics.fmean(v) for n, v in per_node_errors.items() if v}
    convergence_mae = statistics.fmean(v for n, v in per_node_mae.items() if n in convergence)
    non_convergence_mae = statistics.fmean(v for n, v in per_node_mae.items() if n not in convergence)

    return {
        "per_node_mae": per_node_mae,
        "convergence_nodes": sorted(convergence),
        "convergence_node_mae": convergence_mae,
        "non_convergence_node_mae": non_convergence_mae,
        "trials_per_root": trials_per_root,
    }
