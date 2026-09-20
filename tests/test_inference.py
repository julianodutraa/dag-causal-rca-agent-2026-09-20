from dagrca.dag import build_synthetic_pipeline_dag
from dagrca.inference import rank_root_causes, log_likelihood_observed_failures
from dagrca.propagation import simulate_incident
import random


def test_ranking_posterior_sums_to_one():
    dag = build_synthetic_pipeline_dag(seed=7)
    incident = simulate_incident(dag, root="ingest_orders", rng=random.Random(3))
    if len(incident.failed) < 2:
        incident = simulate_incident(dag, root="ingest_orders", rng=random.Random(11))
    ranked = rank_root_causes(dag, incident.failed)
    total = sum(c.posterior for c in ranked)
    assert abs(total - 1.0) < 1e-9


def test_ranking_descending_by_log_score():
    dag = build_synthetic_pipeline_dag(seed=7)
    incident = simulate_incident(dag, root="clean_orders", rng=random.Random(5))
    if len(incident.failed) < 2:
        incident = simulate_incident(dag, root="clean_orders", rng=random.Random(9))
    ranked = rank_root_causes(dag, incident.failed)
    scores = [c.log_score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_true_root_is_plausible_hypothesis():
    dag = build_synthetic_pipeline_dag(seed=7)
    # actual failure sets can only be explained by a node that is itself
    # in the observed failed set (a node can only propagate failure if it
    # failed), so the true root must appear among the ranked candidates.
    incident = simulate_incident(dag, root="ingest_reference", rng=random.Random(21))
    ranked = rank_root_causes(dag, incident.failed)
    ranked_nodes = {c.node for c in ranked}
    assert "ingest_reference" in ranked_nodes


def test_log_likelihood_is_higher_for_true_root_on_average():
    dag = build_synthetic_pipeline_dag(seed=7)
    rng = random.Random(123)
    wins = 0
    trials = 0
    for _ in range(200):
        root = "ingest_orders"
        incident = simulate_incident(dag, root, rng)
        if len(incident.failed) < 3:
            continue
        trials += 1
        candidates = [c for c in incident.failed if c != root]
        if not candidates:
            continue
        other = rng.choice(candidates)
        ll_true = log_likelihood_observed_failures(dag, root, incident.failed)
        ll_other = log_likelihood_observed_failures(dag, other, incident.failed)
        if ll_true >= ll_other:
            wins += 1
    assert trials > 20
    assert wins / trials > 0.7
