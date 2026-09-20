from dagrca.dag import build_synthetic_pipeline_dag
from dagrca.propagation import simulate_incident
from dagrca.evidence import generate_evidence
from dagrca.llm_agent import StubLLMClient, fuse_scores
import random


def _get_incident(root: str, min_failed: int = 3):
    dag = build_synthetic_pipeline_dag(seed=7)
    rng = random.Random(17)
    for _ in range(200):
        incident = simulate_incident(dag, root, rng)
        if len(incident.failed) >= min_failed:
            return dag, incident
    raise AssertionError("could not find a large-enough incident in 200 tries")


def test_evidence_generated_for_every_failed_node():
    dag, incident = _get_incident("ingest_orders")
    evidence = generate_evidence(dag, incident.root, incident.failed, seed=1)
    assert set(evidence.keys()) == set(incident.failed)


def test_evidence_is_deterministic_given_seed():
    dag, incident = _get_incident("ingest_orders")
    e1 = generate_evidence(dag, incident.root, incident.failed, seed=99)
    e2 = generate_evidence(dag, incident.root, incident.failed, seed=99)
    assert e1 == e2


def test_stub_llm_gives_higher_score_to_incriminated_node():
    dag, incident = _get_incident("ingest_orders")
    # force high incrimination probability for the root, zero red herrings,
    # to check the agent responds correctly to unambiguous evidence.
    evidence = generate_evidence(
        dag, incident.root, incident.failed, seed=2, red_herring_prob=0.0, incriminating_prob_for_root=1.0
    )
    candidates = sorted(incident.failed, key=dag.nodes.index)
    agent = StubLLMClient()
    verdicts = agent.score_candidates(candidates, evidence)
    scores = {v.node: v.evidence_score for v in verdicts}
    root_score = scores[incident.root]
    other_scores = [s for n, s in scores.items() if n != incident.root]
    if other_scores:
        assert root_score >= max(other_scores)


def test_fuse_scores_preserves_all_candidates():
    dag, incident = _get_incident("ingest_orders")
    bayes_scores = {n: -1.0 * i for i, n in enumerate(sorted(incident.failed, key=dag.nodes.index))}
    evidence = generate_evidence(dag, incident.root, incident.failed, seed=3)
    agent = StubLLMClient()
    verdicts = agent.score_candidates(list(bayes_scores.keys()), evidence)
    fused = fuse_scores(bayes_scores, verdicts, fusion_weight=0.5)
    assert set(fused.keys()) == set(bayes_scores.keys())


def test_fusion_weight_zero_reduces_to_bayesian_order():
    dag, incident = _get_incident("ingest_orders")
    bayes_scores = {n: float(i) for i, n in enumerate(sorted(incident.failed, key=dag.nodes.index))}
    evidence = generate_evidence(dag, incident.root, incident.failed, seed=4)
    agent = StubLLMClient()
    verdicts = agent.score_candidates(list(bayes_scores.keys()), evidence)
    fused = fuse_scores(bayes_scores, verdicts, fusion_weight=0.0)
    assert fused == bayes_scores
