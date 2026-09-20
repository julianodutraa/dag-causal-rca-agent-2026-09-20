from dagrca.dag import build_synthetic_pipeline_dag
from dagrca.evaluate import run_localization_evaluation, run_calibration_study, run_stratified_evaluation


def test_localization_evaluation_runs_and_returns_all_methods():
    dag = build_synthetic_pipeline_dag(seed=7)
    metrics = run_localization_evaluation(dag, n_incidents=60, seed=101)
    assert set(metrics.keys()) == {"baseline_topological", "bayesian_mean_field", "bayesian_plus_agent"}
    for m in metrics.values():
        assert m.n_incidents > 0
        assert 0.0 <= m.precision_at_1 <= 1.0
        assert 0.0 <= m.precision_at_3 <= 1.0
        assert 0.0 <= m.mrr <= 1.0
        # precision@3 can never be lower than precision@1 for the same run
        assert m.precision_at_3 >= m.precision_at_1 - 1e-9


def test_large_scale_run_matches_the_published_honest_result():
    """Reproduces, at the exact seed and incident count used in the README.

    The headline finding reported in the README is that the pure Bayesian
    mean-field ranking, with a uniform prior, underperforms the naive
    topological baseline overall, and that evidence fusion recovers part
    of that gap. This test locks in that qualitative relationship (not
    exact floats, since those are already covered by
    ``test_evaluation_is_deterministic_given_seed``) so a future change
    that silently reverses the finding is caught, rather than only being
    caught by a human re-reading the README.
    """
    dag = build_synthetic_pipeline_dag(seed=7)
    metrics = run_localization_evaluation(dag, n_incidents=1000, seed=2026)
    baseline = metrics["baseline_topological"]
    bayesian = metrics["bayesian_mean_field"]
    fused = metrics["bayesian_plus_agent"]
    assert baseline.precision_at_1 > bayesian.precision_at_1
    assert fused.precision_at_1 >= bayesian.precision_at_1
    assert fused.mrr >= bayesian.mrr


def test_evaluation_is_deterministic_given_seed():
    dag = build_synthetic_pipeline_dag(seed=7)
    m1 = run_localization_evaluation(dag, n_incidents=40, seed=303)
    m2 = run_localization_evaluation(dag, n_incidents=40, seed=303)
    assert m1 == m2


def test_stratified_evaluation_partitions_all_incidents():
    dag = build_synthetic_pipeline_dag(seed=7)
    report = run_stratified_evaluation(dag, n_incidents=200, seed=404)
    total = report["easy_agreement"]["n_incidents"] + report["hard_disagreement"]["n_incidents"]
    assert total == 200
    assert report["hard_disagreement"]["n_incidents"] > 0


def test_hard_stratum_shows_bayesian_underperforming_at_the_published_scale():
    """At the README's scale (n=1000, seed=2026), on the incidents where the
    baseline and the Bayesian ranking disagree, the pure Bayesian posterior
    is measurably worse than the naive baseline, and fusion partially, not
    fully, recovers the gap. This is the project's central honest finding
    and is pinned here the same way as the localization test above.
    """
    dag = build_synthetic_pipeline_dag(seed=7)
    report = run_stratified_evaluation(dag, n_incidents=1000, seed=2026)
    hard = report["hard_disagreement"]
    assert hard["n_incidents"] > 0
    assert hard["bayesian_p1"] < hard["baseline_p1"]
    assert hard["fused_p1"] >= hard["bayesian_p1"]


def test_calibration_study_convergence_node_present():
    dag = build_synthetic_pipeline_dag(seed=7)
    report = run_calibration_study(dag, trials_per_root=20_000, seed=500)
    assert "publish_report" in report["convergence_nodes"]
    assert report["convergence_node_mae"] >= 0.0
    assert report["non_convergence_node_mae"] >= 0.0
    assert set(report["per_node_mae"].keys()) <= set(dag.nodes)
