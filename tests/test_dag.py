from dagrca.dag import build_synthetic_pipeline_dag, PipelineDAG


def test_dag_is_valid_topological_order():
    dag = build_synthetic_pipeline_dag(seed=7)
    index = {n: i for i, n in enumerate(dag.nodes)}
    for u, v in dag.edges:
        assert index[u] < index[v]


def test_edge_and_leak_probabilities_are_valid():
    dag = build_synthetic_pipeline_dag(seed=7)
    for p in dag.edge_prob.values():
        assert 0.0 < p < 1.0
    for p in dag.leak_prob.values():
        assert 0.0 < p < 1.0


def test_parents_children_consistency():
    dag = build_synthetic_pipeline_dag(seed=7)
    for u, v in dag.edges:
        assert u in dag.parents[v]
        assert v in dag.children[u]


def test_convergence_node_detected():
    dag = build_synthetic_pipeline_dag(seed=7)
    convergent = dag.convergence_nodes()
    assert "publish_report" in convergent
    # simple two-parent nodes whose parents share no ancestry should not be flagged
    assert "agg_daily_metrics" not in convergent
    assert "agg_customer_360" not in convergent


def test_deterministic_construction():
    dag1 = build_synthetic_pipeline_dag(seed=7)
    dag2 = build_synthetic_pipeline_dag(seed=7)
    assert dag1.edge_prob == dag2.edge_prob
    assert dag1.leak_prob == dag2.leak_prob


def test_rejects_non_topological_edges():
    import pytest

    with pytest.raises(ValueError):
        PipelineDAG(
            nodes=["a", "b"],
            edges=[("b", "a")],
            edge_prob={("b", "a"): 0.5},
            leak_prob={"a": 0.01, "b": 0.01},
            layer={"a": "x", "b": "x"},
        )
