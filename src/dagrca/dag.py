"""Synthetic pipeline DAG generation.

The DAG models a layered data pipeline, similar in spirit to a set of
orchestrated tasks in an Airflow-like scheduler: raw ingestion tasks feed
transform tasks, transform tasks feed load tasks, and load tasks feed
aggregate and reporting tasks. Edges represent data or control dependencies
along which a failure can (probabilistically) propagate downstream.

The generator is deterministic given a random seed, and it intentionally
creates at least one convergence point, a node with two or more parents
that share a common upstream ancestor, because that structural pattern is
central to the calibration analysis in ``propagation.py`` (see the
"multiple non-independent paths" discussion in the README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass(frozen=True)
class PipelineDAG:
    """A directed acyclic graph of pipeline tasks.

    Attributes:
        nodes: Topologically ordered list of node ids.
        edges: List of (parent, child) tuples. A parent failure can trigger
            a child failure with some transmission probability.
        edge_prob: Mapping from (parent, child) to transmission probability
            in (0, 1).
        leak_prob: Mapping from node id to its background (spontaneous)
            failure probability, independent of any parent.
        layer: Mapping from node id to a human-readable pipeline stage,
            used only for presentation and evidence generation.
    """

    nodes: list[str]
    edges: list[tuple[str, str]]
    edge_prob: dict[tuple[str, str], float]
    leak_prob: dict[str, float]
    layer: dict[str, str]
    parents: dict[str, list[str]] = field(init=False)
    children: dict[str, list[str]] = field(init=False)

    def __post_init__(self) -> None:
        parents: dict[str, list[str]] = {n: [] for n in self.nodes}
        children: dict[str, list[str]] = {n: [] for n in self.nodes}
        for u, v in self.edges:
            parents[v].append(u)
            children[u].append(v)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "children", children)
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        index = {n: i for i, n in enumerate(self.nodes)}
        for u, v in self.edges:
            if index[u] >= index[v]:
                raise ValueError(
                    f"edge ({u}, {v}) violates the topological order; "
                    "nodes must be listed in a valid topological order"
                )

    def ancestors(self, node: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.parents[node])
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(self.parents[p])
        return seen

    def descendants(self, node: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.children[node])
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            stack.extend(self.children[c])
        return seen

    def convergence_nodes(self) -> list[str]:
        """Nodes with two or more parents that share a common ancestor.

        These are exactly the nodes at which the mean-field forward
        propagation formula used in ``propagation.py`` is not exact: the
        formula treats parent failure events as conditionally independent
        given the root, which is only true when no two parents of a node
        share an ancestor (a polytree). When parents do share an ancestor,
        their failure events are positively correlated through that shared
        ancestor, and the mean-field formula is a biased approximation.
        """
        result = []
        for n in self.nodes:
            ps = self.parents[n]
            if len(ps) < 2:
                continue
            shared = False
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    if self.ancestors(ps[i]) & self.ancestors(ps[j]):
                        shared = True
                    if ps[i] in self.ancestors(ps[j]) or ps[j] in self.ancestors(ps[i]):
                        shared = True
            if shared:
                result.append(n)
        return result


def build_synthetic_pipeline_dag(seed: int = 7) -> PipelineDAG:
    """Builds a deterministic synthetic data-pipeline DAG.

    The pipeline has four stages: ingest -> transform -> load -> serve,
    loosely modeled on a batch orchestration DAG (extract raw telemetry or
    transactional tables, transform and join them, load into a warehouse or
    analytical store, then materialize aggregates and reports). Node names
    are generic and carry no proprietary or company-specific meaning.
    """
    rng = random.Random(seed)

    layer_defs = {
        "ingest": ["ingest_orders", "ingest_events", "ingest_reference"],
        "transform": ["clean_orders", "clean_events", "join_orders_events", "enrich_reference"],
        "load": ["load_warehouse_facts", "load_warehouse_dims"],
        "serve": ["agg_daily_metrics", "agg_customer_360", "publish_report"],
    }
    nodes = [n for stage in layer_defs.values() for n in stage]
    layer = {n: stage for stage, ns in layer_defs.items() for n in ns}

    edges: list[tuple[str, str]] = [
        ("ingest_orders", "clean_orders"),
        ("ingest_events", "clean_events"),
        ("ingest_reference", "enrich_reference"),
        ("clean_orders", "join_orders_events"),
        ("clean_events", "join_orders_events"),
        ("join_orders_events", "load_warehouse_facts"),
        ("enrich_reference", "load_warehouse_dims"),
        # convergence point: agg_daily_metrics has two parents that both
        # descend from ingest_orders / ingest_events via different paths.
        ("load_warehouse_facts", "agg_daily_metrics"),
        ("load_warehouse_dims", "agg_daily_metrics"),
        ("load_warehouse_facts", "agg_customer_360"),
        ("load_warehouse_dims", "agg_customer_360"),
        ("agg_daily_metrics", "publish_report"),
        ("agg_customer_360", "publish_report"),
    ]

    edge_prob = {(u, v): round(rng.uniform(0.55, 0.9), 3) for (u, v) in edges}
    leak_prob = {n: round(rng.uniform(0.01, 0.03), 3) for n in nodes}

    return PipelineDAG(nodes=nodes, edges=edges, edge_prob=edge_prob, leak_prob=leak_prob, layer=layer)
