"""Synthetic per-node evidence generation for the evidence-fusion agent.

For every node that failed in an incident, we synthesize three short text
snippets that mimic what an on-call engineer would actually pull up while
triaging: a deployment log line, an error message, and a metric snapshot
description. The true root cause is more likely (but not certain) to show
an incriminating deploy or config-change signal; non-root nodes usually
show generic downstream failure symptoms, but occasionally show an
unrelated, coincidental deploy event too (a red herring), which is what
makes the evidence genuinely ambiguous rather than a trivial lookup.

This is a synthetic stand-in for the kind of unstructured operational
telemetry (log lines, deploy events, metric snapshots) that a real
observability stack would provide; no proprietary system, product, or log
format is modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from .dag import PipelineDAG

_INCRIMINATING_DEPLOYS = [
    "deploy at {t}: config change to connection pool size (5 -> 2) on {node}",
    "deploy at {t}: schema migration applied to upstream table feeding {node}",
    "deploy at {t}: credential rotation completed for {node}, new secret not yet propagated",
    "deploy at {t}: dependency upgrade on {node} changed default timeout from 30s to 5s",
]
_NEUTRAL_DEPLOYS = [
    "no deploys recorded for {node} in the incident window",
    "deploy at {t}: unrelated dashboard label update on {node}, no logic change",
    "no deploys recorded for {node} in the incident window",
]
_INCRIMINATING_ERRORS = [
    "ERROR {node}: connection pool exhausted, 0 available connections",
    "ERROR {node}: authentication failed, secret rejected by downstream service",
    "ERROR {node}: read timeout after 5000ms, previously 30000ms",
]
_GENERIC_ERRORS = [
    "ERROR {node}: upstream dependency did not produce output within SLA",
    "ERROR {node}: task marked failed, upstream task {node} did not succeed",
    "WARN {node}: retry exhausted after 3 attempts, upstream unavailable",
]
_INCRIMINATING_METRICS = [
    "metric snapshot {node}: active_connections spiked to pool_max immediately before failure",
    "metric snapshot {node}: p99 latency increased 8x in the 5 minutes before failure",
]
_NEUTRAL_METRICS = [
    "metric snapshot {node}: resource utilization within normal historical range",
    "metric snapshot {node}: no anomalous metric pattern in the incident window",
]


@dataclass(frozen=True)
class NodeEvidence:
    node: str
    deploy_log: str
    error_message: str
    metric_snapshot: str


def generate_evidence(
    dag: PipelineDAG,
    true_root: str,
    observed_failed: frozenset[str],
    seed: int,
    red_herring_prob: float = 0.15,
    incriminating_prob_for_root: float = 0.75,
) -> dict[str, NodeEvidence]:
    """Generates one evidence bundle per failed node, keyed by node id.

    The evidence is intentionally imperfect: with probability
    ``1 - incriminating_prob_for_root`` even the true root shows only
    generic symptoms (a real incident where the deploy log genuinely has
    nothing useful in it), and with probability ``red_herring_prob`` a
    non-root failed node shows an incriminating-looking but unrelated
    deploy event.
    """
    rng = random.Random(seed)
    evidence: dict[str, NodeEvidence] = {}
    for node in sorted(observed_failed, key=dag.nodes.index):
        is_root = node == true_root
        show_incriminating = (
            rng.random() < incriminating_prob_for_root if is_root else rng.random() < red_herring_prob
        )
        t = f"T-{rng.randint(1, 45)}min"
        if show_incriminating:
            deploy = rng.choice(_INCRIMINATING_DEPLOYS).format(node=node, t=t)
            error = rng.choice(_INCRIMINATING_ERRORS).format(node=node)
            metric = rng.choice(_INCRIMINATING_METRICS).format(node=node)
        else:
            deploy = rng.choice(_NEUTRAL_DEPLOYS).format(node=node, t=t)
            error = rng.choice(_GENERIC_ERRORS).format(node=node)
            metric = rng.choice(_NEUTRAL_METRICS).format(node=node)
        evidence[node] = NodeEvidence(node=node, deploy_log=deploy, error_message=error, metric_snapshot=metric)
    return evidence
