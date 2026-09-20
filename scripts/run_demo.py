#!/usr/bin/env python3
"""End-to-end reproducible demo.

Runs the calibration study and the root-cause localization evaluation with
fixed seeds, prints a human-readable summary, and writes the measured
results to ``results/`` as JSON so the numbers in the README are exactly
what this script produces (no numbers in this repository are hand-edited
after the fact).

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dagrca.dag import build_synthetic_pipeline_dag
from dagrca.evaluate import run_calibration_study, run_localization_evaluation, run_stratified_evaluation
from dagrca.propagation import simulate_incident
from dagrca.inference import rank_root_causes
from dagrca.evidence import generate_evidence
from dagrca.llm_agent import StubLLMClient, fuse_scores
import random


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    dag = build_synthetic_pipeline_dag(seed=7)

    print("=== Calibration study: mean-field vs Monte Carlo ground truth ===")
    calibration = run_calibration_study(dag, trials_per_root=50_000, seed=500)
    print(f"Convergence node(s): {calibration['convergence_nodes']}")
    print(f"Mean absolute error at convergence nodes:     {calibration['convergence_node_mae']:.4f}")
    print(f"Mean absolute error at non-convergence nodes:  {calibration['non_convergence_node_mae']:.4f}")
    ratio = calibration["convergence_node_mae"] / max(calibration["non_convergence_node_mae"], 1e-9)
    print(f"Ratio (convergence / non-convergence):         {ratio:.2f}x")
    (results_dir / "calibration_report.json").write_text(json.dumps(calibration, indent=2))

    print("\n=== Root-cause localization evaluation (1000 synthetic incidents) ===")
    metrics = run_localization_evaluation(dag, n_incidents=1000, seed=2026)
    metrics_json = {
        name: {
            "precision_at_1": m.precision_at_1,
            "precision_at_3": m.precision_at_3,
            "mrr": m.mrr,
            "n_incidents": m.n_incidents,
        }
        for name, m in metrics.items()
    }
    for name, m in metrics_json.items():
        print(
            f"{name:24s} P@1={m['precision_at_1']:.3f}  P@3={m['precision_at_3']:.3f}  "
            f"MRR={m['mrr']:.3f}  (n={m['n_incidents']})"
        )
    (results_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    print("\n=== Stratified evaluation: incidents where baseline and Bayesian disagree ===")
    strat = run_stratified_evaluation(dag, n_incidents=1000, seed=2026)
    for stratum, row in strat.items():
        if row.get("n_incidents"):
            print(
                f"{stratum:18s} n={row['n_incidents']:4d}  baseline_P@1={row['baseline_p1']:.3f}  "
                f"bayesian_P@1={row['bayesian_p1']:.3f}  fused_P@1={row['fused_p1']:.3f}"
            )
    (results_dir / "stratified_report.json").write_text(json.dumps(strat, indent=2))

    print("\n=== Fusion weight sweep (does the LLM evidence agent help or hurt?) ===")
    sweep = {}
    for weight in [0.0, 0.3, 0.6, 1.0, 1.5]:
        m = run_localization_evaluation(dag, n_incidents=500, seed=42, fusion_weight=weight)
        sweep[str(weight)] = {"precision_at_1": m["bayesian_plus_agent"].precision_at_1, "mrr": m["bayesian_plus_agent"].mrr}
        print(f"fusion_weight={weight:<4} P@1={m['bayesian_plus_agent'].precision_at_1:.3f}  MRR={m['bayesian_plus_agent'].mrr:.3f}")
    (results_dir / "fusion_weight_sweep.json").write_text(json.dumps(sweep, indent=2))

    print("\n=== Example incident (for the README and the demo slide) ===")
    rng = random.Random(9001)
    true_root = "join_orders_events"
    incident = simulate_incident(dag, true_root, rng)
    while len(incident.failed) < 4:
        incident = simulate_incident(dag, true_root, rng)
    bayes_ranked = rank_root_causes(dag, incident.failed)
    evidence = generate_evidence(dag, true_root, incident.failed, seed=77)
    top3 = [c.node for c in bayes_ranked[:3]]
    verdicts = StubLLMClient().score_candidates(top3, evidence)
    bayes_scores = {c.node: c.log_score for c in bayes_ranked}
    fused = fuse_scores(bayes_scores, verdicts, fusion_weight=0.6)
    fused_ranking = sorted(fused, key=lambda n: fused[n], reverse=True)

    example = {
        "true_root": true_root,
        "observed_failed": sorted(incident.failed, key=dag.nodes.index),
        "bayesian_ranking": [{"node": c.node, "posterior": round(c.posterior, 4)} for c in bayes_ranked],
        "agent_verdicts_on_top3": [
            {"node": v.node, "evidence_score": round(v.evidence_score, 4), "rationale": v.rationale} for v in verdicts
        ],
        "fused_ranking": fused_ranking,
    }
    print(json.dumps(example, indent=2))
    (results_dir / "example_incident.json").write_text(json.dumps(example, indent=2))

    print(f"\nWrote results to {results_dir}")


if __name__ == "__main__":
    main()
