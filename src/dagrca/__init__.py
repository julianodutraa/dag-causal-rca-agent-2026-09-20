"""dagrca: causal root-cause localization for data pipeline DAGs.

This package implements a noisy-OR causal model of failure propagation over
a directed acyclic graph (DAG) of pipeline tasks, an approximate (mean-field)
closed-form inference routine for ranking candidate root causes, a Monte
Carlo ground-truth simulator used to both generate synthetic incidents and
to measure the approximation error of the closed-form routine, and a
tool-augmented evidence fusion agent (LLM-backed, with a deterministic stub
implementation for reproducible offline evaluation) that combines textual
evidence with the statistical posterior.
"""

__version__ = "0.1.0"
