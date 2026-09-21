# Causal Root Cause Localization for Data Pipeline DAGs

A noisy OR Bayesian network over a pipeline dependency graph, combined with a tool augmented LLM evidence agent, used to rank candidate root causes when a batch orchestration DAG (in the style of Airflow) fails across several tasks at once.

## Executive summary

When a data pipeline DAG fails, it rarely fails at a single task. A source ingestion problem, a schema change, or a resource exhaustion event upstream tends to cascade: five, ten, sometimes dozens of downstream tasks go red within minutes, and the on call engineer is left staring at a wall of failed tasks with no direct indication of which one actually started the incident. Time to root cause is one of the largest components of total incident cost in data platform operations: every minute spent scanning failed tasks in topological order, or reading logs one task at a time, is a minute of delayed remediation, delayed stakeholder communication, and in the worst case, delayed detection of a data quality problem that keeps propagating into downstream reports and decisions.

This project treats root cause localization as a structured inference problem instead of a manual troubleshooting checklist. The pipeline's task dependency graph is modeled as a causal Bayesian network: each edge carries a probability that a failure transmits from a parent task to a child task, learned or configured per pipeline, and inference over this network produces a ranked, probabilistic list of which task most likely started the incident, given only the set of tasks observed to have failed. A second component, a tool augmented evidence agent, reads the same operational signals an engineer would (deploy logs, error messages, metric snapshots) and fuses that qualitative evidence with the statistical ranking.

The honest headline result of this project is nuanced, and that nuance is the point. On this synthetic pipeline, a naive heuristic (the earliest failed task in topological order) is a strong baseline, and a purely statistical Bayesian ranking with an uninformative prior actually underperforms it overall (86.7 percent versus 88.2 percent top one accuracy over 1000 simulated incidents). The value of the statistical and agentic layers shows up specifically on the harder incidents, the roughly six percent of cases where the naive heuristic and the Bayesian ranking disagree about which task is the true root cause: there, evidence fusion measurably recovers accuracy that the pure statistical model loses. That is the kind of result a rigorous evaluation is supposed to surface: not a cherry picked win, but a precise statement of where a more sophisticated method earns its complexity and where it does not yet. For a platform team deciding whether to invest in this kind of tooling, that is a more useful answer than an inflated aggregate number, because it tells you exactly which incidents the tool will help with.

## Problem framing

Consider an orchestrated batch pipeline with four stages: ingestion, transformation, warehouse loading, and serving (daily metrics, a customer facing aggregate table, and a downstream report). Failures propagate along data and control dependencies: if an ingestion task fails or produces bad data, every task that reads its output is at risk of failing too, with some probability that depends on how tightly coupled the tasks are and how failure prone the transmission path is (a hard dependency failing usually cascades close to deterministically, while a soft dependency, like a cached fallback, might not).

Given only the set of tasks that failed in a given incident window, the objective is to rank the tasks by how likely each one is to be the actual root cause, and to do so in a way that is calibrated (the reported probabilities are honest estimates, not just an arbitrary score) and that can incorporate unstructured operational evidence (logs, metrics, deploy history) alongside the graph structure itself.

## Method

### 1. Generative model: noisy OR failure propagation

Fix a root cause task `r` that fails with probability 1. Every other task `v` has an independent background failure probability (a leak term, representing causes external to the graph, such as an unrelated infrastructure blip), and every edge `(u, v)` has an independent transmission probability. A task fails if and only if its own leak fires, or at least one of its parents failed and successfully transmitted the failure:

```
Fails(v) = Leak_v OR (OR over parents u of v of (Transmit_uv AND Fails(u)))
```

This is the standard noisy OR causal independence model (Pearl, 1988): each parent is treated as an independent sufficient cause. It is implemented in `src/dagrca/propagation.py` both as an explicit Monte Carlo simulator (used as ground truth and as the incident generator for evaluation) and as a closed form recursion.

### 2. Closed form inference and its calibration limit

The closed form (mean field) recursion computes each node's marginal failure probability directly, in one topological pass:

```
P(v fails) = 1 - (1 - leak_prob[v]) * prod over parents u of v of (1 - edge_prob[u, v] * P(u fails))
```

This formula is exact when the parents of every node are conditionally independent given the root, which holds on a polytree (no two parents of any node share a common ancestor). On a general DAG with convergence points, where two parents of the same task trace back to a shared upstream ancestor, that independence assumption is violated: the two parents' failure events are positively correlated through the shared ancestor, and the closed form marginal is a biased approximation.

This project measures that bias directly rather than asserting it. `run_calibration_study` compares the closed form marginal against a 50000 trial Monte Carlo estimate, separately at the graph's one convergence node (`publish_report`, whose two parents both trace back through the warehouse loading stage) and at every non convergence node. The measured result: mean absolute error of 0.0280 at the convergence node versus 0.0011 at non convergence nodes, roughly a 26 fold gap. This is the kind of calibration bug that is easy to ship silently: the formula still returns a valid looking probability in [0, 1], it is just wrong at exactly the structurally interesting nodes, the ones with more than one upstream path, which are disproportionately likely to be the nodes an operator actually cares about (aggregation and reporting tasks near the end of a pipeline). See `results/calibration_report.json` for the full per node breakdown.

### 3. Bayesian root cause ranking

Exact posterior inference under the full noisy OR joint distribution is intractable once the graph has convergence points, so root cause ranking uses the standard product of marginals (mean field) approximate likelihood: treat each task's observed fail or not fail status as conditionally independent given the hypothesized root, and combine with a prior over candidates (uniform by default in this project, to isolate the discriminative power of the likelihood model on its own, see the limitations section for why this choice matters):

```
P(observed failures | root = r) ~= product over failed v of P(v fails | r) times product over non failed v of (1 - P(v fails | r))
```

Bayes' rule then gives a normalized posterior over which task is the true root cause. Implementation: `src/dagrca/inference.py`.

### 4. Tool augmented evidence agent and log linear fusion

A second, independent signal comes from reading unstructured operational evidence: a synthetic deploy log line, an error message, and a metric snapshot description for each of the top ranked candidates (`src/dagrca/evidence.py` generates these; they are deliberately noisy, with roughly a 15 percent chance that a non root task shows an unrelated, coincidentally incriminating deploy event, a red herring). The evidence agent (`src/dagrca/llm_agent.py`) reads those fields and returns a probability per candidate that it is the true root cause.

The agent is implemented behind a small `LLMClient` protocol with two concrete backends: a deterministic, dependency free `StubLLMClient` used by the test suite and the default evaluation run (so the entire project is reproducible offline, at zero cost, with no external API dependency), and an optional `AnthropicLLMClient` that wires the same interface to a real Claude model through the `anthropic` SDK for interactive use. The Bayesian log posterior and the agent's evidence score are combined by log linear fusion (a product of experts), a standard way to merge two independently calibrated probabilistic opinions:

```
fused_score(r) = bayesian_log_posterior(r) + fusion_weight * log(agent_evidence_score(r))
```

### 5. Evaluation methodology

`scripts/run_demo.py` runs, with fixed seeds, on the synthetic pipeline DAG in `src/dagrca/dag.py`:

* A calibration study (closed form versus Monte Carlo, 50000 trials per root).
* A localization evaluation over 1000 simulated incidents, comparing the topological baseline, the pure Bayesian ranking, and the Bayesian plus agent fused ranking, on Precision at 1, Precision at 3, and Mean Reciprocal Rank. Incidents where the root's failure did not propagate to any other task (fewer than 2 failed tasks) are excluded, since they carry no localization signal for any method and would inflate every method's score identically.
* A stratified evaluation that separates incidents into the subset where the baseline and the Bayesian ranking agree on the top candidate, and the subset where they disagree, since agreement cases are uninformative about which method is actually better.
* A fusion weight sweep, to check whether the evidence agent helps monotonically or has a clear operating point.

## Honest, measured results

All numbers below are exactly what `python scripts/run_demo.py` prints and writes to `results/`; none were edited after the fact.

Calibration study (mean field closed form versus Monte Carlo ground truth, 50000 trials per root):

| Node group | Mean absolute error |
|---|---|
| Convergence node (`publish_report`) | 0.0280 |
| Non convergence nodes (average) | 0.0011 |
| Ratio | 26.46x |

Root cause localization, 1000 simulated incidents:

| Method | Precision at 1 | Precision at 3 | MRR |
|---|---|---|---|
| Topological baseline (naive) | 0.882 | 0.969 | 0.925 |
| Bayesian mean field posterior | 0.867 | 0.965 | 0.918 |
| Bayesian plus evidence agent | 0.878 | 0.972 | 0.925 |

The pure Bayesian ranking underperforms the naive baseline overall. This is a genuine negative result and it is reported as such rather than hidden: with a uniform prior over candidates, the mean field likelihood alone discards the useful information that failures tend to propagate forward through the graph, information the topological baseline exploits by construction.

Stratified by agreement between the baseline and the Bayesian ranking (1000 incidents, minimum 2 failed tasks):

| Stratum | n | Baseline P at 1 | Bayesian P at 1 | Fused P at 1 |
|---|---|---|---|---|
| Easy (methods agree) | 940 | 0.900 | 0.900 | 0.902 |
| Hard (methods disagree) | 60 | 0.600 | 0.350 | 0.500 |

On the hard, disagreement subset, the one place where the choice of method actually matters, the pure Bayesian posterior is markedly worse than the naive baseline (0.350 versus 0.600). Evidence fusion recovers a substantial part of that gap (0.500) but does not fully close it. This is the project's most useful finding: a statistically principled model is not automatically better than a simple heuristic, the uniform prior is throwing away real information that the graph's topology already encodes, and the practical fix suggested by this result is not to add more agent evidence but to replace the uniform prior with one informed by topological position, an experiment left as documented future work rather than retrofitted into the headline numbers after the fact.

Fusion weight sweep (500 incidents per weight):

| Fusion weight | Precision at 1 | MRR |
|---|---|---|
| 0.0 (no agent) | 0.854 | 0.912 |
| 0.3 | 0.862 | 0.916 |
| 0.6 | 0.866 | 0.920 |
| 1.0 | 0.878 | 0.926 |
| 1.5 | 0.870 | 0.922 |

Accuracy improves as the fusion weight increases up to 1.0 and then degrades at 1.5, a plausible sign of the evidence agent's score starting to dominate and overriding a statistical ranking that was already correct; 1.0 is used as the default operating point in `evaluate.py`.

## Repository structure

```
src/dagrca/
  dag.py            synthetic pipeline DAG construction
  propagation.py    noisy OR generative model, Monte Carlo simulator, closed form marginals
  inference.py      Bayesian root cause ranking (product of marginals likelihood)
  baseline.py       naive topological earliness heuristic
  evidence.py       synthetic per task operational evidence generator
  llm_agent.py      tool augmented evidence agent, stub and Anthropic backed, log linear fusion
  evaluate.py       calibration study, localization evaluation, stratified evaluation
scripts/run_demo.py end to end reproducible run, writes results/
tests/              25 automated tests, including a brute force exact inference reference
results/            measured output of the last run_demo.py execution
```

## Installation and usage

Requires Python 3.10 or later. No paid API access is required to run the full test suite or the default demo.

```bash
git clone https://github.com/julianodutraa/causal-rca.git
cd causal-rca
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/run_demo.py
```

To use a real Claude model for the evidence agent instead of the deterministic stub, install the optional extra and set an API key:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=your_key_here
```

Then pass `llm_client=AnthropicLLMClient()` to `run_localization_evaluation` or `run_stratified_evaluation`.

## Limitations and honest scope

This project uses a synthetic DAG and a synthetic generative model of failure propagation; the edge transmission probabilities and leak probabilities are hand set to plausible ranges rather than fit to real incident history, since no proprietary pipeline data is used. In a real deployment, these parameters would be estimated from historical incident logs (a maximum likelihood or method of moments estimator over the same noisy OR model is a direct extension of the code here, not a redesign).

The uniform prior over candidate roots in the Bayesian ranking is a deliberate simplification to isolate the likelihood model's own discriminative power, and the stratified results above show plainly that this simplification costs real accuracy on the hardest incidents. The evidence agent's synthetic evidence generator is calibrated by hand (a 75 percent chance the true root shows incriminating evidence, a 15 percent red herring rate for non root tasks) rather than fit to real log data. The `StubLLMClient` used for all reported numbers is a deterministic keyword matching policy, not a language model; it is a legitimate, reproducible agent policy in its own right, but the `AnthropicLLMClient` path, which would use a real model's qualitative reasoning over the evidence text, was not run for the numbers in this README, and its behavior on this evaluation set is genuinely unknown rather than assumed to be better.

## License

MIT, see `LICENSE`.
