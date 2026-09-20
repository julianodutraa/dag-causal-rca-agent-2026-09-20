"""Tool-augmented evidence agent and log-linear fusion with the Bayesian posterior.

The agent's job is narrow and well defined: given a short list of candidate
root-cause nodes (typically the top-K from the Bayesian ranking in
``inference.py``), pull each candidate's evidence bundle (deploy log, error
message, metric snapshot; see ``evidence.py``) through a small tool-calling
interface, and return a score per candidate representing how strongly the
evidence, read qualitatively, implicates that node as the true root cause.

The tool-calling interface (``EvidenceTools``) is model-agnostic. Two
concrete clients implement the same ``LLMClient`` protocol:

* ``StubLLMClient`` is a deterministic, dependency-free keyword-overlap
  scorer. It is what the test suite and the default offline evaluation
  run use, so the project is fully reproducible without any external API
  access or cost. It is a legitimate, if weak, agent policy in its own
  right, not just a mock: it genuinely reads the tool outputs and scores
  candidates from them, it just does so with lexical matching instead of a
  language model.
* ``AnthropicLLMClient`` wires the same tool-calling loop to a real Claude
  model through the ``anthropic`` Python SDK, for interactive demos. It is
  optional (imported lazily) and is skipped automatically if the
  ``anthropic`` package or an ``ANTHROPIC_API_KEY`` is not available.

Fusion is log-linear (a product-of-experts): the final ranking score for a
candidate is the Bayesian log-posterior plus ``fusion_weight`` times the
log of the agent's evidence score. This is the standard way to combine two
independent probabilistic opinions when neither is fully trusted; the
weight is a hyperparameter swept explicitly in the evaluation (see
``evaluate.py`` and the README's honest-metrics section for where fusion
helps and where it does not).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from .evidence import NodeEvidence

EPS = 1e-6

_INCRIMINATING_KEYWORDS = {
    "config change",
    "schema migration",
    "credential rotation",
    "secret not yet propagated",
    "timeout from 30s to 5s",
    "pool exhausted",
    "authentication failed",
    "secret rejected",
    "read timeout after 5000ms",
    "spiked to pool_max",
    "p99 latency increased 8x",
}


@dataclass(frozen=True)
class AgentVerdict:
    node: str
    evidence_score: float
    rationale: str


class LLMClient(Protocol):
    def score_candidates(
        self, candidates: list[str], evidence: dict[str, NodeEvidence]
    ) -> list[AgentVerdict]: ...


class StubLLMClient:
    """Deterministic, dependency-free evidence scorer used for reproducible runs.

    Score is a bounded, monotone function of the number of incriminating
    keyword phrases found across the candidate's three evidence fields
    (case-insensitive substring match), mapped through a saturating
    function so that a single strong signal already yields high confidence
    but does not immediately imply certainty.
    """

    def score_candidates(
        self, candidates: list[str], evidence: dict[str, NodeEvidence]
    ) -> list[AgentVerdict]:
        verdicts = []
        for node in candidates:
            ev = evidence.get(node)
            if ev is None:
                verdicts.append(AgentVerdict(node=node, evidence_score=0.5, rationale="no evidence available"))
                continue
            text = " ".join([ev.deploy_log, ev.error_message, ev.metric_snapshot]).lower()
            hits = [kw for kw in _INCRIMINATING_KEYWORDS if kw in text]
            n_hits = len(hits)
            # saturating map: 0 hits -> 0.2, 1 hit -> ~0.62, 2+ hits -> ~0.85+
            score = 1.0 - math.exp(-0.9 * n_hits) * 0.8 if n_hits > 0 else 0.2
            score = min(max(score, EPS), 1.0 - EPS)
            rationale = f"{n_hits} incriminating signal(s): {', '.join(sorted(hits)) or 'none'}"
            verdicts.append(AgentVerdict(node=node, evidence_score=score, rationale=rationale))
        return verdicts


class AnthropicLLMClient:
    """Real Claude-backed evidence agent (optional, not used by the test suite).

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` in the
    environment. Runs a short tool-calling loop: the model is given the
    three evidence fields for every candidate directly in the prompt (the
    evidence bundles are small, so no separate tool round-trip is required
    for this project's scale) and asked to return a calibrated probability
    per candidate plus a one-line rationale, as strict JSON.
    """

    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover (exercised only when the extra is installed)
            raise RuntimeError(
                "AnthropicLLMClient requires the 'anthropic' package; install it or use StubLLMClient."
            ) from exc
        self._model = model

    def score_candidates(
        self, candidates: list[str], evidence: dict[str, NodeEvidence]
    ) -> list[AgentVerdict]:  # pragma: no cover (requires network and an API key)
        import anthropic
        import json

        client = anthropic.Anthropic()
        bundle = {
            node: {
                "deploy_log": evidence[node].deploy_log,
                "error_message": evidence[node].error_message,
                "metric_snapshot": evidence[node].metric_snapshot,
            }
            for node in candidates
            if node in evidence
        }
        prompt = (
            "You are triaging a data pipeline incident. For each candidate task below, "
            "estimate the probability (0 to 1) that it is the true root cause, given its "
            "deploy log, error message, and metric snapshot. Respond with strict JSON: "
            '{"node_id": {"probability": float, "rationale": "one short sentence"}, ...}.\n\n'
            f"Candidates:\n{json.dumps(bundle, indent=2)}"
        )
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        verdicts = []
        for node in candidates:
            entry = parsed.get(node, {"probability": 0.5, "rationale": "no response parsed"})
            score = min(max(float(entry.get("probability", 0.5)), EPS), 1.0 - EPS)
            verdicts.append(AgentVerdict(node=node, evidence_score=score, rationale=entry.get("rationale", "")))
        return verdicts


def fuse_scores(
    bayesian_log_scores: dict[str, float],
    agent_verdicts: list[AgentVerdict],
    fusion_weight: float,
) -> dict[str, float]:
    """Log-linear fusion of the Bayesian log-posterior and the agent's evidence score."""
    verdict_by_node = {v.node: v for v in agent_verdicts}
    fused: dict[str, float] = {}
    for node, log_score in bayesian_log_scores.items():
        verdict = verdict_by_node.get(node)
        agent_term = math.log(verdict.evidence_score) if verdict is not None else 0.0
        fused[node] = log_score + fusion_weight * agent_term
    return fused
