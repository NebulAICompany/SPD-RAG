"""
Benchmark Evaluator for SPD-RAG (Sub-Agent Per Document RAG)
=======================================================

Academic-grade evaluation harness aligned with the metrics and reporting
standards used in:

  - Zhang et al. (2025) "Recursive Language Models", arXiv:2512.24601
  - Bertsch et al. (2025) "OOLONG: Evaluating Long Context Reasoning
    and Aggregation Capabilities"

Scoring Metrics
---------------
  * Numerical answers  → OOLONG proximity score:  score(ŷ) = 0.75^|y − ŷ|
  * Text answers       → Normalized exact-match   (case/article/ws-insensitive)
  * List answers       → Set-level F1             (order-insensitive)

Statistical Reporting  (follows ICML 2025 position paper on CLT limitations)
--------------------
  * Multiple independent trials per question  (default n_trials = 3)
  * Bootstrap 95% confidence intervals        (not CLT — unsuitable for n < 300)
  * Per-question mean ± std
  * Aggregate reported as  score ± std  |  cost ± std   (RLM paper Table 1 format)

Reproducibility  (OLMES standard, NAACL 2025)
----------------
  * Full experiment metadata recorded  (model, temperature, prompts, seed, …)
  * Raw model outputs persisted for every trial
  * Deterministic ordering; optional random seed

Usage
-----
    python -m benchmark.evaluator --dataset 150k --mode direct_llm
    python -m benchmark.evaluator --dataset 1m   --mode rrm_graph --max-questions 10
    python -m benchmark.evaluator --dataset 150k  --mode direct_llm --n-trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import platform
import random
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import tiktoken
from backend.shared.logger import get_logger

logger = get_logger("BENCHMARK_EVALUATOR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"

DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "150k": {
        "context_file": "150k_context.txt",
        "qa_file": "150k_qa_pairs.json",
        "description": "Single-episode D&D transcript (~150K chars)",
    },
    "1m": {
        "context_file": "1m_context.txt",
        "qa_file": "1m_qa_pairs.json",
        "description": "Multi-episode D&D transcript (~1M chars)",
    },
}

# Tiktoken encoder for accurate token counts (cl100k_base covers GPT-4 family)
_ENCODER: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Return the exact token count using the cl100k_base encoding."""
    return len(_get_encoder().encode(text))


# =====================================================================
# 1.  DATA CLASSES
# =====================================================================
@dataclass
class TrialResult:
    """Result of a single trial (one model invocation for one question)."""

    trial_id: int
    raw_model_output: str
    extracted_answer: str
    score: float
    score_type: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class QuestionResult:
    """Aggregated result across all trials for a single question."""

    question_id: int
    question: str
    expected_answer: str
    expected_answer_type: str  # "numerical", "list", "text"
    trials: List[TrialResult] = field(default_factory=list)

    # -- Aggregates over trials ------------------------------------------

    @property
    def mean_score(self) -> float:
        return statistics.mean(t.score for t in self.trials) if self.trials else 0.0

    @property
    def std_score(self) -> float:
        if len(self.trials) < 2:
            return 0.0
        return statistics.stdev(t.score for t in self.trials)

    @property
    def mean_latency(self) -> float:
        return (
            statistics.mean(t.latency_seconds for t in self.trials)
            if self.trials
            else 0.0
        )

    @property
    def mean_cost(self) -> float:
        return statistics.mean(t.cost_usd for t in self.trials) if self.trials else 0.0

    @property
    def best_score(self) -> float:
        return max(t.score for t in self.trials) if self.trials else 0.0


@dataclass
class ExperimentMetadata:
    """
    Full experiment metadata for reproducibility (OLMES-aligned).
    Recorded in every results file so that experiments can be replicated.
    """

    experiment_id: str
    timestamp_utc: str
    dataset_key: str
    eval_mode: str
    model_name: str
    temperature: float
    n_trials: int
    max_questions: Optional[int]
    random_seed: Optional[int]
    system_prompt_hash: str
    context_length_chars: int
    context_length_tokens: int
    num_questions_total: int
    num_questions_evaluated: int
    python_version: str
    platform: str
    evaluator_version: str = "1.0.0"


@dataclass
class BenchmarkReport:
    """Top-level benchmark report matching RLM paper Table 1 format."""

    metadata: ExperimentMetadata
    questions: List[QuestionResult] = field(default_factory=list)
    total_wall_time_seconds: float = 0.0

    # -- Aggregate metrics -----------------------------------------------

    @property
    def all_scores(self) -> List[float]:
        return [q.mean_score for q in self.questions]

    @property
    def overall_score(self) -> float:
        """Mean score across questions (each question's mean over trials)."""
        scores = self.all_scores
        return statistics.mean(scores) if scores else 0.0

    @property
    def overall_score_std(self) -> float:
        scores = self.all_scores
        return statistics.stdev(scores) if len(scores) >= 2 else 0.0

    @property
    def overall_cost_mean(self) -> float:
        costs = [q.mean_cost for q in self.questions]
        return statistics.mean(costs) if costs else 0.0

    @property
    def overall_cost_std(self) -> float:
        costs = [q.mean_cost for q in self.questions]
        return statistics.stdev(costs) if len(costs) >= 2 else 0.0

    @property
    def overall_latency_mean(self) -> float:
        lats = [q.mean_latency for q in self.questions]
        return statistics.mean(lats) if lats else 0.0

    @property
    def scores_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Breakdown of scores by answer type."""
        buckets: Dict[str, List[float]] = {}
        for q in self.questions:
            buckets.setdefault(q.expected_answer_type, []).append(q.mean_score)
        out = {}
        for atype, scores in buckets.items():
            out[atype] = {
                "n": len(scores),
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) >= 2 else 0.0,
                "min": min(scores),
                "max": max(scores),
            }
        return out

    def bootstrap_ci(
        self, alpha: float = 0.05, n_bootstrap: int = 10_000
    ) -> Tuple[float, float]:
        """
        Non-parametric bootstrap 95% confidence interval for the mean score.

        This is the recommended approach for LLM benchmarks with < 300 data
        points, per "Don't use the CLT in LLM evals" (ICML 2025).
        """
        scores = self.all_scores
        n = len(scores)
        if n == 0:
            return (0.0, 0.0)

        rng = random.Random(42)  # deterministic bootstrap
        boot_means: List[float] = []
        for _ in range(n_bootstrap):
            sample = [rng.choice(scores) for _ in range(n)]
            boot_means.append(statistics.mean(sample))

        boot_means.sort()
        lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
        hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
        return (boot_means[lo_idx], boot_means[hi_idx])

    def wilson_score_interval(
        self, threshold: float = 0.5, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Wilson score interval for the proportion of questions scoring above
        *threshold*.  More robust than Wald (CLT) intervals for small n.
        """
        import math as _math

        n = len(self.questions)
        if n == 0:
            return (0.0, 0.0)
        k = sum(1 for q in self.questions if q.mean_score >= threshold)
        p_hat = k / n
        z = 1.96 if alpha == 0.05 else 1.645  # z_{1-alpha/2}
        denom = 1 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denom
        margin = (z / denom) * _math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-serialisable representation."""
        ci_lo, ci_hi = self.bootstrap_ci()
        ws_lo, ws_hi = self.wilson_score_interval()
        return {
            "metadata": asdict(self.metadata),
            "aggregate": {
                "overall_score": round(self.overall_score, 4),
                "overall_score_std": round(self.overall_score_std, 4),
                "bootstrap_95_ci": [round(ci_lo, 4), round(ci_hi, 4)],
                "wilson_pass_rate_95_ci": [round(ws_lo, 4), round(ws_hi, 4)],
                "mean_cost_usd": round(self.overall_cost_mean, 4),
                "std_cost_usd": round(self.overall_cost_std, 4),
                "mean_latency_seconds": round(self.overall_latency_mean, 2),
                "total_wall_time_seconds": round(self.total_wall_time_seconds, 2),
                "scores_by_type": {
                    k: {kk: round(vv, 4) for kk, vv in v.items()}
                    for k, v in self.scores_by_type.items()
                },
            },
            "questions": [
                {
                    "question_id": q.question_id,
                    "question": q.question,
                    "expected_answer": q.expected_answer,
                    "expected_answer_type": q.expected_answer_type,
                    "mean_score": round(q.mean_score, 4),
                    "std_score": round(q.std_score, 4),
                    "mean_latency": round(q.mean_latency, 2),
                    "mean_cost": round(q.mean_cost, 4),
                    "n_trials": len(q.trials),
                    "trials": [
                        {
                            "trial_id": t.trial_id,
                            "extracted_answer": t.extracted_answer,
                            "score": round(t.score, 4),
                            "score_type": t.score_type,
                            "latency_seconds": round(t.latency_seconds, 2),
                            "prompt_tokens": t.prompt_tokens,
                            "completion_tokens": t.completion_tokens,
                            "cost_usd": round(t.cost_usd, 6),
                        }
                        for t in q.trials
                    ],
                }
                for q in self.questions
            ],
        }


# =====================================================================
# 2.  SCORING — aligned with OOLONG & RLM paper §2.1
# =====================================================================
class Scorer:
    """
    Scoring functions implementing the exact metrics described in the RLM
    paper and the OOLONG benchmark.

    Numerical:  score(ŷ) = 0.75^|y − ŷ|
    Text:       normalised exact-match (case, articles, punctuation, whitespace)
    List:       set-level F1 (order-insensitive, normalised per item)
    """

    # -- Answer normalisation (academic standard) -------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalise an answer string following standard academic NLP practice:
          1. Lower-case
          2. Strip leading/trailing whitespace
          3. Remove articles (a, an, the)
          4. Remove punctuation (except hyphens inside words)
          5. Collapse multiple spaces
        """
        s = text.lower().strip()
        # Remove articles
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # Remove punctuation except hyphens within words
        s = re.sub(r"(?<!\w)[^\w\s]|[^\w\s](?!\w)", " ", s)
        # Collapse whitespace
        s = " ".join(s.split())
        return s

    @staticmethod
    def _normalise_list_item(text: str) -> str:
        """Normalise a single item within a comma-separated list."""
        return Scorer._normalise(text)

    # -- Metric implementations -------------------------------------------

    @staticmethod
    def oolong_numerical(predicted: str, expected: str) -> float:
        """
        OOLONG proximity score for numerical answers.

        score(ŷ) = 0.75^|y − ŷ|

        Reference: Bertsch et al. (2025), §3.
        """
        try:
            y_hat = float(predicted.strip())
            y = float(expected.strip())
            return 0.75 ** abs(y - y_hat)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def normalised_exact_match(predicted: str, expected: str) -> float:
        """
        Normalised exact-match for categorical / text answers.

        Both sides are normalised (lowercase, remove articles, strip
        punctuation) before comparison.
        """
        return (
            1.0 if Scorer._normalise(predicted) == Scorer._normalise(expected) else 0.0
        )

    @staticmethod
    def set_f1(predicted: str, expected: str) -> float:
        """
        Set-level F1 for comma-separated list answers.

        Each item is normalised independently.  Order does not matter
        (set semantics).  Duplicates within prediction or reference are
        collapsed.
        """
        pred_items = {
            Scorer._normalise_list_item(item)
            for item in predicted.split(",")
            if item.strip()
        }
        exp_items = {
            Scorer._normalise_list_item(item)
            for item in expected.split(",")
            if item.strip()
        }

        if not pred_items and not exp_items:
            return 1.0
        if not pred_items or not exp_items:
            return 0.0

        tp = len(pred_items & exp_items)
        precision = tp / len(pred_items)
        recall = tp / len(exp_items)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # -- Dispatch ---------------------------------------------------------

    @classmethod
    def classify(cls, expected: str) -> str:
        """
        Classify the expected answer into a scoring category.

        Returns one of: "numerical", "list", "text".
        """
        stripped = expected.strip()

        # Numerical
        try:
            float(stripped)
            return "numerical"
        except ValueError:
            pass

        # Percentage stripped of '%'
        if stripped.endswith("%"):
            try:
                float(stripped[:-1])
                return "numerical"
            except ValueError:
                pass

        # Comma-separated list
        if "," in stripped:
            return "list"

        return "text"

    @classmethod
    def score(cls, predicted: str, expected: str) -> Tuple[float, str]:
        """
        Score a single predicted answer against the gold answer.

        Returns (score, metric_name).
        """
        answer_type = cls.classify(expected)
        if answer_type == "numerical":
            return cls.oolong_numerical(predicted, expected), "oolong_numerical"
        elif answer_type == "list":
            return cls.set_f1(predicted, expected), "set_f1"
        else:
            return (
                cls.normalised_exact_match(predicted, expected),
                "normalised_exact_match",
            )


# =====================================================================
# 3.  ANSWER EXTRACTION
# =====================================================================
def extract_answer(text: str) -> str:
    r"""
    Extract the final answer from model output.

    Strategy (ordered by priority):
      1. Last ``\boxed{...}`` occurrence (handles nested braces)
      2. ``boxed{...}`` without backslash
      3. "Answer: ..." / "Final Answer: ..." prefix on the last line
      4. Last non-empty line (ultimate fallback)

    This multi-strategy pipeline follows recommendations from recent
    work on robust LLM answer extraction (ACL 2025).
    """
    # --- Strategy 1: \boxed{...} with nested brace support ---------------
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if matches:
        last_match = matches[-1]
        start = last_match.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            return text[start : pos - 1].strip()

    # --- Strategy 2: boxed{...} without backslash ------------------------
    simple_matches = re.findall(r"boxed\{([^}]+)\}", text)
    if simple_matches:
        return simple_matches[-1].strip()

    # --- Strategy 3: "Answer:" prefix ------------------------------------
    answer_prefixes = [
        r"(?:final\s+)?answer\s*:\s*",
        r"the\s+answer\s+is\s*:?\s*",
    ]
    lines = text.strip().split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        for prefix_pat in answer_prefixes:
            m = re.match(prefix_pat, stripped, re.IGNORECASE)
            if m:
                return stripped[m.end() :].strip()

    # --- Strategy 4: last non-empty line ---------------------------------
    for line in reversed(lines):
        if line.strip():
            return line.strip()

    return text.strip()


# =====================================================================
# 4.  LLM-AS-JUDGE  (optional, for ambiguous answers)
# =====================================================================
_JUDGE_PROMPT = """You are an impartial judge evaluating whether a predicted answer matches the expected gold answer for a question.

Question: {question}
Expected (gold) answer: {expected}
Predicted answer: {predicted}

Evaluate STRICTLY:
- If the predicted answer is semantically equivalent to the expected answer, respond with: CORRECT
- If the predicted answer is partially correct (contains some but not all required information), respond with: PARTIAL
- If the predicted answer is wrong or unrelated, respond with: INCORRECT

Respond with exactly one word: CORRECT, PARTIAL, or INCORRECT."""


async def llm_judge(
    question: str,
    expected: str,
    predicted: str,
    model_name: str = "gpt-5",
) -> float:
    """
    Use an LLM as a judge for cases where automated metrics may be
    insufficient (e.g. paraphrased answers, equivalent formulations).

    Returns 1.0 (CORRECT), 0.5 (PARTIAL), or 0.0 (INCORRECT).
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    judge_llm = ChatOpenAI(model=model_name, temperature=0.0)
    prompt = _JUDGE_PROMPT.format(
        question=question, expected=expected, predicted=predicted
    )
    response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    verdict = response.content.strip().upper()
    if "CORRECT" in verdict and "INCORRECT" not in verdict:
        return 1.0
    elif "PARTIAL" in verdict:
        return 0.5
    return 0.0


# =====================================================================
# 5.  COST ESTIMATION
# =====================================================================
# Prices per 1M tokens (USD) — update as pricing changes.
# Source: OpenAI pricing page, accessed 2026-02.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5": {"input": 2.00, "output": 8.00},
    "gpt-5-mini": {"input": 0.40, "output": 1.60},
}


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the API cost for a single call in USD."""
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return 0.0  # Unknown model — cannot estimate
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# =====================================================================
# 6.  DATA LOADING
# =====================================================================
def load_benchmark_data(
    dataset_key: str,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    """Load context and QA pairs; compute exact token counts."""
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_key]
    context_path = DATA_DIR / config["context_file"]
    qa_path = DATA_DIR / config["qa_file"]

    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA pairs file not found: {qa_path}")

    context_text = context_path.read_text(encoding="utf-8")
    qa_pairs: List[Dict[str, str]] = json.loads(qa_path.read_text(encoding="utf-8"))

    token_count = count_tokens(context_text)

    metadata = {
        "description": config["description"],
        "context_length_chars": len(context_text),
        "context_length_tokens": token_count,
        "num_questions": len(qa_pairs),
        "context_file_sha256": hashlib.sha256(context_text.encode()).hexdigest(),
    }

    logger.info(
        "Loaded dataset '%s': %s chars, %s tokens, %d questions",
        dataset_key,
        f"{metadata['context_length_chars']:,}",
        f"{metadata['context_length_tokens']:,}",
        metadata["num_questions"],
    )

    return context_text, qa_pairs, metadata


# =====================================================================
# 7.  EVALUATION BACKENDS
# =====================================================================
SYSTEM_PROMPT_DIRECT = (
    "You are a precise data analyst. Answer the question based strictly "
    "on the provided context. Return your final answer inside \\boxed{}. "
    "Do not guess or approximate. Show your reasoning step by step."
)


async def evaluate_direct_llm(
    context: str,
    question: str,
    model_name: str = "gpt-5",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Baseline: send context + question directly to an LLM.

    Returns dict with keys:
      raw_output, latency, prompt_tokens, completion_tokens
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    prompt = f"{context}\n\nQuestion: {question}"
    prompt_tokens = count_tokens(SYSTEM_PROMPT_DIRECT + prompt)

    start = time.perf_counter()
    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT_DIRECT),
            HumanMessage(content=prompt),
        ]
    )
    latency = time.perf_counter() - start

    raw_output = response.content
    completion_tokens = count_tokens(raw_output)

    return {
        "raw_output": raw_output,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def evaluate_rrm_graph(
    context: str,
    question: str,
) -> Dict[str, Any]:
    """
    System-under-test: route through the RRM LangGraph pipeline.

    Returns dict with the same keys as evaluate_direct_llm.
    """
    from langchain_core.messages import HumanMessage
    from backend.core.graph import get_compiled_graph

    graph = get_compiled_graph()

    user_message = (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Provide your final answer inside \\boxed{{}}."
    )
    prompt_tokens = count_tokens(user_message)

    start = time.perf_counter()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "selected_documents": [],
        },
        config={"configurable": {"thread_id": f"bench_{uuid.uuid4().hex[:12]}"}},
    )
    latency = time.perf_counter() - start

    raw_output = ""
    if result.get("messages"):
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                raw_output = msg.content
                break

    completion_tokens = count_tokens(raw_output)
    return {
        "raw_output": raw_output,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# =====================================================================
# 8.  MAIN EVALUATION LOOP
# =====================================================================
async def run_benchmark(
    dataset_key: str,
    eval_mode: Literal["direct_llm", "rrm_graph"] = "direct_llm",
    model_name: str = "gpt-5",
    temperature: float = 0.0,
    n_trials: int = 3,
    max_questions: Optional[int] = None,
    random_seed: Optional[int] = None,
    use_llm_judge: bool = False,
    save_results: bool = True,
) -> BenchmarkReport:
    """
    Run the full benchmark evaluation with academic rigour.

    Args:
        dataset_key:   "150k" or "1m".
        eval_mode:     "direct_llm" (baseline) or "rrm_graph" (system under test).
        model_name:    Model for direct_llm mode.
        temperature:   Sampling temperature.
        n_trials:      Number of independent trials per question.
        max_questions: Cap the number of questions (for debugging).
        random_seed:   Optional seed for reproducibility.
        use_llm_judge: Additionally run LLM-as-judge on text/list answers.
        save_results:  Persist results and raw outputs to disk.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # --- Load data -------------------------------------------------------
    context, qa_pairs, data_meta = load_benchmark_data(dataset_key)

    questions_to_eval = qa_pairs
    if max_questions is not None:
        questions_to_eval = qa_pairs[:max_questions]

    # --- Experiment metadata ---------------------------------------------
    experiment_id = uuid.uuid4().hex[:16]
    meta = ExperimentMetadata(
        experiment_id=experiment_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dataset_key=dataset_key,
        eval_mode=eval_mode,
        model_name=model_name,
        temperature=temperature,
        n_trials=n_trials,
        max_questions=max_questions,
        random_seed=random_seed,
        system_prompt_hash=hashlib.sha256(SYSTEM_PROMPT_DIRECT.encode()).hexdigest()[
            :16
        ],
        context_length_chars=data_meta["context_length_chars"],
        context_length_tokens=data_meta["context_length_tokens"],
        num_questions_total=data_meta["num_questions"],
        num_questions_evaluated=len(questions_to_eval),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )

    report = BenchmarkReport(metadata=meta)

    # --- Raw output directory (for auditability) -------------------------
    raw_out_dir: Optional[Path] = None
    if save_results:
        raw_out_dir = RAW_OUTPUTS_DIR / experiment_id
        raw_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Run evaluation --------------------------------------------------
    logger.info("=" * 72)
    logger.info(
        "EXPERIMENT %s | dataset=%s | mode=%s | model=%s | "
        "n_trials=%d | questions=%d",
        experiment_id,
        dataset_key,
        eval_mode,
        model_name,
        n_trials,
        len(questions_to_eval),
    )
    logger.info("=" * 72)

    wall_start = time.perf_counter()

    for q_idx, qa in enumerate(questions_to_eval):
        question = qa["question"]
        expected = qa["answer"]
        answer_type = Scorer.classify(expected)

        q_result = QuestionResult(
            question_id=q_idx,
            question=question,
            expected_answer=expected,
            expected_answer_type=answer_type,
        )

        logger.info(
            "[Q %d/%d] %s  (type=%s)",
            q_idx + 1,
            len(questions_to_eval),
            question[:80] + ("..." if len(question) > 80 else ""),
            answer_type,
        )

        for trial in range(n_trials):
            try:
                if eval_mode == "direct_llm":
                    result = await evaluate_direct_llm(
                        context, question, model_name, temperature
                    )
                elif eval_mode == "rrm_graph":
                    result = await evaluate_rrm_graph(context, question)
                else:
                    raise ValueError(f"Unknown eval_mode: {eval_mode}")

                raw_output = result["raw_output"]
                extracted = extract_answer(raw_output)
                score_val, score_type = Scorer.score(extracted, expected)

                # Optionally refine with LLM judge for non-numerical answers
                if use_llm_judge and answer_type != "numerical":
                    judge_score = await llm_judge(question, expected, extracted)
                    # Use the maximum of automated and judge scores
                    # (judge can rescue correct paraphrased answers)
                    score_val = max(score_val, judge_score)
                    if judge_score > 0:
                        score_type += "+judge"

                cost = estimate_cost(
                    model_name,
                    result["prompt_tokens"],
                    result["completion_tokens"],
                )

                trial_result = TrialResult(
                    trial_id=trial,
                    raw_model_output=raw_output,
                    extracted_answer=extracted,
                    score=score_val,
                    score_type=score_type,
                    latency_seconds=result["latency"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    cost_usd=cost,
                )
                q_result.trials.append(trial_result)

                # Save raw output for auditability
                if raw_out_dir is not None:
                    raw_file = raw_out_dir / f"q{q_idx:03d}_trial{trial}.txt"
                    raw_file.write_text(raw_output, encoding="utf-8")

                status = (
                    "PASS"
                    if score_val >= 0.99
                    else ("PARTIAL" if score_val > 0 else "FAIL")
                )
                logger.info(
                    "  trial %d/%d  [%s]  score=%.4f (%s)  "
                    "expected='%s'  predicted='%s'  latency=%.1fs  cost=$%.4f",
                    trial + 1,
                    n_trials,
                    status,
                    score_val,
                    score_type,
                    expected[:40],
                    extracted[:40],
                    result["latency"],
                    cost,
                )

            except Exception as exc:
                logger.error("  trial %d/%d  [ERROR]  %s", trial + 1, n_trials, exc)
                q_result.trials.append(
                    TrialResult(
                        trial_id=trial,
                        raw_model_output=str(exc),
                        extracted_answer="ERROR",
                        score=0.0,
                        score_type="error",
                        latency_seconds=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cost_usd=0.0,
                    )
                )

        # Log question-level aggregate
        logger.info(
            "  => Q%d mean_score=%.4f ± %.4f  (best=%.4f)",
            q_idx + 1,
            q_result.mean_score,
            q_result.std_score,
            q_result.best_score,
        )
        report.questions.append(q_result)

    report.total_wall_time_seconds = time.perf_counter() - wall_start

    # --- Summary ---------------------------------------------------------
    _print_report(report)

    if save_results:
        _save_report(report)

    return report


# =====================================================================
# 9.  REPORTING  (RLM paper Table 1 format)
# =====================================================================
def _fmt(val: float, places: int = 2) -> str:
    return f"{val:.{places}f}"


def _print_report(report: BenchmarkReport) -> None:
    """Pretty-print results in a format suitable for inclusion in a paper."""
    m = report.metadata
    ci_lo, ci_hi = report.bootstrap_ci()
    ws_lo, ws_hi = report.wilson_score_interval()

    print("\n" + "=" * 74)
    print("  BENCHMARK RESULTS  (RLM Paper Table 1 Format)")
    print("=" * 74)
    print(f"  Experiment ID    : {m.experiment_id}")
    print(f"  Timestamp (UTC)  : {m.timestamp_utc}")
    print(f"  Dataset          : {m.dataset_key}  ({m.context_length_tokens:,} tokens)")
    print(f"  Eval Mode        : {m.eval_mode}")
    print(f"  Model            : {m.model_name}  (T={m.temperature})")
    print(f"  Trials/question  : {m.n_trials}")
    print(f"  Questions        : {m.num_questions_evaluated} / {m.num_questions_total}")
    print(f"  Random Seed      : {m.random_seed}")
    print("-" * 74)

    # Table 1 format:  Score ± Std  |  Cost ± Std
    print(
        f"  Overall Score    : {_fmt(report.overall_score, 4)} "
        f"± {_fmt(report.overall_score_std, 4)}"
    )
    print(f"  Bootstrap 95% CI : [{_fmt(ci_lo, 4)}, {_fmt(ci_hi, 4)}]")
    print(
        f"  Wilson Pass Rate : [{_fmt(ws_lo, 4)}, {_fmt(ws_hi, 4)}]  (threshold ≥ 0.5)"
    )
    print(
        f"  Avg Cost/Query   : ${_fmt(report.overall_cost_mean, 4)} "
        f"± ${_fmt(report.overall_cost_std, 4)}"
    )
    print(f"  Avg Latency      : {_fmt(report.overall_latency_mean)}s")
    print(f"  Wall Clock       : {_fmt(report.total_wall_time_seconds)}s")

    # Breakdown by answer type
    print("-" * 74)
    print("  Scores by Answer Type:")
    print(f"  {'Type':<24} {'n':>4}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    for atype, info in report.scores_by_type.items():
        print(
            f"  {atype:<24} {info['n']:>4}  "
            f"{_fmt(info['mean'], 4):>8}  "
            f"{_fmt(info['std'], 4):>8}  "
            f"{_fmt(info['min'], 4):>8}  "
            f"{_fmt(info['max'], 4):>8}"
        )

    # Failure analysis
    failures = [q for q in report.questions if q.mean_score < 0.5]
    if failures:
        print("-" * 74)
        print(
            f"  Questions with mean score < 0.5  ({len(failures)} / {len(report.questions)}):"
        )
        for q in failures[:10]:
            print(f"    Q{q.question_id + 1}: {q.question[:65]}...")
            print(f"      Expected : {q.expected_answer[:55]}")
            best_pred = (
                max(q.trials, key=lambda t: t.score).extracted_answer
                if q.trials
                else "N/A"
            )
            print(f"      Best pred: {best_pred[:55]}")
            print(f"      Score    : {_fmt(q.mean_score, 4)} ± {_fmt(q.std_score, 4)}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    print("=" * 74 + "\n")


def _save_report(report: BenchmarkReport) -> None:
    """Save the full report (metadata + per-trial results) to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    m = report.metadata
    filename = f"{m.dataset_key}_{m.eval_mode}_{m.model_name}_{m.experiment_id}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Full report saved to %s", filepath)
    logger.info(
        "Raw model outputs saved to %s",
        RAW_OUTPUTS_DIR / m.experiment_id,
    )


# =====================================================================
# 10.  CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RRM Benchmark Evaluator — academic-grade evaluation harness "
            "with metrics aligned to the RLM paper (Zhang et al., 2025)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Baseline (direct LLM), 3 trials per question on the 150K dataset
  python -m benchmark.evaluator --dataset 150k --mode direct_llm --n-trials 3

  # RRM graph pipeline, first 10 questions of the 1M dataset
  python -m benchmark.evaluator --dataset 1m --mode rrm_graph --max-questions 10

  # Full run with LLM judge fallback and fixed seed
  python -m benchmark.evaluator --dataset 150k --mode direct_llm \\
      --n-trials 5 --use-llm-judge --seed 42
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        required=True,
        help="Which dataset to evaluate on.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct_llm", "rrm_graph"],
        default="direct_llm",
        help="Evaluation mode (default: direct_llm).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Model for direct_llm mode (default: gpt-5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Independent trials per question (default: 3).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap number of questions evaluated (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM-as-judge for text/list answers.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist results to disk.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            dataset_key=args.dataset,
            eval_mode=args.mode,
            model_name=args.model,
            temperature=args.temperature,
            n_trials=args.n_trials,
            max_questions=args.max_questions,
            random_seed=args.seed,
            use_llm_judge=args.use_llm_judge,
            save_results=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()
