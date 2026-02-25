"""
Loong Benchmark Evaluator for SPD-RAG System
=============================================
Implements the evaluation methodology from the Loong paper
(Wang et al., "Leave No Document Behind: Benchmarking Long-Context
LLMs with Extended Multi-Doc QA", EMNLP 2024).

Key components:
- GPT-5 as judge (Appendix A of the paper)
- Evaluates Accuracy/Hallucinations and Completeness
- Scores 1-100 per question
- Two aggregate metrics: Avg Score and Perfect Rate
- Oracle document setting: all gold documents provided to SPD-RAG

Loong task levels:
  1 = Spotlight Locating
  2 = Comparison
  3 = Clustering
  4 = Chain of Reasoning

Usage:
    python benchmark/loong/loong_evaluator.py --data benchmark/loong/data/loong_set1.jsonl
    python benchmark/loong/loong_evaluator.py --data benchmark/loong/data/loong_set1.jsonl --upload-docs --limit 5
    python benchmark/loong/loong_evaluator.py --level 1 --language en --limit 10
    python benchmark/loong/loong_evaluator.py --resume  # skip already-evaluated task IDs
    python benchmark/loong/loong_evaluator.py --summarize benchmark/loong/loong_results.jsonl
"""

import json
import asyncio
import time
import uuid
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.vector import VectorStorePipeline

# ── Paths ──────────────────────────────────────────────────────────────────────

DEFAULT_DATA = PROJECT_ROOT / "benchmark" / "loong" / "data" / "loong_set1.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "benchmark" / "loong" / "loong_results.jsonl"
DEFAULT_AGENTIC_RESULTS = PROJECT_ROOT / "benchmark" / "loong" / "agentic_rag_results.jsonl"

# ── Tokenizer ──────────────────────────────────────────────────────────────────

_ENCODER: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ── Task level labels ──────────────────────────────────────────────────────────

LEVEL_NAMES = {
    1: "Spotlight Locating",
    2: "Comparison",
    3: "Clustering",
    4: "Chain of Reasoning",
}

# ── Loong GPT-4-as-Judge Prompt (Paper Appendix A) ────────────────────────────

LOONG_JUDGE_PROMPT = """\
[Question]
{question}

[Gold Answer]
{answer}

[The Start of Assistant's Predicted Answer]
{predicted}
[The End of Assistant's Predicted Answer]

[System]
We would like to request your feedback on the performance of the AI assistant \
in response to the user question displayed above according to the gold answer. \
Please use the following listed aspects and their descriptions as evaluation \
criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically \
consistent with the gold answer; The numerical value and order need to be \
accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer \
should contain all the key points needed to answer the user's question; further \
elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that \
the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher \
score indicates better overall performance.
Please note that if the assistant's answer and the gold answer fully meet the \
above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding \
any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO \
100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE \
"Rating: [[100]]":
<start output>
Evaluation evidence: your evaluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_answer(answer: Any) -> str:
    """Format a gold answer into a string for the judge prompt.

    Loong answers can be str, list, or dict.
    """
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False)


def filter_data(
    data: List[Dict[str, Any]],
    level: Optional[int] = None,
    language: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter dataset by task level, language, offset, and max count."""
    if level is not None:
        data = [d for d in data if d["level"] == level]
    if language is not None:
        data = [d for d in data if d["language"] == language]
    if offset is not None:
        data = data[offset:]
    if limit is not None:
        data = data[:limit]
    return data


def load_evaluated_ids(results_path: str) -> set:
    """Return the set of task IDs already present in a results JSONL file.

    Returns an empty set if the file does not exist or cannot be read.
    """
    path = Path(results_path)
    if not path.exists():
        return set()
    ids: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "id" in entry:
                    ids.add(entry["id"])
            except json.JSONDecodeError:
                pass
    return ids


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════


async def upload_documents(data: List[Dict[str, Any]]):
    """Upload all oracle documents for the given tasks to the vector store.

    Each task has:
      - doc: list of filenames (e.g. ["2401.06209.md", ...])
      - docs: list of full-text contents (parallel to doc)
    """
    pipeline = VectorStorePipeline()
    total = sum(len(task["doc"]) for task in data)
    uploaded = 0

    for task in data:
        task_id = task["id"]
        question = task["question"]
        doc_names = task["doc"]
        doc_texts = task["docs"]

        for filename, text in zip(doc_names, doc_texts):
            try:
                await pipeline.run(
                    text, filename,
                    metadata={"id": task_id, "question": question},
                )
                uploaded += 1
                print(f"  [{uploaded}/{total}] Uploaded: {filename}")
            except Exception as e:
                print(f"  [WARN] Failed to upload {filename}: {e}")

    print(f"Upload complete: {uploaded}/{total} documents.")


# ═══════════════════════════════════════════════════════════════════════════════
# SPD-RAG INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


async def run_spd_rag(prompt: str, doc_filenames: List[str]) -> Dict[str, Any]:
    """Run the SPD-RAG multi-agent system on a Loong task.

    Args:
        prompt: The pre-built user prompt (instruction + question, no docs).
        doc_filenames: Original filenames from the 'doc' field, passed as
                       selected_documents so the graph fans out sub-agents.
    """
    from langchain_core.messages import HumanMessage
    from backend.core.graph import get_compiled_graph

    graph = get_compiled_graph()
    prompt_tokens = count_tokens(prompt)

    start = time.perf_counter()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "selected_documents": doc_filenames,
        },
        config={"configurable": {"thread_id": f"loong_{uuid.uuid4().hex[:12]}"}, "metadata": {"model": "gemini-2.5-pro"}},
    )
    latency = time.perf_counter() - start

    
    raw_output = ""
    if result.get("messages"):
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if isinstance(content, str):
                    raw_output = content
                elif isinstance(content, list):
                    # Tool-call messages have list content blocks
                    texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    raw_output = " ".join(texts)
                if raw_output:
                    break


    completion_tokens = count_tokens(raw_output)
    return {
        "raw_output": raw_output,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE LLM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


async def run_baseline_llm(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run the baseline LLM on a Loong task (all docs in context, no retrieval).

    Concatenates all oracle document texts followed by the task prompt and
    sends the whole thing to RESEARCH_LLM_REASONING in a single call.
    Returns the same dict shape as run_spd_rag for drop-in use in evaluate().
    """
    from backend.shared.constants import RESEARCH_LLM_REASONING

    docs_text = "".join(task["docs"])
    full_prompt = f"{docs_text}\n{task['prompt']}"
    prompt_tokens = count_tokens(full_prompt)

    start = time.perf_counter()
    response = await RESEARCH_LLM_REASONING.ainvoke(full_prompt)
    latency = time.perf_counter() - start

    content = response.content
    if isinstance(content, list):
        raw_output = " ".join(
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        raw_output = content or ""

    return {
        "raw_output": raw_output,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": count_tokens(raw_output),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC RAG BASELINE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


async def run_agentic_rag_loong(prompt: str, doc_filenames: List[str]) -> Dict[str, Any]:
    """Run the Agentic RAG baseline on a Loong task prompt."""
    from benchmark.agentic_rag import run_agentic_rag

    return await run_agentic_rag(query=prompt, selected_files=doc_filenames)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE (Loong Appendix A)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_judge(
    question: str,
    gold_answer_str: str,
    predicted: str,
    model: str = "gpt-5",
) -> Dict[str, Any]:
    """Send the predicted answer to the Loong LLM judge and parse the score."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model=model, temperature=0.0)
    prompt = LOONG_JUDGE_PROMPT.format(
        question=question,
        answer=gold_answer_str,
        predicted=predicted,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return parse_judge_response(response.content)


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the [[score]] from the judge output.

    The judge is instructed to output: Rating: [[score]]
    """
    result: Dict[str, Any] = {
        "score": 0,
        "explanation": "",
        "raw_judge_output": text,
    }

    score_match = re.search(r"\[\[(\d+)\]\]", text)
    if score_match:
        result["score"] = int(score_match.group(1))

    explanation_match = re.search(
        r"Evaluation evidence:\s*(.+?)(?:\nRating:|\Z)",
        text,
        re.DOTALL,
    )
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARIZE EXISTING RESULTS
# ═══════════════════════════════════════════════════════════════════════════════


def summarize_results(results_path: str):
    """Print aggregate metrics from a previously saved results JSONL file."""
    results = load_data(results_path)
    if not results:
        print("No results found.")
        return

    _print_summary(results)


def _print_summary(results: List[Dict[str, Any]]):
    """Print Avg Score, Perfect Rate, and per-level breakdown."""
    n = len(results)
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / n
    perfect_rate = sum(1 for s in scores if s == 100) / n

    print("\n" + "=" * 65)
    print("  Loong Evaluation Summary (SPD-RAG, Oracle Documents)")
    print("=" * 65)
    print(f"  Questions evaluated : {n}")
    print(f"  Avg Score (0-100)   : {avg_score:.2f}")
    print(f"  Perfect Rate        : {perfect_rate:.4f} ({perfect_rate*100:.1f}%)")

    # Per-level breakdown
    levels_seen = sorted(set(r.get("level", 0) for r in results))
    if len(levels_seen) > 1 or (levels_seen and levels_seen[0] != 0):
        print("\n  Per-task breakdown:")
        print("  " + "-" * 55)
        for lev in levels_seen:
            lev_results = [r for r in results if r.get("level") == lev]
            lev_scores = [r["score"] for r in lev_results]
            lev_n = len(lev_scores)
            lev_avg = sum(lev_scores) / lev_n
            lev_pr = sum(1 for s in lev_scores if s == 100) / lev_n
            name = LEVEL_NAMES.get(lev, f"Level {lev}")
            print(f"  {name:25s} : Avg={lev_avg:6.2f}  PR={lev_pr:.4f}  (n={lev_n})")

    # Per-language breakdown
    langs_seen = sorted(set(r.get("language", "?") for r in results))
    if len(langs_seen) > 1:
        print("\n  Per-language breakdown:")
        print("  " + "-" * 55)
        for lang in langs_seen:
            lang_results = [r for r in results if r.get("language") == lang]
            lang_scores = [r["score"] for r in lang_results]
            lang_n = len(lang_scores)
            lang_avg = sum(lang_scores) / lang_n
            lang_pr = sum(1 for s in lang_scores if s == 100) / lang_n
            print(f"  {lang:25s} : Avg={lang_avg:6.2f}  PR={lang_pr:.4f}  (n={lang_n})")

    print("\n  " + "-" * 55)
    print("  Reference (Loong paper, overall baselines):")
    print("  Gemini-1.5-pro  : Avg=55.37  PR=0.27")
    print("  GPT-4o          : Avg=53.47  PR=0.26")
    print("  Claude3.5-Sonnet: Avg=48.85  PR=0.23")
    avg_latency = sum(r.get("latency", 0) for r in results) / n
    print(f"\n  Avg Latency (s)  : {avg_latency:.1f}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate(
    data_path: str = str(DEFAULT_DATA),
    results_path: str = str(DEFAULT_RESULTS),
    upload_docs: bool = False,
    level: Optional[int] = None,
    language: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    judge_model: str = "gpt-5",
    resume: bool = False,
    baseline: bool = False,
    agentic: bool = False,
):
    """Run the full Loong evaluation pipeline.

    Three inference modes (mutually exclusive; ``agentic`` takes priority over
    ``baseline`` if both are set):

    * **SPD-RAG** (default) — multi-agent, per-document sub-agents, oracle
      document filtering.  Requires ``--upload-docs``.
    * **Baseline LLM** (``--baseline``) — all oracle docs concatenated into
      context, single LLM call.  No vectorstore needed.
    * **Agentic RAG** (``--agentic``) — single agent, iterative retrieval
      scoped to the oracle documents for the task.  Requires ``--upload-docs``.

    Steps for each task:
    1. Run the selected inference mode with the task prompt and oracle documents.
    2. Send the response to the Loong LLM judge (Appendix A).
    3. Parse the 0-100 score.
    4. Append the result to the JSONL results file.
    5. Print an aggregate summary at the end.
    """
    data = load_data(data_path)
    print(f"Loaded {len(data)} tasks from {data_path}")

    data = filter_data(data, level=level, language=language, offset=offset, limit=limit)
    print(f"After filtering: {len(data)} tasks", end="")
    filters = []
    if level is not None:
        filters.append(f"level={level} ({LEVEL_NAMES.get(level, '?')})")
    if language is not None:
        filters.append(f"lang={language}")
    if offset is not None:
        filters.append(f"offset={offset}")
    if limit is not None:
        filters.append(f"limit={limit}")
    if filters:
        print(f" [{', '.join(filters)}]")
    else:
        print()

    if resume:
        done_ids = load_evaluated_ids(results_path)
        if done_ids:
            before = len(data)
            data = [d for d in data if d["id"] not in done_ids]
            print(
                f"Resume mode: skipping {before - len(data)} already-evaluated tasks "
                f"({len(data)} remaining)."
            )
        else:
            print("Resume mode: no existing results found, starting fresh.")

    if not data:
        print("No tasks to evaluate.")
        return

    if agentic:
        mode_label = "Agentic RAG"
    elif baseline:
        mode_label = "Baseline LLM"
    else:
        mode_label = "SPD-RAG"
    print(f"Inference mode: {mode_label}")

    if upload_docs:
        if baseline:
            print("(Skipping vector store upload — not needed for baseline mode)")
        else:
            print("\n── Uploading oracle documents to vector store ──")
            await upload_documents(data)
            print()

    all_results: List[Dict[str, Any]] = []

    with open(results_path, "a", encoding="utf-8") as f:
        for i, task in enumerate(data):
            task_id = task["id"]
            question = task["question"]
            answer = task["answer"]
            level_num = task["level"]
            lang = task["language"]
            doc_type = task["type"]
            prompt = task["prompt"]
            doc_filenames = task["doc"]

            level_name = LEVEL_NAMES.get(level_num, f"Level {level_num}")
            print(
                f"\n── [{i + 1}/{len(data)}] {level_name} | {doc_type} | {lang} ──"
            )
            print(f"   Q: {question[:100]}...")
            print(f"   Docs: {len(doc_filenames)}")

            try:
                if agentic:
                    rag_result = await run_agentic_rag_loong(prompt, doc_filenames)
                elif baseline:
                    rag_result = await run_baseline_llm(task)
                else:
                    rag_result = await run_spd_rag(prompt, doc_filenames)
            except Exception as e:
                print(f"   [ERROR] {mode_label} failed: {e}")
                continue
            print(f"   Latency: {rag_result['latency']:.1f}s")

            gold_str = format_answer(answer)
            try:
                judge_result = await run_judge(
                    question, gold_str, rag_result["raw_output"],
                    model=judge_model,
                )
            except Exception as e:
                print(f"   [ERROR] Judge failed: {e}")
                continue

            score = judge_result["score"]
            is_perfect = score == 100
            print(f"   Score: {score}/100{' (PERFECT)' if is_perfect else ''}")

            result_entry = {
                "id": task_id,
                "mode": "agentic_rag" if agentic else ("baseline" if baseline else "spd_rag"),
                "question": question,
                "level": level_num,
                "level_name": level_name,
                "type": doc_type,
                "language": lang,
                "num_docs": len(doc_filenames),
                "gold_answer": answer,
                "predicted_answer": rag_result["raw_output"],
                "score": score,
                "explanation": judge_result["explanation"],
                "latency": rag_result["latency"],
                "prompt_tokens": rag_result["prompt_tokens"],
                "completion_tokens": rag_result["completion_tokens"],
                "iterations": rag_result.get("iterations"),
                "raw_judge_output": judge_result["raw_judge_output"],
            }
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f.flush()
            all_results.append(result_entry)

    if all_results:
        _print_summary(all_results)
        print(f"\n  Results saved to: {results_path}")
    else:
        print("\nNo results were collected.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loong Benchmark Evaluator — SPD-RAG, Agentic RAG, and Baseline LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # SPD-RAG (default)
  python benchmark/loong/loong_evaluator.py --upload-docs --limit 3

  # Agentic RAG baseline (single agent, global search)
  python benchmark/loong/loong_evaluator.py --agentic --upload-docs --limit 3

  # Baseline LLM (full context, no retrieval)
  python benchmark/loong/loong_evaluator.py --baseline --limit 3

  # Filter and summarise
  python benchmark/loong/loong_evaluator.py --level 1 --language en --limit 10
  python benchmark/loong/loong_evaluator.py --summarize benchmark/loong/loong_results.jsonl
""",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Path to JSONL data file (default: loong_set1.jsonl)",
    )
    parser.add_argument(
        "--results", default=None,
        help=(
            "Path to write/append results. Defaults to loong_results.jsonl "
            "(SPD-RAG), agentic_rag_results.jsonl (--agentic), or "
            "baseline_loong_results.jsonl (--baseline)."
        ),
    )
    parser.add_argument(
        "--upload-docs", action="store_true",
        help="Upload oracle documents to vector store before evaluating",
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4],
        help="Filter by task level (1=Spotlight, 2=Comparison, 3=Clustering, 4=Chain)",
    )
    parser.add_argument(
        "--language", type=str, choices=["en", "zh"],
        help="Filter by language",
    )
    parser.add_argument(
        "--offset", type=int,
        help="Skip the first N tasks (after level/language filtering)",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Max number of tasks to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help=(
            "Use baseline LLM inference: all oracle docs are concatenated into "
            "the context and sent to RESEARCH_LLM_REASONING in a single call "
            "(no vector store needed)."
        ),
    )
    parser.add_argument(
        "--agentic", action="store_true",
        help=(
            "Use Agentic RAG inference: a single agent iteratively searches the "
            "oracle documents for each task (same document set as SPD-RAG, but "
            "one agent instead of N). Requires --upload-docs on the first run."
        ),
    )
    parser.add_argument(
        "--judge-model", default="gpt-5",
        help="Model to use as judge (default: gpt-5)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Skip tasks whose IDs are already present in the --results file. "
            "Useful for continuing an interrupted run."
        ),
    )
    parser.add_argument(
        "--summarize", type=str, metavar="FILE",
        help="Just print summary metrics from an existing results file, then exit",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize_results(args.summarize)
    else:
        # Resolve the default results path based on the chosen inference mode
        if args.results is not None:
            results_path = args.results
        elif args.agentic:
            results_path = str(DEFAULT_AGENTIC_RESULTS)
        elif args.baseline:
            results_path = str(PROJECT_ROOT / "baseline_loong_results.jsonl")
        else:
            results_path = str(DEFAULT_RESULTS)

        asyncio.run(
            evaluate(
                data_path=args.data,
                results_path=results_path,
                upload_docs=args.upload_docs,
                level=args.level,
                language=args.language,
                offset=args.offset,
                limit=args.limit,
                judge_model=args.judge_model,
                resume=args.resume,
                baseline=args.baseline,
                agentic=args.agentic,
            )
        )
