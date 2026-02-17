"""
MoNaCo Benchmark Evaluator for SPD-RAG System
===============================================
Implements the evaluation methodology from the MoNaCo paper
(Wolfson et al., "MONACO: More Natural and Complex Questions
for Reasoning Across Dozens of Documents").

Key components:
- LLM-as-judge prompt (Figure 12 from the paper)
- Precision / Recall / F1 metric computation
- Oracle Retrieval setting: all gold evidence docs are provided
- Compares SPD-RAG multi-agent system against Oracle Retrieval baselines

Usage:
    python benchmark/new_monaco_evaluator.py --data benchmark/data/100q_withmds.jsonl
    python benchmark/new_monaco_evaluator.py --data benchmark/data/3q_withmds.jsonl --upload-docs
    python benchmark/new_monaco_evaluator.py --ids 1621 215 1053
    python benchmark/new_monaco_evaluator.py --summarize benchmark/evaluation_results.jsonl
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.vector import VectorStorePipeline

# ── Paths ──────────────────────────────────────────────────────────────────────

MD_PATH = PROJECT_ROOT / "benchmark" / "data" / "wiki_md_esradan"
DEFAULT_DATA = PROJECT_ROOT / "benchmark" / "data" / "100q_withmds.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "benchmark" / "evaluation_results.jsonl"

# ── Tokenizer ──────────────────────────────────────────────────────────────────

_ENCODER: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ── MoNaCo LLM-as-Judge Prompt (Figure 12) ────────────────────────────────────

MONACO_JUDGE_PROMPT = """\
Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact final answer to extract \
from the response.

[correct_answer]: {correct_answer}

final answer length: Provide the overall number of unique answers that appear \
in [response], not just the correct ones. Be sure to provide a number, not an \
estimate!

reasoning: Explain why the extracted_final_answer is correct or incorrect based \
on [correct_answer], focusing only on if there are meaningful differences between \
[correct_answer] and the extracted_final_answer. Do not comment on any background \
to the problem, do not attempt to solve the problem, do not argue for any answer \
different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] \
given above, or is within a small margin of error for numerical problems, a \
margin of 1 to 5.5 percentage points is acceptable. Answer 'no' otherwise, \
i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the \
extracted answer is incorrect.

precision: Answer '1' if extracted_final_answer matches the [correct_answer] \
given above. Answer '0' otherwise, i.e. if there is any inconsistency, \
ambiguity, non-equivalency, or if the extracted answer is incorrect. In the \
case where [correct_answer] is a number or percentage, then answer with the \
following formula to compute the normalized similarity score:
[1 - (abs([correct_answer] - extracted_final_answer) / \
max(abs([correct_answer]), abs(extracted_final_answer)))]

final precision: Extract the precision score from above, just the final score \
(number).

overlapping answers: List all of the answers in [response] that also appear in \
[correct_answer]. You can consider an answer from [response] to match with an \
answer in [correct_answer] if it is equivalent or is within a small margin of \
error for numerical problems, a margin of 1 to 5.5 percentage points is \
acceptable. List all of the [response] answer appearing in [correct_answer] \
with each answer delimited by '###'. If the number of overlapping answers is \
zero, output 'NULL'."""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_document_text(doc_name: str) -> str:
    """Read a markdown document from the wiki_md directory."""
    with open(MD_PATH / doc_name, "r", encoding="utf-8") as f:
        return f.read()


def format_gold_answer(answer: Any) -> str:
    """Format the gold answer into a readable string for the judge prompt.

    Handles various answer formats from MoNaCo:
    - Single value: "White" or 38.46
    - List of strings: ["Portugal", "France"]
    - List of lists: [["Columbia University", 46], ...]
    """
    if isinstance(answer, list):
        if len(answer) == 0:
            return "None"
        if all(isinstance(item, list) for item in answer):
            return ", ".join(
                " - ".join(str(x) for x in item) for item in answer
            )
        return ", ".join(str(item) for item in answer)
    return str(answer)


def count_gold_answers(answer: Any) -> int:
    """Count the number of distinct answer items in the gold answer."""
    if isinstance(answer, list):
        return max(len(answer), 1)
    return 1


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD (Oracle Retrieval Setup)
# ═══════════════════════════════════════════════════════════════════════════════


async def upload_documents(data: List[Dict[str, Any]]):
    """Upload all oracle documents for the given questions to the vector store."""
    pipeline = VectorStorePipeline()
    total = sum(len(task["doc_paths"]) for task in data)
    uploaded = 0

    for task in data:
        qid = task["id"]
        question = task["question"]
        for doc_name in task["doc_paths"]:
            try:
                text = load_document_text(doc_name)
                clean_name = doc_name.split(".md")[0]
                await pipeline.run(
                    text, clean_name,
                    metadata={"id": qid, "question": question},
                )
                uploaded += 1
                print(f"  [{uploaded}/{total}] Uploaded: {clean_name}")
            except FileNotFoundError:
                print(f"  [WARN] Document not found: {doc_name}")

    print(f"Upload complete: {uploaded}/{total} documents.")


# ═══════════════════════════════════════════════════════════════════════════════
# SPD-RAG INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


async def run_spd_rag(question: str, doc_paths: List[str]) -> Dict[str, Any]:
    """Run the SPD-RAG multi-agent system on a question with oracle docs.

    This is the Oracle Retrieval setting: all gold evidence documents are
    provided via selected_documents so the system can focus on reasoning.
    """
    from langchain_core.messages import HumanMessage
    from backend.core.graph import get_compiled_graph

    graph = get_compiled_graph()
    user_message = f"Question: {question}"
    prompt_tokens = count_tokens(user_message)

    start = time.perf_counter()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "selected_documents": doc_paths,
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


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE (MoNaCo Figure 12)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_judge(
    question: str,
    gold_answer_str: str,
    predicted: str,
    model: str = "gpt-4.1",
) -> Dict[str, Any]:
    """Send the predicted answer to the MoNaCo LLM judge and parse the result."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model=model, temperature=0.0)
    prompt = MONACO_JUDGE_PROMPT.format(
        question=question,
        response=predicted,
        correct_answer=gold_answer_str,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return parse_judge_response(response.content)


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the structured fields from the judge's output.

    Extracts: final_answer_length, correct, final_precision, overlapping_answers.
    """
    result: Dict[str, Any] = {
        "final_answer_length": 0,
        "correct": False,
        "final_precision": 0.0,
        "overlapping_answers": [],
        "raw_judge_output": text,
    }

    match = re.search(r"final answer length:\s*(\d+)", text, re.IGNORECASE)
    if match:
        result["final_answer_length"] = int(match.group(1))

    match = re.search(r"\ncorrect:\s*(yes|no)", text, re.IGNORECASE)
    if match:
        result["correct"] = match.group(1).lower() == "yes"

    match = re.search(r"final precision:\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        try:
            result["final_precision"] = float(match.group(1))
        except ValueError:
            result["final_precision"] = 0.0

    match = re.search(
        r"overlapping answers:\s*(.+?)(?:\n\n|\n[a-zA-Z]|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        raw = match.group(1).strip()
        if raw.upper() == "NULL" or raw.upper() == "NONE":
            result["overlapping_answers"] = []
        else:
            result["overlapping_answers"] = [
                a.strip() for a in raw.split("###") if a.strip()
            ]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    judge_result: Dict[str, Any],
    gold_answer_count: int,
) -> Dict[str, float]:
    """Compute Precision, Recall, F1 from the judge output.

    For single/numerical answers:
        precision = recall = final_precision (from the judge's similarity score)
    For list answers:
        precision = |overlapping| / |predicted|
        recall    = |overlapping| / |gold|
        F1        = harmonic mean of precision and recall
    """
    num_overlapping = len(judge_result["overlapping_answers"])
    num_predicted = judge_result["final_answer_length"]

    if gold_answer_count <= 1:
        precision = judge_result["final_precision"]
        recall = judge_result["final_precision"]
    else:
        precision = num_overlapping / num_predicted if num_predicted > 0 else 0.0
        recall = num_overlapping / gold_answer_count if gold_answer_count > 0 else 0.0

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARIZE EXISTING RESULTS
# ═══════════════════════════════════════════════════════════════════════════════


def summarize_results(results_path: str):
    """Print aggregate metrics from a previously saved results JSONL file."""
    results = load_data(results_path)
    if not results:
        print("No results found.")
        return

    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]
    n = len(results)

    avg_p = sum(precisions) / n
    avg_r = sum(recalls) / n
    avg_f1 = sum(f1s) / n
    perfect_pct = sum(1 for f in f1s if f >= 1.0) / n * 100
    correct_pct = sum(1 for r in results if r.get("correct")) / n * 100
    avg_latency = sum(r.get("latency", 0) for r in results) / n

    print("\n" + "=" * 60)
    print("  MoNaCo Evaluation Summary (Oracle Retrieval - SPD-RAG)")
    print("=" * 60)
    print(f"  Questions evaluated : {n}")
    print(f"  Avg Precision       : {avg_p:.4f}")
    print(f"  Avg Recall          : {avg_r:.4f}")
    print(f"  Avg F1              : {avg_f1:.4f}")
    print(f"  Perfect score (%)   : {perfect_pct:.1f}%")
    print(f"  Correct (%)         : {correct_pct:.1f}%")
    print(f"  Avg Latency (s)     : {avg_latency:.1f}")
    print("=" * 60)

    # Comparison reference from the paper (Table 6, Oracle setting)
    print("\n  Reference from MoNaCo paper (Oracle Retrieval):")
    print("  ─────────────────────────────────────────────────")
    print("  GPT-4o       : P=67.28  R=56.08  F1=58.67")
    print("  LLaMA 3.1-405B: P=66.68  R=56.57  F1=58.83")
    print("  ─────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate(
    data_path: str = str(DEFAULT_DATA),
    results_path: str = str(DEFAULT_RESULTS),
    upload_docs: bool = False,
    question_ids: Optional[List[int]] = None,
    judge_model: str = "gpt-4.1",
):
    """Run the full MoNaCo evaluation pipeline.

    Steps for each question:
    1. Run SPD-RAG with oracle documents (Oracle Retrieval setting)
    2. Send response to the MoNaCo LLM judge (Figure 12)
    3. Parse judge output and compute Precision, Recall, F1
    4. Log per-question results to JSONL
    5. Print aggregate summary at the end
    """
    data = load_data(data_path)
    print(f"Loaded {len(data)} questions from {data_path}")

    if question_ids:
        data = [d for d in data if d["id"] in question_ids]
        print(f"Filtered to {len(data)} questions: {question_ids}")

    if not data:
        print("No questions to evaluate.")
        return

    if upload_docs:
        print("\n── Uploading oracle documents to vector store ──")
        await upload_documents(data)
        print()

    all_results: List[Dict[str, Any]] = []

    with open(results_path, "a", encoding="utf-8") as f:
        for i, task in enumerate(data):
            qid = task["id"]
            question = task["question"]
            answer = task["answer"]
            doc_paths = task["doc_paths"]
            gold_count = count_gold_answers(answer)

            print(f"\n── [{i + 1}/{len(data)}] Q{qid} ({gold_count} gold answers) ──")
            print(f"   {question[:100]}...")

            # Step 1: Run SPD-RAG
            try:
                rag_result = await run_spd_rag(question, doc_paths)
            except Exception as e:
                print(f"   [ERROR] SPD-RAG failed: {e}")
                continue
            print(f"   Latency: {rag_result['latency']:.1f}s")

            # Step 2: Run MoNaCo LLM judge
            gold_str = format_gold_answer(answer)
            try:
                judge_result = await run_judge(
                    question, gold_str, rag_result["raw_output"],
                    model=judge_model,
                )
            except Exception as e:
                print(f"   [ERROR] Judge failed: {e}")
                continue

            # Step 3: Compute metrics
            metrics = compute_metrics(judge_result, gold_count)
            print(
                f"   P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  "
                f"F1={metrics['f1']:.3f}  "
                f"Correct={judge_result['correct']}"
            )

            # Step 4: Save result
            result_entry = {
                "id": qid,
                "question": question,
                "gold_answer": answer,
                "gold_answer_count": gold_count,
                "predicted_answer": rag_result["raw_output"],
                "latency": rag_result["latency"],
                "prompt_tokens": rag_result["prompt_tokens"],
                "completion_tokens": rag_result["completion_tokens"],
                "correct": judge_result["correct"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "final_answer_length": judge_result["final_answer_length"],
                "overlapping_answers": judge_result["overlapping_answers"],
                "raw_judge_output": judge_result["raw_judge_output"],
            }
            f.write(json.dumps(result_entry) + "\n")
            f.flush()
            all_results.append(result_entry)

    # Step 5: Print summary
    if all_results:
        # Write a temp summary using the just-collected results
        n = len(all_results)
        avg_p = sum(r["precision"] for r in all_results) / n
        avg_r = sum(r["recall"] for r in all_results) / n
        avg_f1 = sum(r["f1"] for r in all_results) / n
        perfect_pct = sum(1 for r in all_results if r["f1"] >= 1.0) / n * 100
        correct_pct = sum(1 for r in all_results if r["correct"]) / n * 100

        print("\n" + "=" * 60)
        print("  MoNaCo Evaluation Results (Oracle Retrieval - SPD-RAG)")
        print("=" * 60)
        print(f"  Questions evaluated : {n}")
        print(f"  Avg Precision       : {avg_p:.4f}")
        print(f"  Avg Recall          : {avg_r:.4f}")
        print(f"  Avg F1              : {avg_f1:.4f}")
        print(f"  Perfect score (%)   : {perfect_pct:.1f}%")
        print(f"  Correct (%)         : {correct_pct:.1f}%")
        print("=" * 60)
        print("\n  Reference (MoNaCo paper, Oracle Retrieval):")
        print("  GPT-4o        : P=67.28  R=56.08  F1=58.67")
        print("  LLaMA 3.1-405B: P=66.68  R=56.57  F1=58.83")
        print(f"\n  Results saved to: {results_path}")
    else:
        print("\nNo results were collected.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MoNaCo Benchmark Evaluator for SPD-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python benchmark/new_monaco_evaluator.py --data benchmark/data/3q_withmds.jsonl --upload-docs
  python benchmark/new_monaco_evaluator.py --ids 1621 215
  python benchmark/new_monaco_evaluator.py --summarize benchmark/evaluation_results.jsonl
""",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Path to JSONL data file (default: 100q_withmds.jsonl)",
    )
    parser.add_argument(
        "--results", default=str(DEFAULT_RESULTS),
        help="Path to write/append results (default: evaluation_results.jsonl)",
    )
    parser.add_argument(
        "--upload-docs", action="store_true",
        help="Upload oracle documents to vector store before evaluating",
    )
    parser.add_argument(
        "--ids", type=int, nargs="+",
        help="Evaluate only specific question IDs",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4.1",
        help="Model to use as judge (default: gpt-4.1, as in the paper)",
    )
    parser.add_argument(
        "--summarize", type=str, metavar="FILE",
        help="Just print summary metrics from an existing results file, then exit",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize_results(args.summarize)
    else:
        asyncio.run(
            evaluate(
                data_path=args.data,
                results_path=args.results,
                upload_docs=args.upload_docs,
                question_ids=args.ids,
                judge_model=args.judge_model,
            )
        )
