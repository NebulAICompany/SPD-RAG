import json
import asyncio
import time
import uuid
import sys
from pathlib import Path
from typing import Dict, Any
import tiktoken
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.vector import VectorStorePipeline

# Tiktoken encoder for accurate token counts (cl100k_base covers GPT-4 family)
_ENCODER: Optional[tiktoken.Encoding] = None

MD_PATH = PROJECT_ROOT / "benchmark" / "data" / "wiki_md_esradan"

def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Return the exact token count using the cl100k_base encoding."""
    return len(_get_encoder().encode(text))

_JUDGE_PROMPT = """You are an impartial judge evaluating whether a predicted answer matches the expected gold answer for a question.

Question: {question}
Expected (gold) answer: {expected}
Predicted answer: {predicted}

Evaluate STRICTLY:
- If the predicted answer is semantically equivalent to the expected answer, respond with: CORRECT
- If the predicted answer is partially correct (contains some but not all required information), respond with: PARTIAL
- If the predicted answer is wrong or unrelated, respond with: INCORRECT

Respond with exactly one word: CORRECT, PARTIAL, or INCORRECT.
"""

def load_monaco_data(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def load_document_text(doc_name: str):
    with open(MD_PATH / doc_name, "r", encoding="utf-8") as f:
        text = f.read()
    return text

async def _upload_single_document(doc: Dict[str, Any]):
    text_content = doc["text_content"]
    document_name = doc["document_name"]
    metadata = doc["metadata"]
    pipeline = VectorStorePipeline()
    await pipeline.run(text_content, document_name, metadata)

async def upload_documents_to_vectorstore(data: list[Dict[str, Any]]):
    for task in data:
        id = task["id"]
        question = task["question"]
        answer = task["answer"]
        doc_paths = task["doc_paths"]
        for doc_name in doc_paths:
            text = load_document_text(doc_name)
            doc_name = doc_name.split(".md")[0]
            print(f"Doc name: {doc_name}")
            doc = {
                "text_content": text,
                "document_name": doc_name,
                "metadata": {"id": id, "question": question}
            }
            await _upload_single_document(doc)

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


async def perform_spd_rag(question: str, selected_files: list[str]):
    from langchain_core.messages import HumanMessage
    from backend.core.graph import get_compiled_graph

    graph = get_compiled_graph()

    user_message = (
        f"Question: {question}\n\n"
        "Provide your final answer inside \\boxed{{}}."
    )
    prompt_tokens = count_tokens(user_message)

    start = time.perf_counter()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "selected_documents": selected_files,
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


async def evaluate_with_llm_judge(doc_paths: list[Dict[str, Any]], question: str, answer: str):
    selected_files = doc_paths
    print(f"Selected files: {selected_files}")
    response = await perform_spd_rag(question, selected_files)
    print(f"Response: {response}")
    judge_response = await llm_judge(question, answer, response["raw_output"])
    print(f"Judge response: {judge_response}")
    return {
        "raw_output": response["raw_output"],
        "latency": response["latency"],
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
        "judge_response": judge_response,
    }


async def evaluate_monaco_data(data: list[Dict[str, Any]]):
    test_ids = [1621]#1621, 215, 1053]

    log_file = PROJECT_ROOT / "benchmark" / "monaco_question_docs_answers_with_md_results.jsonl"
    with open(log_file, "a+", encoding="utf-8") as f:
        for task in data:
            if task["id"] not in test_ids:
                continue
            response = await evaluate_with_llm_judge(task["doc_paths"], task["question"], task["answer"])
            response["id"] = task["id"]
            response["question"] = task["question"]
            response["answer"] = task["answer"]
            print(f"Task {task['id']}: {response}")
            f.write(json.dumps(response) + "\n")
            f.flush()





if __name__ == "__main__":
    data = load_monaco_data(PROJECT_ROOT / "benchmark" / "data" / "monaco_question_docs_answers_with_md.jsonl")
    UPLOAD_DOCS_BOOL = False
    if UPLOAD_DOCS_BOOL:
        asyncio.run(upload_documents_to_vectorstore(data))
    asyncio.run(evaluate_monaco_data(data))