from langsmith import Client
from typing import Optional, Sequence
from dotenv import load_dotenv
import os
load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

client = Client()


def summarize_run(run) -> dict:
    """Extract token + cost info from a single LangSmith Run."""
    input_tokens = getattr(run, "input_tokens", None) or getattr(run, "prompt_tokens", None) or 0  # [web:12]
    output_tokens = getattr(run, "output_tokens", None) or getattr(run, "completion_tokens", None) or 0  # [web:12]
    total_tokens = getattr(run, "total_tokens", None) or (input_tokens + output_tokens)  # [web:12]

    total_cost = getattr(run, "total_cost", None)  # Decimal or float, may be None [web:23]
    total_cost_usd = float(total_cost) if total_cost is not None else None

    return {
        "run_id": str(run.id),
        "name": run.name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
    }


def get_run_usage(
    run_id: Optional[str] = None,
    *,
    project_name: Optional[str] = None,
    n_recent: int = 5,
) -> Sequence[dict]:
    """
    If run_id is provided, return usage for that run.
    Otherwise, return usage for the most recent n_recent runs in a project.
    """
    if run_id:
        run = client.read_run(run_id)  # [web:19]
        return [summarize_run(run)]

    if not project_name:
        raise ValueError("project_name is required when run_id is not provided.")

    # Most recent runs: order by start_time desc, limit = n_recent
    runs_iter = client.list_runs(
        project_name=project_name,
        is_root=True,           # often you only care about root traces; drop if you want all [web:19]
        order_by="-start_time",  # descending, newest first (string form per docs) [web:19]
        limit=n_recent,         # only fetch N most recent runs [web:19]
    )

    return [summarize_run(r) for r in runs_iter]


# --- Examples ---


# 2) Most recent 3 runs in a project
recent = get_run_usage(project_name=PROJECT_NAME, n_recent=2)
for r in recent:
    print(r)
    