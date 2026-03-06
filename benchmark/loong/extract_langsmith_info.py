from annotated_types import Len
from langsmith import Client
from typing import Optional, Sequence
from dotenv import load_dotenv
import os,json
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
    latency = run.latency

    return {
        "run_id": str(run.id),
        "name": run.name,
        "model": getattr(run, 'metadata')['model'] if getattr(run, 'metadata') and 'model' in getattr(run, 'metadata') else None,
        "mode": getattr(run, 'metadata')['mode'] if getattr(run, 'metadata') and 'mode' in getattr(run, 'metadata') else None,
        "task_id": getattr(run, 'metadata')['task_id'] if getattr(run, 'metadata') and 'task_id' in getattr(run, 'metadata') else None,
        "is_error": getattr(run, 'error', None) is not None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "latency": latency,
        "start_time": run.start_time,
        "status": run.status,
    }


def get_run_usage(
    run_id: Optional[str] = None,
    *,
    project_name: Optional[str] = None,
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

    from datetime import datetime, timedelta, timezone

    start_time = datetime(2026, 3, 4, 0, 0, 0, tzinfo=timezone.utc)
    run_filter = f'gt(start_time, "{start_time.isoformat()}")'

    runs_iter = client.list_runs(
        project_name=project_name,
        is_root=True,           # often you only care about root traces; drop if you want all [web:19]
        # order_by="start_time",  # descending, newest first (string form per docs) [web:19]
        # limit=n_recent,         # only fetch N most recent runs [web:19]
        filter=run_filter,
    )
    # a = 0
    # for i in runs_iter:
    #     if a==1:
    #         print(i)
    #         break
    #     a += 1

    return [summarize_run(r) for r in runs_iter]


# --- Examples ---


INPUT_JSONL = "baseline_25pro_p3.jsonl"
OUTPUT_JSONL = INPUT_JSONL.replace(".jsonl", "_other_metrics.jsonl")

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

task_id_to_row = {row["id"]: row for row in rows}
print("len(task_id_to_row):", len(task_id_to_row))
recent = get_run_usage(project_name=PROJECT_NAME)
print("len(recent):", len(recent))
runs_by_task = {}
for r in recent:
    if (
        r["name"] != "ChatOpenAI"
        and r["model"] == "gemini-2.5-pro"
        and r["is_error"] == False
        and r["mode"] == "baseline_llm"
        and r["task_id"] in task_id_to_row
        and r["status"] == "success"
        # and (r["latency"] < 390 or r["latency"] > 415)
    ):
        runs_by_task.setdefault(r["task_id"], []).append(r)

def pick_latest_consecutive_session(runs: list, gap_minutes: float = 5.0) -> dict:
    """Given a list of runs for a task, find the largest consecutive session
    (no gap > gap_minutes between adjacent runs) and return an aggregated row.
    If multiple sessions exist, the latest one (by start_time) is chosen.
    """
    if not runs:
        raise ValueError("No runs provided")

    sorted_runs = sorted(runs, key=lambda r: r["start_time"])

    # Split into sessions: a new session starts when the gap exceeds gap_minutes
    sessions = []
    current = [sorted_runs[0]]
    for prev, curr in zip(sorted_runs, sorted_runs[1:]):
        gap = (curr["start_time"] - prev["start_time"]).total_seconds() / 60
        if gap > gap_minutes:
            sessions.append(current)
            current = [curr]
        else:
            current.append(curr)
    sessions.append(current)

    # Pick the latest session (highest start_time)
    latest_session = max(sessions, key=lambda s: s[0]["start_time"])

    return {
        "run_id": latest_session[-1]["run_id"],  # last run id as representative
        "name": latest_session[0]["name"],
        "model": latest_session[0]["model"],
        "mode": latest_session[0]["mode"],
        "task_id": latest_session[0]["task_id"],
        "is_error": any(r["is_error"] for r in latest_session),
        "input_tokens": sum(r["input_tokens"] for r in latest_session),
        "output_tokens": sum(r["output_tokens"] for r in latest_session),
        "total_tokens": sum(r["total_tokens"] for r in latest_session),
        "total_cost_usd": sum(r["total_cost_usd"] or 0 for r in latest_session),
        "latency": sum(r["latency"] for r in latest_session),
        "start_time": latest_session[0]["start_time"],
        "status": "success" if all(r["status"] == "success" for r in latest_session) else "error",
        "session_run_count": len(latest_session),
    }


print("len(runs_by_task):", len(runs_by_task))
enriched = []
for task_id, row in task_id_to_row.items():
    matched = runs_by_task.get(task_id, [])
    if len(matched) == 0:
        print(f"SKIP {task_id}: no runs matched")
        continue
    # aggregated = pick_latest_consecutive_session(matched)
    # print(aggregated["session_run_count"])
    # enriched_row = {
    #     **row,
    #     "ls_run_id": aggregated["run_id"],
    #     "ls_input_tokens": aggregated["input_tokens"],
    #     "ls_output_tokens": aggregated["output_tokens"],
    #     "ls_total_tokens": aggregated["total_tokens"],
    #     "ls_cost_usd": aggregated["total_cost_usd"],
    #     "ls_latency": aggregated["latency"],
    # }
    # enriched.append(enriched_row)
    if len(matched) >= 1:
        print(f"INFO {task_id}: {len(matched)} runs matched — saving as separate rows")
        print(matched)
    for r in matched:
        enriched_row = {
            **row,
            "ls_run_id": r["run_id"],
            "ls_input_tokens": r["input_tokens"],
            "ls_output_tokens": r["output_tokens"],
            "ls_total_tokens": r["total_tokens"],
            "ls_cost_usd": r["total_cost_usd"],
            "ls_latency": r["latency"],
        }
        enriched.append(enriched_row)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for row in enriched:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nSaved {len(enriched)} enriched rows to {OUTPUT_JSONL}")
print(f"Skipped: {len(rows) - len(enriched)}")





