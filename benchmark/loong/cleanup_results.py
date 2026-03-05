import json

INPUT_JSONL = "baseline_25pro_p3_other_metrics.jsonl"
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
# print(rows[0].keys())
OUTPUT_JSONL = INPUT_JSONL.replace(".jsonl", "_cleanup.jsonl")
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for row in rows:
        new_row = {
            "id": row["id"],
            "mode": row["mode"],
            "level_name": row["level_name"],
            "type": row["type"],
            "num_docs": row["num_docs"],
            "score": row["score"],
            "iterations": row["iterations"],
            "ls_run_id": row["ls_run_id"],
            "ls_input_tokens": row["ls_input_tokens"],
            "ls_output_tokens": row["ls_output_tokens"],
            "ls_total_tokens": row["ls_total_tokens"],
            "ls_cost_usd": row["ls_cost_usd"],
            "ls_latency": row["ls_latency"],
        }
        f.write(json.dumps(new_row, ensure_ascii=False) + "\n")