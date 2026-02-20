# Benchmark Evaluators

## Loong

Evaluates SPD-RAG on the [Loong](https://github.com/MozerWang/Loong) multi-document QA benchmark using a GPT-4 judge (scores 0–100). Reports **Avg Score** and **Perfect Rate**.

### Basic usage

```bash
# Run on set1 (default), all tasks
uv run python benchmark/loong/loong_evaluator.py

# Specify a different set
uv run python benchmark/loong/loong_evaluator.py --data benchmark/loong/data/loong_set2.jsonl
```

### Filtering & pagination

| Flag | Description |
|---|---|
| `--level 1\|2\|3\|4` | 1=Spotlight, 2=Comparison, 3=Clustering, 4=Chain of Reasoning |
| `--language en\|zh` | Filter by language |
| `--offset N` | Skip first N tasks (after level/language filtering) |
| `--limit N` | Evaluate at most N tasks |

```bash
# Quick smoke test: 3 English tasks, upload docs first
uv run python benchmark/loong/loong_evaluator.py --language en --limit 3 --upload-docs

# Resume from task 10, evaluate next 20
uv run python benchmark/loong/loong_evaluator.py --offset 10 --limit 20

# Only Chain of Reasoning tasks, Chinese, starting at task 5
uv run python benchmark/loong/loong_evaluator.py --level 4 --language zh --offset 5 --limit 10
```

### Other flags

```bash
# Custom results output path
uv run python benchmark/loong/loong_evaluator.py --results my_results.jsonl

# Summarize an existing results file
uv run python benchmark/loong/loong_evaluator.py --summarize benchmark/loong/loong_results.jsonl

# Change the judge model (default: gpt-4.1)
uv run python benchmark/loong/loong_evaluator.py --judge-model gpt-4o --limit 5
```

---

## MoNaCo

Evaluates SPD-RAG on the [MoNaCo](https://github.com/tomerwolgithub/monaco) benchmark using an LLM judge. Reports **Precision**, **Recall**, and **F1**.

```bash
# Run on 100-question set
uv run python benchmark/monaco/new_monaco_evaluator.py

# Evaluate specific question IDs
uv run python benchmark/monaco/new_monaco_evaluator.py --ids 1621 215 1053

# Summarize existing results
uv run python benchmark/monaco/new_monaco_evaluator.py --summarize benchmark/monaco/evaluation_results.jsonl
```
