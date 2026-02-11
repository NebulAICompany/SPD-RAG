"""
Test the Hybrid Similarity-Ordered Recursive Summarization pipeline.

Layers tested:
  1. _estimate_tokens        — pure logic, no API
  2. _group_by_tokens        — pure logic with mock clustering data
  3. generate_embeddings     — requires COHERE_API_KEY
  4. recursive_summarize_findings — full flow (Cohere + LLM)
"""

import asyncio
import sys
import os
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 1. Test _estimate_tokens ─────────────────────────────────────────────────

def test_estimate_tokens():
    from backend.core.nodes import _estimate_tokens

    assert _estimate_tokens("") == 0
    assert _estimate_tokens("hello world") > 0

    short = _estimate_tokens("short")
    long = _estimate_tokens("This is a much longer sentence with many more tokens")
    assert long > short

    print(f"  ✅ _estimate_tokens: '' → 0, 'hello world' → {_estimate_tokens('hello world')}, ordering OK")


# ── 2. Test _group_by_tokens ─────────────────────────────────────────────────

def test_group_by_tokens():
    from backend.core.nodes import _group_by_tokens, _estimate_tokens

    texts = [
        "Alpha bravo charlie.",       # ~4 tokens
        "Delta echo foxtrot.",        # ~4 tokens
        "Golf hotel india.",          # ~4 tokens
        "Juliet kilo lima.",          # ~4 tokens
    ]

    # Simulate sklearn children_ for 4 samples:
    #   merge 0: (0, 1) → node 4
    #   merge 1: (2, 3) → node 5
    #   merge 2: (4, 5) → node 6
    children = np.array([[0, 1], [2, 3], [4, 5]])

    # Case A: large limit → everything merges into 1 batch
    batches = _group_by_tokens(texts, children, target_tokens=500)
    total_items = sum(len(b) for b in batches)
    assert total_items == 4, f"Expected 4 items total, got {total_items}"
    assert len(batches) == 1, f"Expected 1 batch with large limit, got {len(batches)}"
    print(f"  ✅ _group_by_tokens (large limit): {len(batches)} batch, {total_items} items")

    # Case B: tiny limit → no merges possible, each text is its own batch
    batches = _group_by_tokens(texts, children, target_tokens=1)
    assert len(batches) == 4, f"Expected 4 batches with tiny limit, got {len(batches)}"
    print(f"  ✅ _group_by_tokens (tiny limit):  {len(batches)} batches (no merging)")

    # Case C: medium limit → allows pair merges but not the final merge
    pair_tokens = _estimate_tokens(texts[0]) + _estimate_tokens(texts[1])
    quad_tokens = sum(_estimate_tokens(t) for t in texts)
    mid_limit = pair_tokens + 1  # fits a pair, but not all 4
    if mid_limit < quad_tokens:
        batches = _group_by_tokens(texts, children, target_tokens=mid_limit)
        assert len(batches) == 2, f"Expected 2 batches with mid limit, got {len(batches)}"
        print(f"  ✅ _group_by_tokens (mid limit):   {len(batches)} batches (pair grouping)")
    else:
        print(f"  ⚠️  Skipped mid-limit test (tokens too small to split)")


# ── 3. Test generate_embeddings ──────────────────────────────────────────────

async def test_generate_embeddings():
    from backend.pipeline.vector import generate_embeddings

    # Empty input
    result = await generate_embeddings([])
    assert result == [], f"Expected empty list, got {result}"
    print("  ✅ generate_embeddings([]): returned []")

    # Real embeddings
    texts = ["Machine learning is fascinating.", "Deep learning uses neural networks."]
    embeddings = await generate_embeddings(texts)

    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert len(embeddings[0]) == 1536, f"Expected dim 1536, got {len(embeddings[0])}"
    print(f"  ✅ generate_embeddings: {len(embeddings)} vectors, dim={len(embeddings[0])}")

    # Verify similarity: similar texts should have higher cosine sim
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"  ✅ Cosine similarity between related texts: {sim:.4f}")
    assert sim > 0.5, f"Expected high similarity for related texts, got {sim}"


# ── 4. Test full recursive_summarize_findings ────────────────────────────────

async def test_recursive_summarize():
    from backend.core.nodes import recursive_summarize_findings

    # 12 substantial findings across different topics — enough to force multiple iterations
    findings = [
        "Document: Finance_Q1\nRelevance: 0.95\nFindings:\nQ1 revenue reached $4.2B, a 15% increase year-over-year. Cloud services segment contributed $1.8B, growing 32% driven by enterprise adoption of hybrid cloud solutions and AI infrastructure services.",
        "Document: Finance_Q2\nRelevance: 0.93\nFindings:\nQ2 revenue was $4.5B with operating income of $1.1B. Gross margins expanded to 68% from 62% due to favorable product mix shift toward higher-margin software subscriptions and reduced hardware dependency.",
        "Document: Finance_Q3\nRelevance: 0.91\nFindings:\nQ3 showed continued momentum with $4.8B revenue. Free cash flow generation improved to $900M as working capital management initiatives reduced DSO from 45 to 38 days.",
        "Document: Finance_Q4\nRelevance: 0.90\nFindings:\nQ4 closed the fiscal year at $5.1B revenue, bringing FY total to $18.6B. The board approved a 20% dividend increase and $2B share buyback program reflecting confidence in future cash flows.",
        "Document: HR_Report\nRelevance: 0.82\nFindings:\nHeadcount grew 8% to 45,000 employees. Voluntary attrition decreased from 18% to 12% following compensation restructuring. Engineering talent acquisition improved with 40% faster time-to-hire through AI-assisted screening.",
        "Document: Engineering_Review\nRelevance: 0.88\nFindings:\nR&D investment reached 18% of revenue ($3.3B). Key milestones included launching the next-gen AI platform, achieving 99.99% uptime SLA on core services, and reducing deployment cycle time from 2 weeks to 3 days.",
        "Document: Marketing_Analysis\nRelevance: 0.79\nFindings:\nCustomer acquisition cost decreased 12% to $340 per enterprise customer. Brand awareness in target segments increased from 45% to 67%. Digital marketing ROI improved 28% through programmatic advertising optimization.",
        "Document: Sales_Pipeline\nRelevance: 0.86\nFindings:\nEnterprise pipeline grew to $8.2B, representing 2.1x coverage. Win rates improved from 31% to 38% through enhanced solution selling methodology. Average deal size increased 22% to $1.2M.",
        "Document: Product_Roadmap\nRelevance: 0.84\nFindings:\nThree major product launches planned for next fiscal year: GenAI Studio (Q1), Edge Computing Platform (Q2), and Unified Data Mesh (Q3). Patent portfolio expanded by 340 filings, bringing total to 12,400.",
        "Document: Risk_Assessment\nRelevance: 0.77\nFindings:\nKey risks include regulatory changes in EU AI Act compliance, potential supply chain disruptions in semiconductor procurement, and competitive pressure from hyperscalers entering adjacent markets.",
        "Document: ESG_Report\nRelevance: 0.72\nFindings:\nCarbon emissions reduced 25% toward 2030 net-zero target. Renewable energy usage reached 78% of data center operations. Diversity metrics improved with women in technical roles increasing from 28% to 34%.",
        "Document: Customer_Satisfaction\nRelevance: 0.80\nFindings:\nNPS score improved from 42 to 58. Enterprise customer retention rate reached 96%. Support ticket resolution time decreased 35% through AI-powered triage. Top customer complaints centered on API documentation quality.",
    ]

    print(f"  ⏳ Running recursive_summarize_findings with {len(findings)} chunks...")
    print(f"     Using target_batch_tokens=1200 (production default)\n")

    summary = await recursive_summarize_findings(
        raw_findings=findings,
        root_query="What are the key financial, operational, and strategic highlights across all departments?",
        target_batch_tokens=600,
    )

    assert isinstance(summary, str), f"Expected string, got {type(summary)}"
    assert len(summary) > 50, f"Summary too short ({len(summary)} chars)"
    print(f"\n  ✅ recursive_summarize_findings: produced {len(summary)} char summary")
    print(f"  📝 Preview:\n{'─'*50}")
    print(f"  {summary[:400]}...")
    print(f"{'─'*50}")


# ── Runner ───────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "="*60)
    print("  Recursive Summarization Pipeline Tests")
    print("="*60)

    print("\n[1/4] Testing _estimate_tokens...")
    test_estimate_tokens()

    print("\n[2/4] Testing _group_by_tokens...")
    test_group_by_tokens()

    print("\n[3/4] Testing generate_embeddings (requires COHERE_API_KEY)...")
    await test_generate_embeddings()

    print("\n[4/4] Testing recursive_summarize_findings (full flow)...")
    await test_recursive_summarize()

    print("\n" + "="*60)
    print("  All tests passed! ✅")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
