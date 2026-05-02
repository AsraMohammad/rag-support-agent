"""
Phase 6: Run all test cases through the RAG pipeline and produce a scorecard.
Outputs eval_results.json and a summary table.
"""
import json
import time
from pathlib import Path

# Import the answer function from rag.py
# (We'll need to import it cleanly — see note below)
from rag import answer_question

TESTS_FILE = Path("evals/test_cases.json")
RESULTS_FILE = Path("evals/eval_results.json")

with open(TESTS_FILE, "r", encoding="utf-8") as f:
    test_cases = json.load(f)

print(f"Running {len(test_cases)} test cases...\n")
print(f"{'ID':<5} {'Verdict':<8} {'Match':<7} {'Time':<8} Question")
print("-" * 80)

results = []
total_input_tokens = 0
total_output_tokens = 0
verifier_input_tokens = 0
verifier_output_tokens = 0
total_time = 0
correct_verdicts = 0

for tc in test_cases:
    start = time.time()
    result = answer_question(tc["question"])
    elapsed = time.time() - start
    
    actual_verdict = result["verification"]["verdict"]
    expected_verdict = tc["expected_verdict"]
    match = "✓" if actual_verdict == expected_verdict else "✗"
    if actual_verdict == expected_verdict:
        correct_verdicts += 1
    
    # Accumulate metrics
    total_input_tokens += result["tokens"]["input"]
    total_output_tokens += result["tokens"]["output"]
    v_tokens = result["verification"].get("tokens", {})
    verifier_input_tokens += v_tokens.get("input", 0)
    verifier_output_tokens += v_tokens.get("output", 0)
    total_time += elapsed
    
    # Compact display
    q_short = tc["question"][:40] + "..." if len(tc["question"]) > 40 else tc["question"]
    print(f"{tc['id']:<5} {actual_verdict:<8} {match:<7} {elapsed:<7.1f}s {q_short}")
    
    # Record full result
    results.append({
        "id": tc["id"],
        "question": tc["question"],
        "expected_behavior": tc["expected_behavior"],
        "expected_verdict": expected_verdict,
        "actual_verdict": actual_verdict,
        "match": actual_verdict == expected_verdict,
        "answer": result["answer"],
        "raw_answer": result.get("raw_answer", result["answer"]),
        "verification_summary": result["verification"].get("summary", ""),
        "sources": [
            {"source": s["source"], "similarity": s["similarity"]}
            for s in result["sources"]
        ],
        "elapsed_seconds": round(elapsed, 2),
        "tokens": {
            "answerer": result["tokens"],
            "verifier": v_tokens,
        },
    })

# Summary
print("-" * 80)
n = len(test_cases)
verdict_accuracy = correct_verdicts / n * 100
avg_latency = total_time / n

# Cost estimation (Sonnet 4.5: $3/M in, $15/M out; Haiku 4.5: $1/M in, $5/M out approx)
answerer_cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
verifier_cost = (verifier_input_tokens * 1 + verifier_output_tokens * 5) / 1_000_000
total_cost = answerer_cost + verifier_cost

summary = {
    "total_tests": n,
    "verdict_accuracy_percent": round(verdict_accuracy, 1),
    "correct_verdicts": correct_verdicts,
    "avg_latency_seconds": round(avg_latency, 2),
    "total_runtime_seconds": round(total_time, 2),
    "total_tokens": {
        "answerer_input": total_input_tokens,
        "answerer_output": total_output_tokens,
        "verifier_input": verifier_input_tokens,
        "verifier_output": verifier_output_tokens,
    },
    "estimated_cost_usd": round(total_cost, 4),
}

print(f"\nSummary:")
print(f"  Verdict accuracy:     {verdict_accuracy:.1f}% ({correct_verdicts}/{n})")
print(f"  Avg latency:          {avg_latency:.2f}s per query")
print(f"  Total runtime:        {total_time:.2f}s")
print(f"  Estimated cost:       ${total_cost:.4f}")
print(f"  Tokens used (total):  {total_input_tokens + total_output_tokens + verifier_input_tokens + verifier_output_tokens:,}")

# Save full results
RESULTS_FILE.parent.mkdir(exist_ok=True)
output = {"summary": summary, "results": results}
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nFull results saved to: {RESULTS_FILE}")