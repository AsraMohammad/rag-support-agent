"""
Phase 5: Grounding verifier.
Takes an answer + the chunks it was generated from, decides whether 
every factual claim in the answer is actually supported by the chunks.
Returns a structured pass/fail with reasoning.
"""
import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

VERIFIER_MODEL = "claude-haiku-4-5"

VERIFIER_SYSTEM_PROMPT = """You are a grounding verifier. Check whether an AI-generated 
answer is fully supported by source documents.

Process:
1. Identify factual claims in the answer.
2. For each, decide if it is supported by the sources.
3. Ignore generic disclaimers like "I don't have that information" — always valid.
4. Keep reasoning brief (one sentence per claim, max).

Output ONLY valid JSON, no markdown, no extra text:
{
  "verdict": "PASS" or "FAIL",
  "claims": [
    {"claim": "brief claim", "supported": true, "reasoning": "one sentence"}
  ],
  "summary": "one sentence overall"
}

PASS = every factual claim is supported.
FAIL = at least one claim is unsupported."""


def verify(answer: str, sources: list) -> dict:
    """Check if an answer is grounded in the provided sources."""
    source_text = "\n\n".join([
        f"[Source: {s['source']}]\n{s['text']}"
        for s in sources
    ])
    
    user_message = f"""Sources:

{source_text}

---

Answer to verify:

{answer}

---

Verify whether every factual claim in the answer is supported by the sources. Return JSON only."""
    
    response = client.messages.create(
        model=VERIFIER_MODEL,
        max_tokens=2048,
        system=VERIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    
    raw = response.content[0].text.strip()
    
    # Try to parse the JSON response
    try:
        # Strip code fences if Claude added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        # JSON broke — usually truncation. Try heuristic: scan for verdict keyword.
        raw_lower = raw.lower()
        if '"verdict": "pass"' in raw_lower or "'verdict': 'pass'" in raw_lower:
            result = {
                "verdict": "PASS",
                "claims": [],
                "summary": "Verifier reached PASS verdict but full JSON could not be parsed.",
                "raw": raw,
                "json_error": str(e),
            }
        else:
            result = {
                "verdict": "FAIL",
                "claims": [],
                "summary": "Verifier output could not be parsed and no PASS verdict found.",
                "raw": raw,
                "json_error": str(e),
            }
    
    result["tokens"] = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    }
    return result


# Quick self-test if you run this file directly
if __name__ == "__main__":
    test_sources = [
        {
            "source": "test.md",
            "text": "Claude Sonnet is balanced for capability and speed. Haiku is the fastest.",
        }
    ]
    
    print("Test 1: grounded answer")
    result = verify(
        answer="Claude Sonnet balances capability and speed.",
        sources=test_sources,
    )
    print(json.dumps(result, indent=2))
    print()
    
    print("Test 2: hallucinated answer")
    result = verify(
        answer="Claude Sonnet costs $50 per million tokens.",
        sources=test_sources,
    )
    print(json.dumps(result, indent=2))