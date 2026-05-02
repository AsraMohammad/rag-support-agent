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

# Use the cheaper model for verification — it's a structured task, no creativity needed
VERIFIER_MODEL = "claude-haiku-4-5"

VERIFIER_SYSTEM_PROMPT = """You are a grounding verifier. Your job is to check whether 
an AI-generated answer is fully supported by a set of source documents.

Process:
1. Identify every factual claim in the answer.
2. For each claim, decide if it is supported by the sources (verbatim or by reasonable paraphrase).
3. Ignore generic disclaimers like "I don't have that information" — those are always valid.
4. Return your verdict as JSON.

Output format (return ONLY valid JSON, no markdown fences, no extra text):
{
  "verdict": "PASS" or "FAIL",
  "claims": [
    {"claim": "...", "supported": true or false, "reasoning": "..."}
  ],
  "summary": "..."
}

PASS = every factual claim is supported.
FAIL = at least one claim is not supported (i.e. potential hallucination)."""


def verify(answer: str, sources: list) -> dict:
    """
    Check if an answer is grounded in the provided sources.
    
    Args:
        answer: the generated answer text
        sources: list of dicts with 'text' and 'source' keys
    
    Returns:
        dict with 'verdict', 'claims', 'summary'
    """
    # Format the sources for the verifier
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
        max_tokens=1024,
        system=VERIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    
    raw = response.content[0].text.strip()
    
    # Try to parse the JSON response
    try:
        # Strip code fences if Claude added them despite our instruction
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        # If parsing fails, return a "FAIL" verdict and the raw text for debugging
        result = {
            "verdict": "FAIL",
            "claims": [],
            "summary": f"Verifier returned invalid JSON: {e}",
            "raw": raw,
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
    
    # Test 1: a grounded answer (should PASS)
    print("Test 1: grounded answer")
    result = verify(
        answer="Claude Sonnet balances capability and speed.",
        sources=test_sources,
    )
    print(json.dumps(result, indent=2))
    print()
    
    # Test 2: a hallucinated answer (should FAIL)
    print("Test 2: hallucinated answer")
    result = verify(
        answer="Claude Sonnet costs $50 per million tokens.",
        sources=test_sources,
    )
    print(json.dumps(result, indent=2))