# Claude API Best Practices

## Use system prompts effectively

The system prompt sets the persona, context, and rules for Claude. Place instructions, role definitions, and formatting requirements in the system prompt rather than the user message. This separation makes prompts more maintainable.

## Provide examples

Few-shot prompting — showing Claude examples of desired input/output pairs — significantly improves consistency. Two or three high-quality examples are usually more effective than long instructions.

## Handle errors gracefully

API calls can fail for many reasons: network issues, rate limits, invalid requests, or service outages. Production code should wrap API calls in try/except blocks and implement exponential backoff for retries.

## Monitor token usage

Every response includes a usage object with input_tokens and output_tokens. Log these values to track cost and identify expensive prompts. Setting max_tokens prevents runaway responses.

## Use the right model

Don't default to the most powerful model. Many tasks — classification, simple extraction, formatting — work fine on Haiku at a fraction of the cost. Reserve Opus and Sonnet for tasks that genuinely need their reasoning power.

## Streaming for user-facing apps

For chat applications, use streaming responses so users see output as it's generated rather than waiting for the full response. This dramatically improves perceived latency.

## Validate structured output

When asking Claude for JSON or other structured output, validate the response with a schema (e.g. Pydantic in Python). LLMs occasionally produce malformed structures; validation catches this before downstream code breaks.