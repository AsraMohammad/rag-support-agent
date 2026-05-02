# Anthropic API Pricing

Anthropic charges for API usage based on the number of tokens processed. A token is roughly 3-4 characters of text, so 1,000 tokens equals about 750 words.

Pricing varies by model. Claude Opus is the most expensive, optimized for complex reasoning. Claude Sonnet offers a balance of capability and cost. Claude Haiku is the cheapest option, ideal for high-volume tasks like classification, extraction, and simple Q&A.

Input tokens (the prompt you send) are billed at a different rate than output tokens (the response). Output tokens are typically more expensive because generation is more compute-intensive than reading.

Anthropic offers prompt caching, which can reduce costs by up to 90% for repeated context. When you cache a prompt, subsequent calls that reuse the same prefix are billed at a lower rate.

Batch API requests can save 50% on cost. The batch API is suitable for non-urgent workloads where results within 24 hours is acceptable.

Free tier credits are available for new accounts. Usage limits scale based on your spending tier — accounts in higher tiers have higher rate limits and access to additional features.