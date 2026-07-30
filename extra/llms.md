# LLM Judge Candidates

Prices were fetched from OpenRouter's `/api/v1/models` endpoint on `2026-07-29`.

All models listed here advertised `tools` support in OpenRouter, which matters because the current judge pipeline uses a forced `submit_score` tool call.

Cost estimate basis:

- Full evaluation calls for one extra judge: `62,658`
- Estimated input tokens: `32,447,138`
- Min output estimate: `375,948` tokens, assuming about `6` output tokens per call
- Max output estimate: `8,020,224` tokens, assuming every call uses the full `128` output-token cap
- Avg is the midpoint between min and max until we have real pilot-run usage for a model

Format:

- `model id`: `$input / $output per 1M tokens`, `min / avg / max full eval`

## 1. Family: OpenAI

- `openai/gpt-5.6-sol`: `$5 / $30`, `$173.51 / $288.18 / $402.84`
- `openai/gpt-5.6-terra`: `$1.25 / $7.5`, `$43.38 / $72.04 / $100.71`
- `openai/gpt-5.6-luna`: `$0.5 / $3`, `$17.35 / $28.82 / $40.28`
- `openai/gpt-5.5`: `$5 / $30`, `$173.51 / $288.18 / $402.84`
- `openai/gpt-5.4`: `$2.5 / $15`, `$86.76 / $144.09 / $201.42`
- `openai/gpt-5.4-mini`: `$0.75 / $4.5`, `$26.03 / $43.23 / $60.43`
- `openai/gpt-5.4-nano`: `$0.2 / $1.25`, `$6.96 / $11.74 / $16.51`
- `openai/gpt-5.1`: `$1.25 / $10`, `$44.32 / $82.54 / $120.76`

## 2. Family: Anthropic

- `anthropic/claude-opus-5`: `$5 / $25`, `$171.63 / $267.19 / $362.74`
- `anthropic/claude-sonnet-5`: `$2 / $10`, `$68.65 / $106.88 / $145.10`
- `anthropic/claude-sonnet-4.6`: `$3 / $15`, `$102.98 / $160.31 / $217.64`
- `anthropic/claude-haiku-4.5`: `$1 / $5`, `$34.33 / $53.44 / $72.55`
- `anthropic/claude-3-haiku`: `$0.25 / $1.25`, `$8.58 / $13.36 / $18.14`

## 3. Family: Google

- `google/gemini-3.6-flash`: `$1.5 / $7.5`, `$51.49 / $80.16 / $108.82`
- `google/gemini-3.5-flash`: `$1.5 / $9`, `$52.05 / $86.45 / $120.85`
- `google/gemini-3.5-flash-lite`: `$0.3 / $2.5`, `$10.67 / $20.23 / $29.78`
- `google/gemini-3.1-pro-preview`: `$2 / $12`, `$69.41 / $115.27 / $161.14`
- `google/gemini-2.5-pro`: `$1.25 / $10`, `$44.32 / $82.54 / $120.76`
- `google/gemma-4-31b-it`: `$0.14 / $0.4`, `$4.69 / $6.22 / $7.75`
- `google/gemma-3-27b-it`: `$0.08 / $0.45`, `$2.76 / $4.48 / $6.20`

## 4. Family: Meta Llama

- `meta-llama/llama-4-maverick`: `$0.2 / $0.8`, `$6.79 / $9.85 / $12.91`
- `meta-llama/llama-4-scout`: `$0.1 / $0.3`, `$3.36 / $4.50 / $5.65`
- `meta-llama/llama-3.3-70b-instruct`: `$0.13 / $0.4`, `$4.37 / $5.90 / $7.43`
- `meta-llama/llama-3.1-8b-instruct`: `$0.05 / $0.08`, `$1.65 / $1.96 / $2.26`

## 6. Family: Qwen

- `qwen/qwen3-235b-a22b-2507`: `$0.09 / $0.55`, `$3.13 / $5.23 / $7.33`
- `qwen/qwen3-235b-a22b-thinking-2507`: `$0.3 / $3`, `$10.86 / $22.33 / $33.79`
- `qwen/qwen3.7-max`: `$1.475 / $4.425`, `$49.52 / $66.44 / $83.35`
- `qwen/qwen3.7-plus`: `$0.32 / $1.28`, `$10.86 / $15.76 / $20.65`
- `qwen/qwen3.7-flash`: `$0.03 / $0.13`, `$1.02 / $1.52 / $2.02`
- `qwen/qwen3.5-35b-a3b`: `$0.14 / $1`, `$4.92 / $8.74 / $12.56`
- `qwen/qwen3.5-397b-a17b`: `$0.39 / $2.34`, `$13.53 / $22.48 / $31.42`

## 7. Family: DeepSeek

- `deepseek/deepseek-v4-pro`: `$0.435 / $0.87`, `$14.44 / $17.77 / $21.09`
- `deepseek/deepseek-v4-flash`: `$0.14 / $0.28`, `$4.65 / $5.72 / $6.79`
- `deepseek/deepseek-v3.2`: `$0.269 / $0.4`, `$8.88 / $10.41 / $11.94`
- `deepseek/deepseek-r1-0528`: `$0.5 / $2.15`, `$17.03 / $25.25 / $33.47`
- `deepseek/deepseek-chat`: `$0.2002 / $0.8001`, `$6.80 / $9.85 / $12.91`



## 24. Family: OpenAI Extra

- `openai/gpt-5.6-sol-pro`: `$5 / $30`, `$173.51 / $288.18 / $402.84`
- `openai/gpt-5.6-terra-pro`: `$1.25 / $7.5`, `$43.38 / $72.04 / $100.71`
- `openai/gpt-5.6-luna-pro`: `$0.5 / $3`, `$17.35 / $28.82 / $40.28`
- `openai/gpt-chat-latest`: `$5 / $30`, `$173.51 / $288.18 / $402.84`
- `openai/gpt-5.5-pro`: `$30 / $180`, `$1041.08 / $1729.07 / $2417.05`
- `openai/gpt-5.4-pro`: `$30 / $180`, `$1041.08 / $1729.07 / $2417.05`
- `openai/gpt-5.3-chat`: `$1.75 / $14`, `$62.05 / $115.56 / $169.07`
- `openai/gpt-5.2`: `$1.75 / $14`, `$62.05 / $115.56 / $169.07`
- `openai/gpt-5.2-chat`: `$1.75 / $14`, `$62.05 / $115.56 / $169.07`
- `openai/gpt-5.2-pro`: `$21 / $168`, `$744.55 / $1386.67 / $2028.79`
- `openai/gpt-5.1-chat`: `$1.25 / $10`, `$44.32 / $82.54 / $120.76`
- `openai/gpt-5-mini`: `$0.25 / $2`, `$8.86 / $16.51 / $24.15`
- `openai/gpt-5-nano`: `$0.05 / $0.4`, `$1.77 / $3.30 / $4.83`

## 25. Family: Anthropic Extra

- `anthropic/claude-opus-5-fast`: `$10 / $50`, `$343.27 / $534.38 / $725.48`
- `anthropic/claude-fable-5`: `$10 / $50`, `$343.27 / $534.38 / $725.48`
- `anthropic/claude-opus-4.8`: `$5 / $25`, `$171.63 / $267.19 / $362.74`
- `anthropic/claude-opus-4.7`: `$5 / $25`, `$171.63 / $267.19 / $362.74`
- `anthropic/claude-opus-4.6`: `$5 / $25`, `$171.63 / $267.19 / $362.74`
- `anthropic/claude-opus-4.5`: `$5 / $25`, `$171.63 / $267.19 / $362.74`
- `anthropic/claude-sonnet-4.5`: `$3 / $15`, `$102.98 / $160.31 / $217.64`
- `anthropic/claude-opus-4.1`: `$15 / $75`, `$514.90 / $801.56 / $1088.22`
- `anthropic/claude-sonnet-4`: `$3 / $15`, `$102.98 / $160.31 / $217.64`

## 26. Family: Google Extra

- `google/gemini-3.6-flash:batch`: `$0.75 / $3.75`, `$25.75 / $40.08 / $54.41`
- `google/gemini-3.5-flash-lite:batch`: `$0.15 / $1.25`, `$5.34 / $10.11 / $14.89`
- `google/gemini-3.1-flash-lite`: `$0.25 / $1.5`, `$8.68 / $14.41 / $20.14`
- `google/gemini-3.1-pro-preview:batch`: `$1 / $6`, `$34.70 / $57.64 / $80.57`
- `google/gemini-3-flash-preview`: `$0.5 / $3`, `$17.35 / $28.82 / $40.28`
- `google/gemini-2.5-flash`: `$0.3 / $2.5`, `$10.67 / $20.23 / $29.78`
- `google/gemini-2.5-flash-lite`: `$0.1 / $0.4`, `$3.40 / $4.92 / $6.45`
- `google/gemma-4-26b-a4b-it`: `$0.07 / $0.34`, `$2.40 / $3.70 / $5.00`
- `google/gemma-3-12b-it`: `$0.05 / $0.15`, `$1.68 / $2.25 / $2.83`

## 27. Family: Qwen Extra

- `qwen/qwen3.6-flash`: `$0.1875 / $1.125`, `$6.51 / $10.81 / $15.11`
- `qwen/qwen3.6-35b-a3b`: `$0.14 / $1`, `$4.92 / $8.74 / $12.56`
- `qwen/qwen3.6-max-preview`: `$1.027 / $6.162`, `$35.64 / $59.19 / $82.74`
- `qwen/qwen3.6-27b`: `$0.3 / $2`, `$10.49 / $18.13 / $25.77`
- `qwen/qwen3.6-plus`: `$0.325 / $1.95`, `$11.28 / $18.73 / $26.18`
- `qwen/qwen3.5-122b-a10b`: `$0.26 / $2.08`, `$9.22 / $17.17 / $25.12`
- `qwen/qwen3.5-flash-02-23`: `$0.065 / $0.26`, `$2.21 / $3.20 / $4.19`
- `qwen/qwen3.5-plus-20260420`: `$0.3 / $1.8`, `$10.41 / $17.29 / $24.17`
- `qwen/qwen3-max`: `$0.78 / $3.9`, `$26.77 / $41.68 / $56.59`
- `qwen/qwen3-next-80b-a3b-instruct`: `$0.1 / $1.1`, `$3.66 / $7.86 / $12.07`
- `qwen/qwen3-30b-a3b-instruct-2507`: `$0.04815 / $0.193`, `$1.63 / $2.37 / $3.11`

## Shortlist For A Multi-Judge Study

- Strong expensive judges: `anthropic/claude-sonnet-5`, `openai/gpt-5.4`, `google/gemini-3.1-pro-preview`
- Strong affordable judges: `qwen/qwen3-235b-a22b-2507`, `deepseek/deepseek-v4-pro`, `google/gemini-2.5-pro`
- Cheap diversity judges: `qwen/qwen3.7-flash`, `meta-llama/llama-4-scout`, `google/gemma-3-27b-it`, `z-ai/glm-4.7-flash`
- Very cheap extra probes: `mistralai/mistral-nemo`, `inclusionai/ling-2.6-flash`, `nex-agi/nex-n2-mini`, `amazon/nova-micro-v1`

## Runtime Estimates

Runtime depends much more on provider latency and rate limits than on token price. The current full judge workload is `62,658` calls. The repo's full-judge orchestrator defaults to `4` workers, so practical wall time is roughly serial time divided by `4`, assuming the provider allows the concurrency.

Known calibration anchors from `evaluation/judge_choice.md`:

| Latency Band | Anchor Judge | Avg Sec / Call | Serial Full Run | 4 Workers | 8 Workers |
|---|---|---:|---:|---:|---:|
| Fast | `qwen/qwen3.5-35b-a3b` | `0.90s` | `15.72h` | `3.93h` | `1.97h` |
| Medium | `qwen/qwen3-235b-a22b-2507` | `1.92s` | `33.36h` | `8.34h` | `4.17h` |
| Slow | `anthropic/claude-sonnet-4.6` | `3.35s` | `58.21h` | `14.55h` | `7.28h` |

Planning estimates:

- Cheap/small models: usually plan around the fast band, about `4h` with `4` workers.
- Large non-reasoning models: usually plan around the medium band, about `8-9h` with `4` workers.
- Claude/OpenAI/Gemini/Grok high-end judges: plan around the slow band until measured, about `14-15h` with `4` workers.
- Thinking/reasoning variants can be slower than the slow band if reasoning is not fully disabled.

Recommended pilot before a full run:

- Run `50` rows x all applicable metrics for each candidate judge.
- Compute observed average seconds per call.
- Full-run wall estimate with `4` workers:

```text
wall_hours = (62658 * avg_seconds_per_call) / (4 * 3600)
```

## Cost Formula

For a different model, estimate one full extra judge run as:

```text
min = (32.447138 * input_price_per_1M) + (0.375948 * output_price_per_1M)
max = (32.447138 * input_price_per_1M) + (8.020224 * output_price_per_1M)
avg = (min + max) / 2
```
