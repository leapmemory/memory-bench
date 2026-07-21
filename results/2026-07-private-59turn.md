# Run: private 59-turn corpus (July 2026)

## Result

| Metric | Value |
|---|---|
| Probes | 10 |
| Accuracy, pasted history | 10/10 |
| Accuracy, LeapMemory recall | 10/10 |
| Scored (both correct) | 10 |
| Median input-token reduction | 82.6% |
| Range | 80.2% - 90.9% |
| Model, both conditions | claude-sonnet-4.6 via OpenRouter, temperature 0 |
| Token source | provider-billed prompt_tokens |

Per-probe billed input tokens (pasted vs recall): 25,426/4,840 - 25,420/4,436 -
25,421/4,517 - 25,414/3,446 - 25,423/5,033 - 25,427/4,052 - 25,422/4,430 -
25,414/2,308 - 25,418/4,856 - 25,422/4,391.

## Corpus

One real conversation: 59 messages of dense technical work between a developer
and an AI assistant, about 25k tokens. The corpus and the probe questions
reference private project internals and are not published. The public
`corpus/sample_100.jsonl` exists so anyone can verify the same harness on
open data.

## Method

`bench.py`, two conditions, same model, same system prompt, temperature 0.
Baseline pastes the full corpus before every question; recall places only the
LeapMemory recall result. A probe counts toward the reduction number only when
BOTH conditions answered correctly.

## Honest notes

- An earlier run of this corpus scored recall 9/10. The miss was a real
  retrieval gap: the answer was a bare number sharing no vocabulary with the
  question, and the passage carrying it ranked below the retrieval cutoff.
  We fixed retrieval generally (facts carry provenance to their source
  passages, and recall pulls those passages into the ranking pool),
  re-ingested, and re-ran. No probe, prompt, or grading change was made.
- Accuracy on a single corpus is a mechanism check, not a scale claim.
  Larger and standardized evaluations (LongMemEval-class) are planned.
- Recall latency on this dense corpus runs tens of seconds on our current
  CPU reranker. Known, being worked on, unrelated to token counts.