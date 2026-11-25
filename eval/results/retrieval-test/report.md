# DIU Academic Assistant — evaluation report

Run: 2026-09-16T17:39:02+00:00 · question set `eval\questions.yaml` (sha256 bf76275f0843ce46) · split **test**

- Questions: **35** (22 answerable, 13 unanswerable)
- Human-reviewed: **0 of 35** — gold answers not yet reviewed by a person are provisional
- Embedding `BAAI/bge-small-en-v1.5@384` · reranker `cross-encoder/ms-marco-MiniLM-L6-v2` · relevance floor 0.05 · answer model `not run`
- Documents indexed: 4 · chunks: 164

Percentages show a Wilson 95% confidence interval in brackets. With this many questions the intervals are wide: differences whose intervals overlap may be noise.

## 1. Retrieval (no LLM)

Rank of the first passage containing the evidence, over answerable questions.

| Retrieval | Hit@1 | Hit@3 | Hit@5 | Evidence recall@5 | MRR | Right document in top 5 | Median time |
|---|---|---|---|---|---|---|---|
| dense | 54.5% (35–73) | 72.7% (52–87) | 72.7% (52–87) | 72.7% | 0.655 | 100.0% | 0.04s |
| sparse | 81.8% (61–93) | 86.4% (67–95) | 95.5% (78–99) | 95.5% | 0.867 | 95.5% | 0.00s |
| hybrid | 68.2% (47–84) | 86.4% (67–95) | 90.9% (72–97) | 88.6% | 0.784 | 100.0% | 0.05s |
| hybrid+rerank | 72.7% (52–87) | 77.3% (57–90) | 86.4% (67–95) | 84.1% | 0.788 | 100.0% | 3.91s |

## 2. Relevance floor (no LLM)

Questions whose best reranked passage scores below 0.05 are refused without calling the model.

| | Rate | Count |
|---|---|---|
| Answerable questions let through | 100.0% (85–100) | 22/22 |
| Unanswerable questions stopped | 53.8% (29–77) | 7/13 |

Unanswerable questions that pass the floor must be caught by the model's own refusal (`INSUFFICIENT_CONTEXT`), measured in section 3.

## 3. End to end

Not run (`--retrieval-only`, or no LLM configured).


## Definitions

- **Hit@k** — the evidence passage is among the top k retrieved. **MRR** — mean of 1/rank of the first evidence passage (0 if not in the 30 candidates). **Evidence recall@5** — share of an item's evidence strings found in the top 5 (matters for questions needing two passages).
- **Accuracy** — answerable questions answered with every gold fact stated. Facts list acceptable forms; times, dates, ordinals, punctuation and citation markers are normalised before matching.
- **Wrong or unsupported answers** — of all answers given (not refusals): answers missing gold facts, plus answers to unanswerable questions. This is the closest honest equivalent of the poster's "hallucination rate".
- **Correct refusals** — unanswerable questions the system declined. **False refusals** — answerable questions it declined.
- **Overall success** — correct answers plus correct refusals, over all questions.
- **LLM only** — the same model and question with no retrieved documents. A reply counts as a refusal when it says it does not know or lacks information.
