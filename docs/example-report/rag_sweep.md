# RAG sweep: rag-mini-sweep

Generator `mock/mock-rag`, verifier `lexical`, 25 questions, 96 settings.

## Top 8 settings by `pass`

| setting | pass | answer_f1 | context_recall | mrr | ndcg_at_k | faithfulness | abstention_correct |
|---|---|---|---|---|---|---|---|
| `fixed/700/80 bm25 k=2 rerank=none emb=hash` | 0.960 [0.88, 1.00] | 1.000 [1.00, 1.00] | 0.950 [0.88, 1.00] | 0.925 [0.85, 1.00] | 0.906 [0.83, 0.96] | 0.762 [0.57, 0.95] | 0.960 [0.88, 1.00] |
| `fixed/700/80 hybrid k=2 rerank=none emb=hash` | 0.960 [0.88, 1.00] | 0.950 [0.85, 1.00] | 0.863 [0.71, 0.97] | 0.875 [0.75, 0.97] | 0.844 [0.71, 0.95] | 0.800 [0.60, 0.95] | 1.000 [1.00, 1.00] |
| `fixed/700/80 vector k=5 rerank=none emb=hash` | 0.960 [0.88, 1.00] | 0.950 [0.85, 1.00] | 0.938 [0.85, 1.00] | 0.672 [0.53, 0.81] | 0.697 [0.58, 0.80] | 0.850 [0.65, 1.00] | 1.000 [1.00, 1.00] |
| `recursive/700/80 hybrid k=5 rerank=none emb=hash` | 0.960 [0.88, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.778 [0.64, 0.90] | 0.798 [0.69, 0.90] | 0.857 [0.71, 1.00] | 0.960 [0.88, 1.00] |
| `fixed/700/80 bm25 k=5 rerank=none emb=hash` | 0.920 [0.80, 1.00] | 1.000 [1.00, 1.00] | 0.975 [0.93, 1.00] | 0.925 [0.82, 1.00] | 0.893 [0.82, 0.96] | 0.818 [0.64, 0.95] | 0.920 [0.80, 1.00] |
| `recursive/700/0 bm25 k=5 rerank=none emb=hash` | 0.920 [0.80, 1.00] | 0.950 [0.85, 1.00] | 0.975 [0.93, 1.00] | 0.927 [0.81, 1.00] | 0.893 [0.80, 0.98] | 0.857 [0.71, 1.00] | 0.960 [0.88, 1.00] |
| `fixed/250/80 bm25 k=5 rerank=none emb=hash` | 0.880 [0.72, 1.00] | 0.990 [0.97, 1.00] | 0.908 [0.79, 1.00] | 0.804 [0.65, 0.93] | 0.824 [0.70, 0.93] | 0.773 [0.59, 0.95] | 0.920 [0.80, 1.00] |
| `fixed/250/80 vector k=5 rerank=none emb=hash` | 0.880 [0.76, 1.00] | 0.875 [0.72, 1.00] | 0.796 [0.63, 0.93] | 0.683 [0.52, 0.84] | 0.687 [0.53, 0.83] | 0.778 [0.61, 0.94] | 0.920 [0.80, 1.00] |

## Main effects (mean metric at each factor level)

### chunk_strategy

| metric | fixed | recursive | spread |
|---|---|---|---|
| context_precision | 0.303 | 0.315 | 0.012 |
| context_recall | 0.723 | 0.746 | 0.023 |
| hit_at_k | 0.790 | 0.800 | 0.010 |
| mrr | 0.662 | 0.689 | 0.027 |
| ndcg_at_k | 0.655 | 0.680 | 0.025 |
| faithfulness | 0.718 | 0.712 | 0.006 |
| hallucination | 0.282 | 0.288 | 0.006 |
| answer_relevance | 0.092 | 0.092 | 0.000 |
| citation_accuracy | 0.590 | 0.649 | 0.060 |
| abstention_correct | 0.726 | 0.809 | 0.083 |
| answer_f1 | 0.674 | 0.751 | 0.077 |
| pass | 0.670 | 0.728 | 0.058 |

### chunk_size

| metric | 250 | 700 | spread |
|---|---|---|---|
| context_precision | 0.314 | 0.304 | 0.010 |
| context_recall | 0.716 | 0.752 | 0.036 |
| hit_at_k | 0.790 | 0.800 | 0.010 |
| mrr | 0.668 | 0.682 | 0.014 |
| ndcg_at_k | 0.659 | 0.675 | 0.016 |
| faithfulness | 0.686 | 0.743 | 0.057 |
| hallucination | 0.314 | 0.257 | 0.057 |
| answer_relevance | 0.091 | 0.094 | 0.002 |
| citation_accuracy | 0.598 | 0.640 | 0.042 |
| abstention_correct | 0.739 | 0.796 | 0.057 |
| answer_f1 | 0.686 | 0.738 | 0.052 |
| pass | 0.667 | 0.731 | 0.063 |

### chunk_overlap

| metric | 0 | 80 | spread |
|---|---|---|---|
| context_precision | 0.296 | 0.322 | 0.026 |
| context_recall | 0.734 | 0.734 | 0.000 |
| hit_at_k | 0.794 | 0.796 | 0.002 |
| mrr | 0.683 | 0.667 | 0.016 |
| ndcg_at_k | 0.671 | 0.663 | 0.007 |
| faithfulness | 0.693 | 0.736 | 0.044 |
| hallucination | 0.307 | 0.264 | 0.044 |
| answer_relevance | 0.092 | 0.093 | 0.001 |
| citation_accuracy | 0.604 | 0.635 | 0.030 |
| abstention_correct | 0.738 | 0.797 | 0.058 |
| answer_f1 | 0.657 | 0.768 | 0.111 |
| pass | 0.658 | 0.740 | 0.082 |

### top_k

| metric | 2 | 5 | spread |
|---|---|---|---|
| context_precision | 0.403 | 0.216 | 0.187 |
| context_recall | 0.648 | 0.821 | 0.173 |
| hit_at_k | 0.715 | 0.875 | 0.160 |
| mrr | 0.653 | 0.698 | 0.045 |
| ndcg_at_k | 0.633 | 0.701 | 0.068 |
| faithfulness | 0.647 | 0.783 | 0.136 |
| hallucination | 0.353 | 0.217 | 0.136 |
| answer_relevance | 0.093 | 0.092 | 0.000 |
| citation_accuracy | 0.589 | 0.650 | 0.062 |
| abstention_correct | 0.712 | 0.823 | 0.112 |
| answer_f1 | 0.626 | 0.799 | 0.174 |
| pass | 0.638 | 0.761 | 0.123 |

### retrieval

| metric | bm25 | hybrid | vector | spread |
|---|---|---|---|---|
| context_precision | 0.341 | 0.316 | 0.271 | 0.070 |
| context_recall | 0.790 | 0.749 | 0.664 | 0.125 |
| hit_at_k | 0.839 | 0.811 | 0.734 | 0.105 |
| mrr | 0.735 | 0.690 | 0.601 | 0.134 |
| ndcg_at_k | 0.724 | 0.684 | 0.594 | 0.131 |
| faithfulness | 0.720 | 0.734 | 0.690 | 0.044 |
| hallucination | 0.280 | 0.266 | 0.310 | 0.044 |
| answer_relevance | 0.096 | 0.092 | 0.089 | 0.007 |
| citation_accuracy | 0.624 | 0.629 | 0.605 | 0.025 |
| abstention_correct | 0.790 | 0.786 | 0.726 | 0.064 |
| answer_f1 | 0.735 | 0.741 | 0.661 | 0.081 |
| pass | 0.715 | 0.719 | 0.664 | 0.055 |

### reranker

| metric | llm | none | spread |
|---|---|---|---|
| context_precision | 0.271 | 0.348 | 0.077 |
| context_recall | 0.657 | 0.811 | 0.154 |
| hit_at_k | 0.719 | 0.871 | 0.152 |
| mrr | 0.588 | 0.762 | 0.174 |
| ndcg_at_k | 0.586 | 0.748 | 0.162 |
| faithfulness | 0.706 | 0.724 | 0.018 |
| hallucination | 0.294 | 0.276 | 0.018 |
| answer_relevance | 0.096 | 0.088 | 0.008 |
| citation_accuracy | 0.609 | 0.630 | 0.022 |
| abstention_correct | 0.704 | 0.831 | 0.127 |
| answer_f1 | 0.634 | 0.791 | 0.157 |
| pass | 0.638 | 0.761 | 0.123 |

## Best setting per metric

- **context_precision**: `fixed/700/0 bm25 k=2 rerank=none emb=hash`
- **context_recall**: `recursive/700/80 hybrid k=5 rerank=none emb=hash`
- **hit_at_k**: `fixed/700/0 bm25 k=2 rerank=none emb=hash`
- **mrr**: `recursive/700/0 bm25 k=5 rerank=none emb=hash`
- **ndcg_at_k**: `fixed/700/0 bm25 k=2 rerank=none emb=hash`
- **faithfulness**: `recursive/700/0 hybrid k=5 rerank=llm emb=hash`
- **hallucination**: `recursive/700/0 hybrid k=5 rerank=llm emb=hash`
- **answer_relevance**: `fixed/700/80 hybrid k=2 rerank=llm emb=hash`
- **citation_accuracy**: `recursive/700/0 hybrid k=5 rerank=llm emb=hash`
- **abstention_correct**: `fixed/700/80 hybrid k=2 rerank=none emb=hash`
- **answer_f1**: `fixed/700/80 bm25 k=2 rerank=none emb=hash`
- **pass**: `fixed/700/80 bm25 k=2 rerank=none emb=hash`
