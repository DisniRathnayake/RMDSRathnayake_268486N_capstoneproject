| Method | Source test pair F1 (span) | Clothing pair F1 (category) | Clothing aspect F1 (category) | Clothing pair F1 (category, strict all reviews) | Clothing pair F1 (span, literal gold) | Abs. drop (span) | Rel. drop (span) | Net-sentiment corr. (Spearman) | Scored / gold reviews | JSON validity (clothing sentences) | s/review (clothing) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.416 | 0.176 | 0.204 | 0.176 | 0.176 | 0.240 | 57.6% | -0.043 | 300 / 300 | 1.000 | 0.02 |
| baseline_lr | 0.384 | 0.158 | 0.204 | 0.158 | 0.160 | 0.224 | 58.3% | -0.153 | 300 / 300 | 1.000 | 0.02 |
| prompted | n/a | 0.291 | 0.365 | 0.278 | 0.195 | n/a | n/a | 0.718 | 225 / 300 | 0.926 | 16.10 |
| finetuned | 0.751 | 0.394 | 0.445 | 0.391 | 0.355 | 0.396 | 52.8% | 0.600 | 297 / 300 | 0.998 | 11.83 |

Gold aspects that occur literally in the review: 52.3% (whole word 39.9%). The span-level column can only score that share; the category-level columns score all gold aspects.
