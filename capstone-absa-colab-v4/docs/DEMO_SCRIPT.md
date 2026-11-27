# Demo video script (target 5 minutes, hard limit 6)

Record last, after the numbers are final. Screen capture with voice over; show the terminal, the notebook and the Streamlit page.

| Time | Show | Say |
|---|---|---|
| 0:00 | Title slide or README top | The problem: star ratings hide which features drive satisfaction. ABSA extracts aspects and sentiment, but models trained on one category degrade on another. |
| 0:30 | `docs/architecture.png` | Walk the pipeline: review, extraction (baseline, prompted LLM, QLoRA LLM), validation, aggregation, report. The cross-domain layer is the weeks 9 to 11 extension. |
| 1:15 | `notebooks/01_eda.ipynb` figures | Source data (laptops, restaurants) vs clothing target: sentiment skew, noun overlap, the 22 of 100 clothing nouns never seen in the source. |
| 2:00 | Terminal: `python -m src.demo --methods baseline prompted finetuned` | Same three reviews through all three extractors. Point out misses by the baseline, JSON validity and the reason spans of the LLMs. |
| 3:00 | `python -m src.compare` / `results/comparison.md` | Source test vs clothing F1, the cross-domain drop for each extractor, JSON validity, seconds per review. |
| 4:00 | `streamlit run app.py` | The per-aspect net sentiment report for clothing. Show an aspect with strongly negative sentiment (for example fit or size) and what a product manager does with it. |
| 4:45 | README status table | What is done, what is next (cross-domain layer, full gold set, final report). |

Tips: pre-run slow commands and show saved output, keep terminal font large, no raw review text beyond the three demo sentences, rehearse once and cut to under 6 minutes.
